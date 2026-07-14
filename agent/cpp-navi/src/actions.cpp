#include "actions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

#include <MaaFramework/MaaAPI.h>
#include <MaaUtils/Logger.h>
#include <meojson/json.hpp>

#include "navigator.h"
#include "position_provider.h"
#include "route.h"
#include "util.h"
#include "websocket_service.h"

namespace navi
{

namespace
{

json::value parse_object(const char* value)
{
    if (value == nullptr || *value == '\0') {
        return json::object {};
    }
    const auto parsed = json::parse(value);
    if (!parsed || !parsed->is_object()) {
        throw std::invalid_argument("custom_action_param must be a JSON object");
    }
    return *parsed;
}

double number_option(const json::value& params, const char* key, double fallback)
{
    return params.contains(key) && params.at(key).is_number() ? params.at(key).as<double>() : fallback;
}

int integer_option(const json::value& params, const char* key, int fallback)
{
    return params.contains(key) && params.at(key).is_number() ? params.at(key).as<int>() : fallback;
}

std::string string_option(const json::value& params, const char* key, std::string fallback)
{
    return params.contains(key) && params.at(key).is_string() ? params.at(key).as<std::string>() : std::move(fallback);
}

bool bool_option(const json::value& params, const char* key, bool fallback)
{
    return params.contains(key) && params.at(key).is_boolean() ? params.at(key).as<bool>() : fallback;
}

std::string normalized(std::string value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    value = value.substr(begin, end - begin);
    std::ranges::transform(value, value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string trimmed_string_option(const json::value& params, const char* key, std::string fallback)
{
    std::string value = string_option(params, key, std::move(fallback));
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

double required_number(const json::value& value, const char* key)
{
    if (!value.is_object() || !value.contains(key) || !value.at(key).is_number()) {
        throw std::invalid_argument(std::string("missing numeric field: ") + key);
    }
    return value.at(key).as<double>();
}

std::pair<double, double> point_xy(const json::value& value)
{
    for (const auto& [x_key, y_key] : {
             std::pair { "worldX", "worldY" },
             std::pair { "rawX", "rawY" },
             std::pair { "pixelX", "pixelY" },
             std::pair { "x", "y" },
         }) {
        if (value.is_object() && value.contains(x_key) && value.contains(y_key)) {
            return { required_number(value, x_key), required_number(value, y_key) };
        }
    }
    if (value.is_object() && value.contains("coordinate") && value.at("coordinate").is_object()) {
        return { required_number(value.at("coordinate"), "x"), required_number(value.at("coordinate"), "y") };
    }
    throw std::invalid_argument("point needs worldX/worldY, pixelX/pixelY, or x/y");
}

std::optional<json::value> load_named_record(const json::value& params)
{
    const std::string point_id = trimmed_string_option(params, "point_id", "");
    if (point_id.empty()) {
        return std::nullopt;
    }
    const std::filesystem::path path = resource_base_path() / trimmed_string_option(params, "points_file", "map_teleport/check_points.json");
    const json::value data = load_json_file(path);
    const json::value* records = &data;
    if (data.is_object() && data.contains("points")) {
        records = &data.at("points");
    }
    if (!records->is_array()) {
        throw std::invalid_argument("points must be a list");
    }
    for (const auto& record : records->as_array()) {
        if (!record.is_object()) {
            continue;
        }
        const std::string id = trimmed_string_option(record, "id", "");
        const std::string name = trimmed_string_option(record, "name", "");
        if (id == point_id || name == point_id) {
            return record;
        }
    }
    throw std::invalid_argument("points record not found: " + point_id);
}

std::pair<double, double> target_point(const json::value& params, const std::optional<json::value>& record)
{
    if (record) {
        return point_xy(*record);
    }
    for (const char* key : { "target", "target_point" }) {
        if (!params.contains(key)) {
            continue;
        }
        const auto& target = params.at(key);
        if (target.is_array() && target.as_array().size() >= 2 && target.as_array()[0].is_number() && target.as_array()[1].is_number()) {
            return { target.as_array()[0].as<double>(), target.as_array()[1].as<double>() };
        }
        if (target.is_object()) {
            return point_xy(target);
        }
    }
    return { required_number(params, "target_x"), required_number(params, "target_y") };
}

std::optional<LocationResult> visual_location(MaaContext* context, bool debug)
{
    cv::Mat frame;
    MaaController* controller = MaaTaskerGetController(MaaContextGetTasker(context));
    if (!capture_frame(controller, frame)) {
        return std::nullopt;
    }
    PositionProvider visual("map", debug);
    LocationResult result = visual.locate(frame);
    return result.found ? std::optional<LocationResult>(std::move(result)) : std::nullopt;
}

std::optional<LocationResult> locate_current_position(
    MaaContext* context,
    std::string backend,
    double timeout,
    double interval,
    bool debug)
{
    backend = normalized(std::move(backend));
    PositionProvider provider(backend, debug);
    if (provider.uses_visual_positioning()) {
        cv::Mat frame;
        MaaController* controller = MaaTaskerGetController(MaaContextGetTasker(context));
        if (!capture_frame(controller, frame)) {
            return std::nullopt;
        }
        LocationResult result = provider.locate(frame);
        return result.found ? std::optional<LocationResult>(std::move(result)) : std::nullopt;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(std::max(timeout, 0.0));
    while (!task_stopping(context)) {
        LocationResult result = provider.locate();
        if (result.found && result.point) {
            return result;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        sleep_interruptible(context, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(std::max(interval, 0.05))));
    }
    return backend == "auto" && !task_stopping(context) ? visual_location(context, debug) : std::nullopt;
}

bool run_context_action(MaaContext* context, const char* entry, const std::string& pipeline_override = "{}")
{
    const MaaRect box {};
    const MaaActId action_id = MaaContextRunAction(context, entry, pipeline_override.c_str(), &box, "");
    if (action_id == MaaInvalidId) {
        return false;
    }
    StringBuffer node_name;
    StringBuffer action;
    StringBuffer detail;
    MaaRect result_box {};
    MaaBool success = 0;
    return MaaTaskerGetActionDetail(
               MaaContextGetTasker(context), action_id, node_name.get(), action.get(), &result_box, &success, detail.get())
           && success;
}

void show_teleport_message(MaaContext* context, bool need_teleport)
{
    const char* node = need_teleport ? "__CheckTeleportRequiredFarMessage" : "__CheckTeleportRequiredNearMessage";
    if (!run_context_action(context, node)) {
        LogWarn << "CheckTeleportRequired message action failed" << VAR(node);
    }
}

bool run_map_teleport(MaaContext* context, const std::string& point_id, const std::string& points_file)
{
    json::object teleport_params {
        { "teleport_point_id", point_id },
        { "teleport_points_file", points_file },
    };
    json::object action_param {
        { "custom_action", "map_teleport_to_point" },
        { "custom_action_param", std::move(teleport_params) },
    };
    json::object action {
        { "type", "Custom" },
        { "param", std::move(action_param) },
    };
    json::object node {
        { "action", std::move(action) },
        { "pre_delay", 0 },
        { "post_delay", 0 },
        { "rate_limit", 0 },
    };
    json::object pipeline_override {
        { "__CheckTeleportRequiredRunTeleport", std::move(node) },
    };
    return run_context_action(context, "__CheckTeleportRequiredRunTeleport", pipeline_override.dumps());
}

json::value load_attach(MaaContext* context, const char* node_name)
{
    StringBuffer buffer;
    if (!MaaContextGetNodeData(context, node_name, buffer.get())) {
        return json::object {};
    }
    const auto parsed = json::parse(buffer.str());
    if (!parsed || !parsed->is_object() || !parsed->contains("attach") || !parsed->at("attach").is_object()) {
        return json::object {};
    }
    return parsed->at("attach");
}

NavigationOptions navigation_options(const json::value& params, std::string default_position_backend)
{
    NavigationOptions options;
    options.tolerance = number_option(params, "tolerance", 5.0);
    options.frame_interval = std::max(0.05, number_option(params, "frame_interval", 0.1));
    options.angle_backend = string_option(params, "angle_backend", "auto");
    options.position_backend = string_option(params, "position_backend", std::move(default_position_backend));
    options.debug = bool_option(params, "debug", false);
    return options;
}

bool run_local_route(MaaContext* context, const json::value& params, const std::filesystem::path& default_path, std::string default_route)
{
    const std::string json_path = string_option(params, "json_path", default_path.string());
    if (json_path.empty()) {
        throw std::invalid_argument("json_path is required");
    }
    const std::string route_name = string_option(params, "route_name", std::move(default_route));
    const int segment_index = integer_option(params, "segment_index", 1);
    RouteSession route;
    RouteRunner runner(context, route, navigation_options(params, "map"));
    const auto data = load_json_file(resolve_route_json_path(json_path));
    const auto waypoints = parse_route_segment(data, route_name, segment_index, CoordinateTransform::MapSize, runner.source_size());
    if (waypoints.empty()) {
        LogWarn << "LocalRouteNavigation route is empty" << VAR(json_path) << VAR(route_name) << VAR(segment_index);
        return false;
    }
    const auto current = runner.update_current_frame();
    route.reset(waypoints, true, current);
    LogInfo << "LocalRouteNavigation route loaded" << VAR(json_path) << VAR(route_name) << VAR(segment_index) << VAR(waypoints.size());
    return runner.run_until_stopped({}, true) == "arrived";
}

} // namespace

MaaBool check_teleport_required(
    MaaContext* context,
    MaaTaskId,
    const char*,
    const char*,
    const char* custom_action_param,
    MaaRecoId,
    const MaaRect*,
    void*)
{
    try {
        const json::value params = parse_object(custom_action_param);
        const auto record = load_named_record(params);
        const auto target = target_point(params, record);
        const double threshold = number_option(params, "threshold", record ? number_option(*record, "threshold", 200.0) : 200.0);
        std::string backend = trimmed_string_option(params, "position_backend", "auto");
        if (backend.empty()) {
            backend = "auto";
        }
        std::string coordinate_type = trimmed_string_option(params, "coordinate_type", "world");
        if (coordinate_type.empty()) {
            coordinate_type = "world";
        }
        coordinate_type = normalized(std::move(coordinate_type));
        const double timeout = number_option(params, "coordinate_timeout", 1.5);
        const double interval = number_option(params, "coordinate_interval", 0.1);
        const bool debug = bool_option(params, "debug", false);

        const auto location = locate_current_position(context, backend, timeout, interval, debug);
        if (!location || !location->found || !location->point) {
            LogWarn << "CheckTeleportRequired position not found" << VAR(backend);
            return 0;
        }

        std::pair<double, double> current;
        if (coordinate_type == "map" || coordinate_type == "pixel" || coordinate_type == "image") {
            current = { location->point->x, location->point->y };
        }
        else if (coordinate_type == "world" || coordinate_type == "raw" || coordinate_type == "coordinate") {
            if (!location->raw_pose) {
                LogWarn << "CheckTeleportRequired raw coordinate unavailable" << VAR(location->mode);
                return 0;
            }
            current = { location->raw_pose->x, location->raw_pose->y };
        }
        else {
            throw std::invalid_argument("coordinate_type must be world or map");
        }

        const double distance = std::hypot(current.first - target.first, current.second - target.second);
        const bool need_teleport = distance >= threshold;
        LogInfo << "CheckTeleportRequired decision" << VAR(current.first) << VAR(current.second) << VAR(target.first) << VAR(target.second)
                << VAR(distance) << VAR(threshold) << VAR(need_teleport) << VAR(location->mode);
        show_teleport_message(context, need_teleport);
        if (!need_teleport) {
            return 1;
        }

        const std::string teleport_point_id = trimmed_string_option(
            params,
            "teleport_point_id",
            trimmed_string_option(params, "teleport_id", ""));
        if (teleport_point_id.empty()) {
            throw std::invalid_argument("teleport_point_id is required");
        }
        const std::string points_file = trimmed_string_option(params, "teleport_points_file", "map_teleport/teleport_points.json");
        return run_map_teleport(context, teleport_point_id, points_file) ? 1 : 0;
    }
    catch (const std::exception& error) {
        LogError << "CheckTeleportRequired failed" << VAR(error.what());
        return 0;
    }
}

MaaBool local_route_navigation(
    MaaContext* context,
    MaaTaskId,
    const char*,
    const char*,
    const char* custom_action_param,
    MaaRecoId,
    const MaaRect*,
    void*)
{
    try {
        return run_local_route(context, parse_object(custom_action_param), {}, "") ? 1 : 0;
    }
    catch (const std::exception& error) {
        LogError << "LocalRouteNavigation failed" << VAR(error.what());
        return 0;
    }
}

MaaBool local_route_navigation_unit_test(
    MaaContext* context,
    MaaTaskId,
    const char*,
    const char*,
    const char* custom_action_param,
    MaaRecoId,
    const MaaRect*,
    void*)
{
    try {
        return run_local_route(context, parse_object(custom_action_param), "penquan", "penquan") ? 1 : 0;
    }
    catch (const std::exception& error) {
        LogError << "LocalRouteNavigationUnitTest failed" << VAR(error.what());
        return 0;
    }
}

MaaBool online_map_navigation(
    MaaContext* context,
    MaaTaskId,
    const char*,
    const char*,
    const char*,
    MaaRecoId,
    const MaaRect*,
    void*)
{
    try {
        json::value params = load_attach(context, "OnlineMapNavigationSettingsConfig");
        const json::value position = load_attach(context, "OnlineMapNavigationPositionBackendConfig");
        const json::value angle = load_attach(context, "OnlineMapNavigationAngleBackendConfig");
        const json::value debug = load_attach(context, "OnlineMapNavigationDebugConfig");
        json::object merged = params.is_object() ? params.as_object() : json::object {};
        if (position.contains("position_backend")) {
            merged["position_backend"] = position.at("position_backend");
        }
        if (angle.contains("angle_backend")) {
            merged["angle_backend"] = angle.at("angle_backend");
        }
        if (debug.contains("debug")) {
            merged["debug"] = debug.at("debug");
        }
        params = merged;
        const int port = integer_option(params, "port", 14514);
        if (port <= 0 || port > 65535) {
            throw std::invalid_argument("port must be between 1 and 65535");
        }

        RouteSession route;
        NavigationWebSocketService* network_ptr = nullptr;
        RouteRunner runner(context, route, navigation_options(params, "auto"), [&network_ptr](const FrameState& frame) {
            if (network_ptr != nullptr) {
                network_ptr->publish_frame(frame);
            }
        });
        NavigationWebSocketService network(
            route,
            static_cast<uint16_t>(port),
            [&runner]() { return runner.source_size(); },
            [&runner]() { return runner.current_point(); });
        network_ptr = &network;
        network.start();
        LogInfo << "OnlineMapNavigation service started" << VAR(port);
        runner.run_until_stopped([&network]() { network.publish_route(); });
        network.stop();
        return 0;
    }
    catch (const std::exception& error) {
        LogError << "OnlineMapNavigation failed" << VAR(error.what());
        return 0;
    }
}

} // namespace navi
