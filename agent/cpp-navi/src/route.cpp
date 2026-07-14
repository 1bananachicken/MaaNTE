#include "route.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "coordinate.h"
#include "util.h"

namespace fs = std::filesystem;

namespace navi
{

namespace
{

constexpr Size kOnlineMapSize { 22528, 22528 };
constexpr double kOnlineOriginX = 11264.0;
constexpr double kOnlineOriginY = 11264.0;
constexpr double kOnlinePixelsPerWorldUnit = 44.0;

double number_at(const json::value& object, const std::string& key)
{
    if (!object.is_object() || !object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument("missing numeric field: " + key);
    }
    return object.at(key).as<double>();
}

std::optional<std::string> string_at(const json::value& object, const std::string& key)
{
    if (!object.is_object() || !object.contains(key) || !object.at(key).is_string()) {
        return std::nullopt;
    }
    return object.at(key).as<std::string>();
}

const json::value& waypoint_holder(const json::value& data)
{
    if (data.is_array()) {
        return data;
    }
    if (!data.is_object()) {
        throw std::invalid_argument("route json must be an object or list");
    }
    for (const char* key : { "waypoints", "points", "path" }) {
        if (data.contains(key) && data.at(key).is_array()) {
            return data.at(key);
        }
    }
    if (data.contains("route") && data.at("route").is_object()) {
        const auto& nested = data.at("route");
        for (const char* key : { "waypoints", "points", "path" }) {
            if (nested.contains(key) && nested.at(key).is_array()) {
                return nested.at(key);
            }
        }
    }
    throw std::invalid_argument("route json needs waypoints, points, or path");
}

const json::value& select_route(const json::value& data, const std::string& route_name)
{
    if (!data.is_object() || !data.contains("routes") || !data.at("routes").is_array()) {
        return data;
    }
    const auto& routes = data.at("routes").as_array();
    if (routes.empty()) {
        throw std::invalid_argument("route json has no routes");
    }
    if (route_name.empty()) {
        if (!routes[0].is_object()) {
            throw std::invalid_argument("route must be an object");
        }
        return routes[0];
    }
    for (const auto& route : routes) {
        if (!route.is_object()) {
            continue;
        }
        if (string_at(route, "name").value_or("") == route_name || string_at(route, "id").value_or("") == route_name) {
            return route;
        }
    }
    throw std::invalid_argument("route not found: " + route_name);
}

std::vector<Point> parse_waypoints_object(const json::value& data, Size source_size, Size target_size)
{
    const Size actual_source = data.is_object() ? parse_source_size(data, source_size) : source_size;
    return parse_waypoint_sequence(waypoint_holder(data), actual_source, target_size);
}

std::string escape_json(const std::string& text)
{
    std::string result;
    result.reserve(text.size() + 8);
    for (const char ch : text) {
        switch (ch) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += ch; break;
        }
    }
    return result;
}

} // namespace

RouteSnapshot RouteSession::snapshot() const
{
    std::scoped_lock lock(mutex_);
    return { waypoints_, active_, current_index_, status_ };
}

void RouteSession::reset(std::vector<Point> waypoints, bool start_now, std::optional<Point> current_point)
{
    std::scoped_lock lock(mutex_);
    waypoints_ = std::move(waypoints);
    active_ = start_now && !waypoints_.empty();
    current_index_ = active_ && current_point ? nearest_index(waypoints_, *current_point) : 0;
    status_ = active_ ? "running" : "ready";
}

void RouteSession::start(std::optional<Point> current_point)
{
    std::scoped_lock lock(mutex_);
    if (waypoints_.empty()) {
        active_ = false;
        status_ = "empty";
        return;
    }
    current_index_ = current_point ? nearest_index(waypoints_, *current_point) : std::min(current_index_, waypoints_.size() - 1);
    active_ = true;
    status_ = "running";
}

void RouteSession::advance()
{
    std::scoped_lock lock(mutex_);
    ++current_index_;
    if (current_index_ >= waypoints_.size()) {
        active_ = false;
        status_ = "arrived";
    }
}

void RouteSession::clear()
{
    std::scoped_lock lock(mutex_);
    waypoints_.clear();
    current_index_ = 0;
    active_ = false;
    status_ = "cleared";
}

void RouteSession::stop()
{
    std::scoped_lock lock(mutex_);
    active_ = false;
    status_ = "stopped";
}

void RouteSession::add(Point point)
{
    std::scoped_lock lock(mutex_);
    waypoints_.push_back(point);
    if (!active_) {
        status_ = "ready";
    }
}

bool RouteSession::target_matches(size_t index, Point point) const
{
    std::scoped_lock lock(mutex_);
    return active_ && current_index_ == index && index < waypoints_.size() && waypoints_[index] == point;
}

size_t RouteSession::nearest_index(const std::vector<Point>& waypoints, Point current_point)
{
    size_t best_index = 0;
    int64_t best_distance = std::numeric_limits<int64_t>::max();
    for (size_t index = 0; index < waypoints.size(); ++index) {
        const int64_t dx = static_cast<int64_t>(waypoints[index].x) - current_point.x;
        const int64_t dy = static_cast<int64_t>(waypoints[index].y) - current_point.y;
        const int64_t distance = dx * dx + dy * dy;
        if (distance < best_distance) {
            best_distance = distance;
            best_index = index;
        }
    }
    return best_index;
}

Size parse_source_size(const json::value& value, Size fallback)
{
    if (!value.is_object()) {
        return fallback;
    }
    if (value.contains("sourceWidth") && value.contains("sourceHeight") && value.at("sourceWidth").is_number() && value.at("sourceHeight").is_number()) {
        return { value.at("sourceWidth").as<int>(), value.at("sourceHeight").as<int>() };
    }
    if (value.contains("sourceSize") && value.at("sourceSize").is_array()) {
        const auto& array = value.at("sourceSize").as_array();
        if (array.size() >= 2 && array[0].is_number() && array[1].is_number()) {
            return { array[0].as<int>(), array[1].as<int>() };
        }
    }
    return fallback;
}

Point parse_waypoint(const json::value& value, Size source_size, Size target_size)
{
    if (!value.is_object()) {
        throw std::invalid_argument("waypoint must be an object");
    }
    if (value.contains("lat") && value.contains("lng")) {
        const double map_x = kOnlineOriginX + number_at(value, "lng") * kOnlinePixelsPerWorldUnit;
        const double map_y = kOnlineOriginY - number_at(value, "lat") * kOnlinePixelsPerWorldUnit;
        return {
            static_cast<int>(std::lround(map_x * target_size.width / kOnlineMapSize.width)),
            static_cast<int>(std::lround(map_y * target_size.height / kOnlineMapSize.height)),
        };
    }
    if (value.contains("x") && value.contains("y")) {
        const auto point = CoordinateTransform::to_map(
            number_at(value, "x"),
            number_at(value, "y"),
            value.contains("z") ? std::optional<double>(number_at(value, "z")) : std::nullopt);
        if (!point) {
            throw std::invalid_argument("raw coordinate waypoint is not finite");
        }
        return {
            static_cast<int>(std::lround(static_cast<double>(point->x) * target_size.width / CoordinateTransform::MapSize.width)),
            static_cast<int>(std::lround(static_cast<double>(point->y) * target_size.height / CoordinateTransform::MapSize.height)),
        };
    }

    double x = 0.0;
    double y = 0.0;
    if (value.contains("pixelX") && value.contains("pixelY")) {
        x = number_at(value, "pixelX");
        y = number_at(value, "pixelY");
    }
    else if (value.contains("target_x") && value.contains("target_y")) {
        x = number_at(value, "target_x");
        y = number_at(value, "target_y");
    }
    else {
        throw std::invalid_argument("waypoint needs pixelX/pixelY, target_x/target_y, x/y, or lat/lng");
    }
    source_size = parse_source_size(value, source_size);
    if (source_size.width <= 0 || source_size.height <= 0) {
        throw std::invalid_argument("waypoint source size must be positive");
    }
    return {
        static_cast<int>(std::lround(x * target_size.width / source_size.width)),
        static_cast<int>(std::lround(y * target_size.height / source_size.height)),
    };
}

std::vector<Point> parse_waypoint_sequence(const json::value& values, Size source_size, Size target_size)
{
    if (!values.is_array()) {
        throw std::invalid_argument("waypoints must be a list");
    }
    std::vector<Point> result;
    result.reserve(values.as_array().size());
    for (const auto& value : values.as_array()) {
        result.push_back(parse_waypoint(value, source_size, target_size));
    }
    return result;
}

std::vector<Point> parse_route_segment(
    const json::value& data,
    const std::string& route_name,
    int segment_index,
    Size source_size,
    Size target_size)
{
    if (!data.is_object() && !data.is_array()) {
        throw std::invalid_argument("route json must be an object or list");
    }
    if (data.is_array()) {
        return parse_waypoint_sequence(data, source_size, target_size);
    }
    const auto& route = select_route(data, route_name);
    if (!route.is_object() || !route.contains("segments") || !route.at("segments").is_array()) {
        return parse_waypoints_object(route, parse_source_size(data, source_size), target_size);
    }
    const auto& segments = route.at("segments").as_array();
    if (segments.empty()) {
        throw std::invalid_argument("route has no segments");
    }
    const size_t index = segment_index <= 1 ? 0 : static_cast<size_t>(segment_index - 1);
    if (index >= segments.size() || !segments[index].is_object()) {
        throw std::invalid_argument("segment index out of range");
    }
    const Size inherited = parse_source_size(route, parse_source_size(data, source_size));
    return parse_waypoints_object(segments[index], inherited, target_size);
}

json::value load_json_file(const fs::path& path)
{
    const auto data = json::open(path);
    if (!data) {
        throw std::runtime_error("failed to parse json: " + path.string());
    }
    return *data;
}

fs::path resolve_route_json_path(const fs::path& path)
{
    if (!path.empty() && fs::exists(path)) {
        return fs::absolute(path);
    }
    fs::path filename = path.filename();
    if (filename.empty()) {
        throw std::invalid_argument("route json path is empty");
    }
    if (filename.extension() != ".json") {
        filename += ".json";
    }
    const fs::path candidate = resource_base_path().parent_path() / "routes" / filename;
    if (!fs::exists(candidate)) {
        throw std::runtime_error("route json not found: " + path.string());
    }
    return candidate;
}

std::string route_payload_json(const RouteSnapshot& route)
{
    std::ostringstream output;
    output << "{\"waypoints\":[";
    for (size_t index = 0; index < route.waypoints.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << "{\"pixelX\":" << route.waypoints[index].x << ",\"pixelY\":" << route.waypoints[index].y << '}';
    }
    output << "],\"active\":" << (route.active ? "true" : "false") << ",\"currentIndex\":" << route.current_index
           << ",\"status\":\"" << escape_json(route.status) << "\"}";
    return output.str();
}

std::string handle_route_message(
    const json::value& message,
    RouteSession& route,
    Size source_size,
    std::optional<Point> current_point)
{
    if (!message.is_object()) {
        return R"({"type":"navi-route-ack","ok":false,"message":"message must be an object"})";
    }
    const std::string type = string_at(message, "type").value_or("");
    if (type == "navi-route-set" || type == "route-set") {
        if (!message.contains("waypoints")) {
            throw std::invalid_argument("waypoints must be a list");
        }
        const Size message_size = parse_source_size(message, source_size);
        const bool start_now = message.contains("start") && message.at("start").is_boolean() && message.at("start").as<bool>();
        route.reset(parse_waypoint_sequence(message.at("waypoints"), message_size, source_size), start_now, current_point);
    }
    else if (type == "navi-route-add" || type == "route-add") {
        route.add(parse_waypoint(message, parse_source_size(message, source_size), source_size));
    }
    else if (type == "navi-route-clear" || type == "route-clear") {
        route.clear();
    }
    else if (type == "navi-route-start" || type == "route-start") {
        route.start(current_point);
    }
    else if (type == "navi-route-stop" || type == "route-stop") {
        route.stop();
    }
    else {
        return R"({"type":"navi-route-ack","ok":false,"message":"unknown type"})";
    }
    return std::string(R"({"type":"navi-route-ack","ok":true,"route":)") + route_payload_json(route.snapshot()) + '}';
}

} // namespace navi
