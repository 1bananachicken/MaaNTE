#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <meojson/json.hpp>

#include "types.h"

namespace navi
{

struct RouteSnapshot
{
    std::vector<Point> waypoints;
    bool active = false;
    size_t current_index = 0;
    std::string status = "waiting";
};

class RouteSession
{
public:
    RouteSnapshot snapshot() const;
    void reset(std::vector<Point> waypoints, bool start, std::optional<Point> current_point);
    void start(std::optional<Point> current_point);
    void advance();
    void clear();
    void stop();
    void add(Point point);
    bool target_matches(size_t index, Point point) const;

private:
    static size_t nearest_index(const std::vector<Point>& waypoints, Point current_point);

    mutable std::mutex mutex_;
    std::vector<Point> waypoints_;
    bool active_ = false;
    size_t current_index_ = 0;
    std::string status_ = "waiting";
};

Size parse_source_size(const json::value& value, Size fallback);
Point parse_waypoint(const json::value& value, Size source_size, Size target_size);
std::vector<Point> parse_waypoint_sequence(const json::value& values, Size source_size, Size target_size);
std::vector<Point> parse_route_segment(
    const json::value& data,
    const std::string& route_name,
    int segment_index,
    Size source_size,
    Size target_size);
json::value load_json_file(const std::filesystem::path& path);
std::filesystem::path resolve_route_json_path(const std::filesystem::path& path);
std::string route_payload_json(const RouteSnapshot& route);
std::string handle_route_message(
    const json::value& message,
    RouteSession& route,
    Size source_size,
    std::optional<Point> current_point);

} // namespace navi
