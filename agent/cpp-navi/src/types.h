#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace navi
{

struct Point
{
    int x = 0;
    int y = 0;

    friend bool operator==(const Point&, const Point&) = default;
};

struct Size
{
    int width = 11264;
    int height = 11264;
};

struct RawPose
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double pitch = 0.0;
    double heading = 0.0;
};

struct LocationResult
{
    bool found = false;
    std::optional<Point> point;
    std::optional<Point> raw_point;
    double score = 0.0;
    std::string mode = "rejected";
    std::optional<RawPose> raw_pose;
    std::optional<double> camera_pitch;
    std::optional<double> camera_heading;
};

struct AngleResult
{
    bool found = false;
    std::optional<double> angle;
    double confidence = 0.0;
};

struct FrameState
{
    LocationResult location;
    AngleResult angle;
};

} // namespace navi
