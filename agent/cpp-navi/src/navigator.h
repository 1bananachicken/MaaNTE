#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <MaaFramework/MaaAPI.h>

#include "angle_predictor.h"
#include "position_provider.h"
#include "route.h"
#include "types.h"

namespace navi
{

struct NavigationOptions
{
    double tolerance = 5.0;
    double frame_interval = 0.1;
    std::string angle_backend = "auto";
    std::string position_backend = "map";
    bool debug = false;
};

class AnglePidController
{
public:
    double update(double error, std::chrono::steady_clock::time_point now);
    void reset();

private:
    double integral_ = 0.0;
    std::optional<double> last_error_;
    std::optional<std::chrono::steady_clock::time_point> last_time_;
};

class WaypointNavigator
{
public:
    using FrameCallback = std::function<void(const FrameState&)>;
    using CancelCallback = std::function<bool()>;

    WaypointNavigator(MaaContext* context, NavigationOptions options, FrameCallback on_frame, CancelCallback should_cancel);
    ~WaypointNavigator();

    std::optional<FrameState> update();
    bool move_to(Point target);
    void sleep_remaining(std::chrono::steady_clock::time_point started) const;
    void close();
    Size source_size() const;
    std::optional<Point> current_point() const { return current_point_; }

private:
    void press_forward();
    void release();

    MaaContext* context_ = nullptr;
    MaaController* controller_ = nullptr;
    NavigationOptions options_;
    FrameCallback on_frame_;
    CancelCallback should_cancel_;
    std::unique_ptr<PositionProvider> position_provider_;
    std::unique_ptr<AnglePredictor> predictor_;
    std::optional<Point> current_point_;
    AnglePidController turn_pid_;
    bool w_down_ = false;
    std::chrono::steady_clock::time_point last_w_down_at_ {};
    bool closed_ = false;
};

class RouteRunner
{
public:
    using FrameCallback = WaypointNavigator::FrameCallback;

    RouteRunner(MaaContext* context, RouteSession& route, NavigationOptions options, FrameCallback on_frame = {});
    ~RouteRunner();

    std::string run_until_stopped(const std::function<void()>& on_tick = {}, bool stop_when_route_done = false);
    std::optional<Point> update_current_frame();
    Size source_size() const;
    std::optional<Point> current_point() const;
    void close();

private:
    bool should_cancel() const;
    void frame_received(const FrameState& frame);

    MaaContext* context_ = nullptr;
    RouteSession& route_;
    FrameCallback on_frame_;
    std::unique_ptr<WaypointNavigator> navigator_;
    mutable std::mutex mutex_;
    std::optional<Point> current_point_;
    std::optional<std::pair<size_t, Point>> moving_target_;
};

} // namespace navi
