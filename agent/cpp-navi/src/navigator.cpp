#include "navigator.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include <MaaUtils/Logger.h>

#include "util.h"

namespace navi
{

namespace
{

constexpr int kKeyW = 87;
constexpr double kNetworkFrameInterval = 1.0 / 60.0;
constexpr double kPi = 3.14159265358979323846;

} // namespace

double AnglePidController::update(double error, std::chrono::steady_clock::time_point now)
{
    constexpr double kp = 0.85;
    constexpr double ki = 0.04;
    constexpr double kd = 0.10;
    constexpr double output_limit = 35.0;
    constexpr double integral_limit = 120.0;
    constexpr double deadband = 4.0;
    constexpr double max_dt = 0.25;
    if (std::abs(error) <= deadband) {
        reset();
        return 0.0;
    }
    double dt = max_dt;
    double derivative = 0.0;
    if (last_time_) {
        dt = std::clamp(std::chrono::duration<double>(now - *last_time_).count(), 1e-3, max_dt);
        derivative = last_error_ ? (error - *last_error_) / dt : 0.0;
    }
    if (last_error_ && error * *last_error_ < 0.0) {
        integral_ = 0.0;
    }
    integral_ = std::clamp(integral_ + error * dt, -integral_limit, integral_limit);
    const double output = std::clamp(kp * error + ki * integral_ + kd * derivative, -output_limit, output_limit);
    last_error_ = error;
    last_time_ = now;
    return output;
}

void AnglePidController::reset()
{
    integral_ = 0.0;
    last_error_.reset();
    last_time_.reset();
}

WaypointNavigator::WaypointNavigator(
    MaaContext* context,
    NavigationOptions options,
    FrameCallback on_frame,
    CancelCallback should_cancel)
    : context_(context)
    , controller_(MaaTaskerGetController(MaaContextGetTasker(context)))
    , options_(std::move(options))
    , on_frame_(std::move(on_frame))
    , should_cancel_(std::move(should_cancel))
{
    options_.frame_interval = std::max(0.05, options_.frame_interval);
    position_provider_ = std::make_unique<PositionProvider>(options_.position_backend, options_.debug);
    if (position_provider_->uses_visual_positioning()) {
        predictor_ = std::make_unique<AnglePredictor>(options_.angle_backend, 0.0, options_.debug);
    }
    else {
        options_.frame_interval = kNetworkFrameInterval;
        LogInfo << "Navi network pose active; visual inference disabled";
    }
}

WaypointNavigator::~WaypointNavigator()
{
    close();
}

std::optional<FrameState> WaypointNavigator::update()
{
    FrameState state;
    if (!position_provider_->uses_visual_positioning()) {
        state.location = position_provider_->locate();
        state.angle.found = state.location.found && state.location.camera_heading.has_value();
        state.angle.angle = state.location.camera_heading;
        state.angle.confidence = state.angle.angle ? 1.0 : 0.0;
    }
    else {
        cv::Mat frame;
        if (!capture_frame(controller_, frame)) {
            return std::nullopt;
        }
        state.location = position_provider_->locate(frame);
        state.angle = predictor_->predict(frame);
    }
    if (state.location.found && state.location.point) {
        current_point_ = state.location.point;
    }
    if (on_frame_) {
        on_frame_(state);
    }
    return state;
}

bool WaypointNavigator::move_to(Point target)
{
    turn_pid_.reset();
    auto last_log = std::chrono::steady_clock::time_point {};
    while (!task_stopping(context_)) {
        if (should_cancel_ && should_cancel_()) {
            release();
            return false;
        }
        const auto started = std::chrono::steady_clock::now();
        const auto frame = update();
        if (!frame) {
            sleep_remaining(started);
            continue;
        }
        const auto& location = frame->location;
        const auto& angle = frame->angle;
        if (!location.found || !location.point || !angle.found || !angle.angle) {
            turn_pid_.reset();
            release();
            sleep_remaining(started);
            continue;
        }
        const double dx = target.x - location.point->x;
        const double dy = target.y - location.point->y;
        const double distance = std::hypot(dx, dy);
        if (distance <= options_.tolerance) {
            LogInfo << "WaypointNavigator arrived" << VAR(target.x) << VAR(target.y) << VAR(location.point->x) << VAR(location.point->y)
                    << VAR(distance);
            release();
            return true;
        }
        double desired = std::atan2(dx, -dy) * 180.0 / kPi;
        desired = std::fmod(desired + 360.0, 360.0);
        const double delta = std::fmod(desired - *angle.angle + 540.0, 360.0) - 180.0;
        const double turn_degrees = turn_pid_.update(delta, std::chrono::steady_clock::now());
        const int turn_dx = static_cast<int>(std::lround(turn_degrees * 10.0));
        press_forward();
        if (turn_dx != 0) {
            wait_controller(controller_, MaaControllerPostRelativeMove(controller_, turn_dx, 0));
        }
        sleep_interruptible(context_, std::chrono::milliseconds(120));
        if (task_stopping(context_)) {
            release();
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(2)) {
            LogInfo << "WaypointNavigator moving" << VAR(target.x) << VAR(target.y) << VAR(location.point->x) << VAR(location.point->y)
                    << VAR(distance) << VAR(delta) << VAR(turn_degrees);
            last_log = now;
        }
        sleep_remaining(started);
    }
    release();
    return false;
}

void WaypointNavigator::sleep_remaining(std::chrono::steady_clock::time_point started) const
{
    const auto target = std::chrono::duration<double>(options_.frame_interval);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    if (elapsed < target) {
        sleep_interruptible(context_, std::chrono::duration_cast<std::chrono::milliseconds>(target - elapsed));
    }
}

void WaypointNavigator::press_forward()
{
    const auto now = std::chrono::steady_clock::now();
    if (w_down_ && now - last_w_down_at_ < std::chrono::milliseconds(500)) {
        return;
    }
    wait_controller(controller_, MaaControllerPostKeyDown(controller_, kKeyW));
    w_down_ = true;
    last_w_down_at_ = now;
}

void WaypointNavigator::release()
{
    turn_pid_.reset();
    if (w_down_) {
        wait_controller(controller_, MaaControllerPostKeyUp(controller_, kKeyW));
        w_down_ = false;
    }
    last_w_down_at_ = {};
}

void WaypointNavigator::close()
{
    if (closed_) {
        return;
    }
    closed_ = true;
    release();
    position_provider_.reset();
    predictor_.reset();
}

Size WaypointNavigator::source_size() const
{
    return position_provider_ ? position_provider_->source_size() : CoordinateTransform::MapSize;
}

RouteRunner::RouteRunner(MaaContext* context, RouteSession& route, NavigationOptions options, FrameCallback on_frame)
    : context_(context)
    , route_(route)
    , on_frame_(std::move(on_frame))
{
    navigator_ = std::make_unique<WaypointNavigator>(
        context,
        std::move(options),
        [this](const FrameState& frame) { frame_received(frame); },
        [this]() { return should_cancel(); });
}

RouteRunner::~RouteRunner()
{
    close();
}

std::string RouteRunner::run_until_stopped(const std::function<void()>& on_tick, bool stop_when_route_done)
{
    while (!task_stopping(context_)) {
        if (on_tick) {
            on_tick();
        }
        const RouteSnapshot route = route_.snapshot();
        if (!route.active) {
            if (stop_when_route_done && (route.status == "arrived" || route.status == "cleared" || route.status == "empty" || route.status == "stopped")) {
                return route.status;
            }
            const auto started = std::chrono::steady_clock::now();
            navigator_->update();
            navigator_->sleep_remaining(started);
            continue;
        }
        if (route.current_index >= route.waypoints.size()) {
            route_.advance();
            continue;
        }
        const Point target = route.waypoints[route.current_index];
        {
            std::scoped_lock lock(mutex_);
            moving_target_ = std::pair { route.current_index, target };
        }
        const bool arrived = navigator_->move_to(target);
        {
            std::scoped_lock lock(mutex_);
            moving_target_.reset();
        }
        if (arrived) {
            route_.advance();
        }
    }
    return "stopped";
}

std::optional<Point> RouteRunner::update_current_frame()
{
    navigator_->update();
    return current_point();
}

Size RouteRunner::source_size() const
{
    return navigator_->source_size();
}

std::optional<Point> RouteRunner::current_point() const
{
    std::scoped_lock lock(mutex_);
    return current_point_;
}

void RouteRunner::close()
{
    navigator_.reset();
}

bool RouteRunner::should_cancel() const
{
    std::scoped_lock lock(mutex_);
    return moving_target_ && !route_.target_matches(moving_target_->first, moving_target_->second);
}

void RouteRunner::frame_received(const FrameState& frame)
{
    if (frame.location.found && frame.location.point) {
        std::scoped_lock lock(mutex_);
        current_point_ = frame.location.point;
    }
    if (on_frame_) {
        on_frame_(frame);
    }
}

} // namespace navi
