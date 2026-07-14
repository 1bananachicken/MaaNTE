#include "position_provider.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

#include <MaaUtils/Logger.h>

namespace navi
{

namespace
{

std::string normalize_backend(std::string value)
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

} // namespace

PositionProvider::PositionProvider(std::string backend, bool debug)
    : debug_(debug)
{
    backend = normalize_backend(std::move(backend));
    if (backend != "map" && backend != "auto" && backend != "coordinate") {
        throw std::invalid_argument("position_backend must be map, auto, or coordinate");
    }

    if (backend != "map") {
        std::string errors;
        for (const char* candidate : { "pcap", "pktmon" }) {
            try {
                auto capture = std::make_unique<CoordinateCapture>(candidate);
                capture->start();
                coordinate_capture_ = std::move(capture);
                LogInfo << "Navi coordinate capture started" << VAR(candidate);
                break;
            }
            catch (const std::exception& error) {
                if (!errors.empty()) {
                    errors += "; ";
                }
                errors += std::string(candidate) + ": " + error.what();
                LogWarn << "Navi coordinate backend unavailable" << VAR(candidate) << VAR(error.what());
            }
        }
        if (!coordinate_capture_ && backend == "coordinate") {
            throw std::runtime_error("Navi coordinate capture unavailable: " + errors);
        }
    }

    if (!coordinate_capture_) {
        locator_ = std::make_unique<MapLocator>(debug_);
    }
}

PositionProvider::~PositionProvider()
{
    close();
}

LocationResult PositionProvider::locate(const cv::Mat& frame)
{
    if (coordinate_capture_) {
        return coordinate_location();
    }
    if (!locator_) {
        throw std::runtime_error("visual map locator is unavailable");
    }
    if (frame.empty()) {
        throw std::invalid_argument("visual positioning requires a frame");
    }
    LocationResult result = locator_->locate(frame);
    if (result.point) {
        const auto raw = CoordinateTransform::to_raw_xy(*result.point);
        if (raw) {
            result.raw_pose = RawPose { raw->first, raw->second, std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };
        }
    }
    return result;
}

LocationResult PositionProvider::coordinate_location()
{
    const auto pose = coordinate_capture_->read(std::chrono::seconds(1));
    if (!pose) {
        LocationResult stale = last_coordinate_location_.value_or(LocationResult {});
        stale.found = false;
        stale.mode = "coordinate_stale";
        if (debug_) {
            LogDebug << "Navi coordinate unavailable" << VAR(coordinate_capture_->stats());
        }
        return stale;
    }
    const auto point = CoordinateTransform::to_map(pose->x, pose->y, pose->z);
    if (!point || !std::isfinite(pose->pitch) || !std::isfinite(pose->heading)) {
        LocationResult invalid = last_coordinate_location_.value_or(LocationResult {});
        invalid.found = false;
        invalid.mode = "coordinate_invalid";
        return invalid;
    }
    LocationResult result;
    result.found = true;
    result.point = point;
    result.raw_point = point;
    result.score = 1.0;
    result.mode = "coordinate";
    result.raw_pose = pose;
    result.camera_pitch = pose->pitch;
    result.camera_heading = std::fmod(pose->heading + 360.0, 360.0);
    last_coordinate_location_ = result;
    return result;
}

Size PositionProvider::source_size() const
{
    return locator_ ? locator_->source_size() : CoordinateTransform::MapSize;
}

void PositionProvider::close()
{
    coordinate_capture_.reset();
    locator_.reset();
}

} // namespace navi
