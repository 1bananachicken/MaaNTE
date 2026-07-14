#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "coordinate.h"
#include "map_locator.h"
#include "types.h"

namespace navi
{

class PositionProvider
{
public:
    PositionProvider(std::string backend, bool debug = false);
    ~PositionProvider();

    PositionProvider(const PositionProvider&) = delete;
    PositionProvider& operator=(const PositionProvider&) = delete;

    LocationResult locate(const cv::Mat& frame = {});
    bool uses_visual_positioning() const { return coordinate_capture_ == nullptr; }
    Size source_size() const;
    void close();

private:
    LocationResult coordinate_location();

    bool debug_ = false;
    std::unique_ptr<CoordinateCapture> coordinate_capture_;
    std::unique_ptr<MapLocator> locator_;
    std::optional<LocationResult> last_coordinate_location_;
    std::chrono::steady_clock::time_point last_coordinate_warning_ {};
    bool coordinate_was_unavailable_ = false;
};

} // namespace navi
