#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "types.h"

namespace navi
{

class CoordinateTransform
{
public:
    static constexpr Size MapSize { 11264, 11264 };

    static std::optional<Point> to_map(double x, double y, std::optional<double> z = std::nullopt);
    static std::optional<std::pair<double, double>> to_raw_xy(Point point);
};

// Network capture and decoding remain inside the protected Python extension.
// This class only talks to its localhost bridge.
class CoordinateCapture
{
public:
    explicit CoordinateCapture(std::string backend);
    ~CoordinateCapture();
    CoordinateCapture(const CoordinateCapture&) = delete;
    CoordinateCapture& operator=(const CoordinateCapture&) = delete;

    void start();
    std::optional<RawPose> read(std::chrono::duration<double> max_age);
    std::string stats() const;
    void close();

private:
    class Bridge;

    std::string backend_name_;
    std::unique_ptr<Bridge> bridge_;
};

} // namespace navi
