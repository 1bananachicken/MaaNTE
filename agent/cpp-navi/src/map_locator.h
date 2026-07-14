#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "types.h"

namespace navi
{

class MapLocator
{
public:
    static constexpr Size MapSize { 11264, 11264 };

    explicit MapLocator(bool debug = false);
    ~MapLocator();

    LocationResult locate(const cv::Mat& frame);
    Size source_size() const { return { origin_width_, origin_height_ }; }
    void close();

private:
    struct ScaleMatch
    {
        size_t index = 0;
        LocationResult result;
    };

    static cv::Mat sanitize_response(cv::Mat response);
    static std::pair<bool, double> match_chat_button(const cv::Mat& frame, const cv::Mat& chat_template);
    ScaleMatch match_all_scales(const cv::Mat& templ, const cv::Mat& mask, bool force_global = false) const;
    LocationResult match_scale(const cv::Mat& templ, const cv::Mat& mask, size_t index, bool force_global) const;
    LocationResult recover_from_teleport(const cv::Mat& templ, const cv::Mat& mask, LocationResult result);
    LocationResult last_result(std::string mode, double score) const;
    void activate_scale(size_t index);
    void show_debug(const cv::Mat& templ, const LocationResult& result);

    bool debug_ = false;
    bool closed_ = false;
    cv::Mat chat_template_;
    cv::Mat big_gray_;
    std::vector<std::pair<double, cv::Mat>> match_maps_;
    size_t active_scale_ = 0;
    int origin_width_ = 0;
    int origin_height_ = 0;
    std::optional<Point> last_center_;
    std::optional<cv::Point2d> smoothed_center_;
    cv::Mat debug_map_;
};

} // namespace navi
