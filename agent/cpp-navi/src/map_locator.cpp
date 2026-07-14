#include "map_locator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <MaaUtils/Logger.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util.h"

namespace navi
{

namespace
{

const cv::Rect kMiniMapRoi { 28, 15, 150, 150 };
const cv::Rect kButtonRoi { 16, 656, 31, 35 };
constexpr std::array<int, 3> kMapCropSizes { 268, 530, 660 };
constexpr int kSearchRadius = 256;
constexpr double kGlobalMinScore = 0.85;
constexpr double kLocalMinScore = 0.75;
constexpr double kTeleportDistance = 320.0;
constexpr double kSmoothingAlpha = 0.7;
constexpr int kMinFilterPixels = 120;
constexpr int kCirclePadding = 11;
const std::array<cv::Rect, 2> kGlobalSearchRegions {
    cv::Rect { 1506, 8976, 2644 - 1506, 9700 - 8976 },
    cv::Rect { 2312, 2561, 7719 - 2312, 8703 - 2561 },
};

cv::Rect clamp_rect(cv::Rect rect, const cv::Size& bounds)
{
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

} // namespace

MapLocator::MapLocator(bool debug)
    : debug_(debug)
{
    const auto base = resource_base_path();
    big_gray_ = cv::imread((base / "image" / "map" / "bigworldmapSecond.png").string(), cv::IMREAD_GRAYSCALE);
    chat_template_ = cv::imread((base / "image" / "Common" / "Button" / "InWorld" / "Chat.png").string(), cv::IMREAD_COLOR);
    if (big_gray_.empty() || chat_template_.empty()) {
        throw std::runtime_error("Navi map assets could not be loaded");
    }
    origin_width_ = big_gray_.cols;
    origin_height_ = big_gray_.rows;
    if (origin_width_ != MapSize.width || origin_height_ != MapSize.height) {
        LogWarn << "Unexpected Navi big map size" << VAR(origin_width_) << VAR(origin_height_);
    }
    match_maps_.reserve(kMapCropSizes.size());
    for (const int crop_size : kMapCropSizes) {
        const double scale = static_cast<double>(kMiniMapRoi.width) / crop_size;
        cv::Mat resized;
        cv::resize(big_gray_, resized, {}, scale, scale, cv::INTER_AREA);
        match_maps_.emplace_back(scale, std::move(resized));
    }
    if (debug_) {
        const double scale = std::min(1.0, 900.0 / origin_width_);
        cv::Mat resized;
        cv::resize(big_gray_, resized, {}, scale, scale, cv::INTER_AREA);
        cv::cvtColor(resized, debug_map_, cv::COLOR_GRAY2BGR);
    }
    LogInfo << "Navi NCC assets loaded" << VAR(origin_width_) << VAR(origin_height_);
}

MapLocator::~MapLocator()
{
    close();
}

LocationResult MapLocator::locate(const cv::Mat& input)
{
    if (closed_) {
        throw std::runtime_error("map locator is closed");
    }
    if (input.empty() || clamp_rect(kMiniMapRoi, input.size()) != kMiniMapRoi) {
        throw std::invalid_argument("frame does not contain the mini map ROI");
    }
    cv::Mat frame;
    if (input.channels() == 4) {
        cv::cvtColor(input, frame, cv::COLOR_BGRA2BGR);
    }
    else {
        frame = input;
    }

    const cv::Mat minimap = frame(kMiniMapRoi);
    cv::Mat circle_mask = cv::Mat::zeros(kMiniMapRoi.size(), CV_8UC1);
    cv::circle(circle_mask, { kMiniMapRoi.width / 2, kMiniMapRoi.height / 2 }, kMiniMapRoi.width / 2 - kCirclePadding, 255, -1);

    cv::Mat hsv;
    cv::cvtColor(minimap, hsv, cv::COLOR_BGR2HSV);
    cv::Mat color_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, 66, 80), color_mask);
    cv::Mat mask;
    cv::bitwise_and(color_mask, circle_mask, mask);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat templ;
    cv::bitwise_and(channels[2], mask, templ);
    cv::subtract(templ, cv::Scalar(3), templ);

    const auto [in_world, chat_score] = match_chat_button(frame, chat_template_);
    (void)chat_score;
    if (!in_world) {
        auto result = last_result("not_in_world", last_center_ ? 1.0 : 0.0);
        if (debug_) {
            show_debug(templ, result);
        }
        return result;
    }

    auto match = match_all_scales(templ, circle_mask);
    if (match.result.found) {
        activate_scale(match.index);
    }
    LocationResult result = recover_from_teleport(templ, circle_mask, std::move(match.result));
    if (!result.raw_point) {
        last_center_.reset();
        smoothed_center_.reset();
        result.point.reset();
    }
    else if (!smoothed_center_ || result.mode == "global_teleport") {
        smoothed_center_ = cv::Point2d(result.raw_point->x, result.raw_point->y);
        result.point = result.raw_point;
    }
    else {
        smoothed_center_->x = smoothed_center_->x * (1.0 - kSmoothingAlpha) + result.raw_point->x * kSmoothingAlpha;
        smoothed_center_->y = smoothed_center_->y * (1.0 - kSmoothingAlpha) + result.raw_point->y * kSmoothingAlpha;
        result.point = Point { static_cast<int>(std::lround(smoothed_center_->x)), static_cast<int>(std::lround(smoothed_center_->y)) };
    }
    last_center_ = result.raw_point;
    result.found = result.point.has_value();
    if (debug_) {
        show_debug(templ, result);
    }
    return result;
}

cv::Mat MapLocator::sanitize_response(cv::Mat response)
{
    for (int row = 0; row < response.rows; ++row) {
        float* values = response.ptr<float>(row);
        for (int column = 0; column < response.cols; ++column) {
            float& value = values[column];
            if (!std::isfinite(value) || value < -1e-6F || value > 1.0F + 1e-6F) {
                value = -1.0F;
            }
            else {
                value = std::clamp(value, 0.0F, 1.0F);
            }
        }
    }
    return response;
}

std::pair<bool, double> MapLocator::match_chat_button(const cv::Mat& frame, const cv::Mat& chat_template)
{
    const cv::Rect roi_rect = clamp_rect(kButtonRoi, frame.size());
    if (roi_rect.width < chat_template.cols || roi_rect.height < chat_template.rows) {
        return { false, 0.0 };
    }
    cv::Mat green;
    cv::inRange(chat_template, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), green);
    cv::bitwise_not(green, green);
    cv::Mat response;
    cv::matchTemplate(frame(roi_rect), chat_template, response, cv::TM_CCOEFF_NORMED, green);
    response = sanitize_response(std::move(response));
    double score = 0.0;
    cv::minMaxLoc(response, nullptr, &score);
    return { score >= 0.2, score };
}

MapLocator::ScaleMatch MapLocator::match_all_scales(const cv::Mat& templ, const cv::Mat& mask, bool force_global) const
{
    std::vector<ScaleMatch> matches;
    matches.reserve(match_maps_.size());
    for (size_t index = 0; index < match_maps_.size(); ++index) {
        matches.push_back({ index, match_scale(templ, mask, index, force_global) });
    }
    const auto best = std::ranges::max_element(matches, [this](const ScaleMatch& left, const ScaleMatch& right) {
        const auto left_key = std::pair { left.result.found ? left.result.score : -1.0, left.index == active_scale_ };
        const auto right_key = std::pair { right.result.found ? right.result.score : -1.0, right.index == active_scale_ };
        return left_key < right_key;
    });
    return best != matches.end() && best->result.found ? *best : matches[active_scale_];
}

LocationResult MapLocator::match_scale(const cv::Mat& templ, const cv::Mat& mask, size_t index, bool force_global) const
{
    LocationResult result;
    if (cv::countNonZero(templ) < kMinFilterPixels) {
        return result;
    }
    const auto& [scale, big_match] = match_maps_[index];
    double score = -1.0;
    cv::Point best_location;
    cv::Point best_offset;
    double threshold = kGlobalMinScore;
    std::string mode = "global";

    if (last_center_ && !force_global) {
        const double center_x = last_center_->x * scale;
        const double center_y = last_center_->y * scale;
        const double radius = kSearchRadius * scale;
        const int start_x = std::max(0, static_cast<int>(std::floor(center_x - radius - templ.cols * 0.5)));
        const int start_y = std::max(0, static_cast<int>(std::floor(center_y - radius - templ.rows * 0.5)));
        const int end_x = std::min(big_match.cols, static_cast<int>(std::ceil(center_x + radius + templ.cols * 0.5)));
        const int end_y = std::min(big_match.rows, static_cast<int>(std::ceil(center_y + radius + templ.rows * 0.5)));
        const cv::Rect area(start_x, start_y, end_x - start_x, end_y - start_y);
        if (area.width >= templ.cols && area.height >= templ.rows) {
            cv::Mat response;
            cv::matchTemplate(big_match(area), templ, response, cv::TM_CCORR_NORMED, mask);
            response = sanitize_response(std::move(response));
            cv::minMaxLoc(response, nullptr, &score, nullptr, &best_location);
            best_offset = area.tl();
        }
        threshold = kLocalMinScore;
        mode = "local";
    }
    else {
        for (const cv::Rect& original : kGlobalSearchRegions) {
            const cv::Rect scaled(
                static_cast<int>(std::lround(original.x * scale)),
                static_cast<int>(std::lround(original.y * scale)),
                static_cast<int>(std::lround(original.width * scale)),
                static_cast<int>(std::lround(original.height * scale)));
            const cv::Rect area = clamp_rect(scaled, big_match.size());
            if (area.width < templ.cols || area.height < templ.rows) {
                continue;
            }
            cv::Mat response;
            cv::matchTemplate(big_match(area), templ, response, cv::TM_CCORR_NORMED, mask);
            response = sanitize_response(std::move(response));
            double candidate_score = -1.0;
            cv::Point candidate_location;
            cv::minMaxLoc(response, nullptr, &candidate_score, nullptr, &candidate_location);
            if (candidate_score > score) {
                score = candidate_score;
                best_location = candidate_location;
                best_offset = area.tl();
            }
        }
    }

    result.score = score;
    if (score >= threshold) {
        const int match_x = best_offset.x + best_location.x;
        const int match_y = best_offset.y + best_location.y;
        result.raw_point = Point {
            static_cast<int>(std::lround((match_x + templ.cols * 0.5) / scale)),
            static_cast<int>(std::lround((match_y + templ.rows * 0.5) / scale)),
        };
        result.point = result.raw_point;
        result.found = true;
        result.mode = mode;
    }
    return result;
}

LocationResult MapLocator::recover_from_teleport(const cv::Mat& templ, const cv::Mat& mask, LocationResult result)
{
    if (!last_center_) {
        return result;
    }
    bool recover = !result.found || !result.raw_point;
    if (result.mode == "local" && result.raw_point) {
        recover = recover || std::hypot(result.raw_point->x - last_center_->x, result.raw_point->y - last_center_->y) >= kTeleportDistance;
    }
    if (!recover) {
        return result;
    }
    auto global = match_all_scales(templ, mask, true);
    if (!global.result.found || !global.result.raw_point) {
        return result;
    }
    activate_scale(global.index);
    global.result.mode = "global_teleport";
    return global.result;
}

LocationResult MapLocator::last_result(std::string mode, double score) const
{
    LocationResult result;
    result.mode = std::move(mode);
    result.score = score;
    if (smoothed_center_) {
        result.point = Point { static_cast<int>(std::lround(smoothed_center_->x)), static_cast<int>(std::lround(smoothed_center_->y)) };
    }
    else {
        result.point = last_center_;
    }
    result.raw_point = last_center_;
    result.found = result.point.has_value();
    return result;
}

void MapLocator::activate_scale(size_t index)
{
    active_scale_ = index;
}

void MapLocator::show_debug(const cv::Mat& templ, const LocationResult& result)
{
    cv::Mat mini;
    cv::cvtColor(templ, mini, cv::COLOR_GRAY2BGR);
    cv::resize(mini, mini, { 280, 280 }, 0.0, 0.0, cv::INTER_NEAREST);
    cv::putText(
        mini,
        "ncc=" + std::to_string(result.score).substr(0, 5) + " mode=" + result.mode,
        { 5, 22 },
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        { 0, 255, 255 },
        1,
        cv::LINE_AA);
    if (result.point && !debug_map_.empty()) {
        const double scale = static_cast<double>(debug_map_.cols) / origin_width_;
        cv::Mat map = debug_map_.clone();
        cv::circle(map, { static_cast<int>(result.point->x * scale), static_cast<int>(result.point->y * scale) }, 2, { 0, 0, 255 }, -1);
        std::filesystem::create_directories("./debug/cpp-navi");
        cv::imwrite("./debug/cpp-navi/map_locator.png", map);
    }
    std::filesystem::create_directories("./debug/cpp-navi");
    cv::imwrite("./debug/cpp-navi/minimap.png", mini);
}

void MapLocator::close()
{
    if (closed_) {
        return;
    }
    closed_ = true;
    chat_template_.release();
    big_gray_.release();
    match_maps_.clear();
}

} // namespace navi
