#pragma once

#include <memory>
#include <string>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "types.h"

namespace navi
{

class AnglePredictor
{
public:
    AnglePredictor(std::string backend, double threshold, bool debug);
    ~AnglePredictor();

    AngleResult predict(const cv::Mat& frame);
    const std::string& provider_name() const { return provider_name_; }
    void close();

private:
    static std::string resolve_backend(std::string backend);
    void show_debug(const cv::Mat& crop, const AngleResult& result, const float* prediction);

    bool debug_ = false;
    double threshold_ = 0.0;
    std::string backend_;
    std::string provider_name_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

} // namespace navi
