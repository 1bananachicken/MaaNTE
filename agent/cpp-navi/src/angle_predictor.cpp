#include "angle_predictor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <MaaUtils/Logger.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "util.h"

#if defined(_WIN32) && __has_include(<onnxruntime/core/providers/dml/dml_provider_factory.h>)
#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#define NAVI_HAS_DIRECTML 1
#endif

namespace navi
{

namespace
{

Ort::Env& ort_environment()
{
    static Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "MaaNTE-Navi");
    return environment;
}

const cv::Rect kPointerRoi { 73, 60, 64, 64 };
constexpr double kPi = 3.14159265358979323846;

} // namespace

AnglePredictor::AnglePredictor(std::string backend, double threshold, bool debug)
    : debug_(debug)
    , threshold_(threshold)
    , backend_(resolve_backend(std::move(backend)))
{
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
#ifdef NAVI_HAS_DIRECTML
    if (backend_ == "directml") {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, 0));
        provider_name_ = "DmlExecutionProvider";
    }
    else
#endif
    {
        if (backend_ == "directml") {
            LogWarn << "DirectML headers are unavailable; falling back to CPU";
            backend_ = "cpu";
        }
        provider_name_ = "CPUExecutionProvider";
    }

    const auto model_path = resource_base_path() / "model" / "navi" / "pointer_model.onnx";
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Angle model not found: " + model_path.string());
    }
#ifdef _WIN32
    session_ = std::make_unique<Ort::Session>(ort_environment(), model_path.c_str(), options);
#else
    session_ = std::make_unique<Ort::Session>(ort_environment(), model_path.string().c_str(), options);
#endif
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
}

AnglePredictor::~AnglePredictor()
{
    close();
}

AngleResult AnglePredictor::predict(const cv::Mat& input)
{
    if (!session_) {
        throw std::runtime_error("angle predictor is closed");
    }
    if (input.empty() || (kPointerRoi & cv::Rect(0, 0, input.cols, input.rows)) != kPointerRoi) {
        throw std::invalid_argument("frame does not contain pointer ROI");
    }
    cv::Mat frame;
    if (input.channels() == 4) {
        cv::cvtColor(input, frame, cv::COLOR_BGRA2BGR);
    }
    else {
        frame = input;
    }
    const cv::Mat crop = frame(kPointerRoi).clone();
    cv::Mat rgb;
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    cv::Mat floating;
    rgb.convertTo(floating, CV_32F, 1.0 / 255.0);

    std::vector<float> input_values(3 * 64 * 64);
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < 64; ++row) {
            for (int column = 0; column < 64; ++column) {
                input_values[channel * 64 * 64 + row * 64 + column] = floating.at<cv::Vec3f>(row, column)[channel];
            }
        }
    }
    const std::array<int64_t, 4> shape { 1, 3, 64, 64 };
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input_values.data(), input_values.size(), shape.data(), shape.size());
    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };
    auto outputs = session_->Run(Ort::RunOptions { nullptr }, input_names, &tensor, 1, output_names, 1);
    if (outputs.empty() || !outputs[0].IsTensor()) {
        return {};
    }
    const auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto dimensions = tensor_info.GetShape();
    if (dimensions.size() != 3 || dimensions[2] < 15) {
        throw std::runtime_error("unexpected angle model output shape");
    }
    const size_t rows = static_cast<size_t>(dimensions[1]);
    const size_t stride = static_cast<size_t>(dimensions[2]);
    const float* data = outputs[0].GetTensorData<float>();
    size_t best_index = 0;
    float confidence = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < rows; ++index) {
        if (data[index * stride + 4] > confidence) {
            confidence = data[index * stride + 4];
            best_index = index;
        }
    }
    const float* prediction = data + best_index * stride;
    AngleResult result;
    result.confidence = confidence;
    if (confidence > threshold_) {
        const double tip_x = prediction[6];
        const double tip_y = prediction[7];
        const double left_x = prediction[9];
        const double left_y = prediction[10];
        const double right_x = prediction[12];
        const double right_y = prediction[13];
        const double tail_x = (left_x + right_x) * 0.5;
        const double tail_y = (left_y + right_y) * 0.5;
        double angle = std::atan2(tip_x - tail_x, -(tip_y - tail_y)) * 180.0 / kPi;
        angle = std::fmod(angle + 360.0, 360.0);
        result.found = true;
        result.angle = angle;
    }
    if (debug_) {
        show_debug(crop, result, prediction);
    }
    return result;
}

std::string AnglePredictor::resolve_backend(std::string backend)
{
    if (backend.empty()) {
        if (const char* configured = std::getenv("MAA_ONNX_BACKEND")) {
            backend = configured;
        }
    }
    std::ranges::transform(backend, backend.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (backend.empty() || backend == "auto") {
#ifdef NAVI_HAS_DIRECTML
        return "directml";
#else
        return "cpu";
#endif
    }
    if (backend == "dml") {
        return "directml";
    }
    if (backend != "cpu" && backend != "directml") {
        LogWarn << "Unknown Navi inference backend; falling back to CPU" << VAR(backend);
        return "cpu";
    }
    return backend;
}

void AnglePredictor::show_debug(const cv::Mat& crop, const AngleResult& result, const float* prediction)
{
    cv::Mat display = crop.clone();
    if (result.found) {
        cv::rectangle(display, { static_cast<int>(prediction[0]), static_cast<int>(prediction[1]) }, { static_cast<int>(prediction[2]), static_cast<int>(prediction[3]) }, { 0, 255, 0 });
        const cv::Point tip(static_cast<int>(prediction[6]), static_cast<int>(prediction[7]));
        const cv::Point tail(
            static_cast<int>((prediction[9] + prediction[12]) * 0.5F),
            static_cast<int>((prediction[10] + prediction[13]) * 0.5F));
        cv::line(display, tail, tip, { 255, 0, 255 }, 2);
    }
    cv::resize(display, display, { 400, 400 }, 0.0, 0.0, cv::INTER_CUBIC);
    const std::string text = result.angle ? "Angle: " + std::to_string(*result.angle) : "NO TARGET";
    cv::putText(display, text, { 10, 28 }, cv::FONT_HERSHEY_SIMPLEX, 0.65, result.found ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    std::filesystem::create_directories("./debug/cpp-navi");
    cv::imwrite("./debug/cpp-navi/angle_predictor.png", display);
}

void AnglePredictor::close()
{
    session_.reset();
}

} // namespace navi
