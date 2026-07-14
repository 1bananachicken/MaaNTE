#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <MaaFramework/MaaAPI.h>
#include <opencv2/core.hpp>

namespace navi
{

class ImageBuffer
{
public:
    ImageBuffer();
    ~ImageBuffer();
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;

    MaaImageBuffer* get() const { return handle_; }

private:
    MaaImageBuffer* handle_ = nullptr;
};

class StringBuffer
{
public:
    StringBuffer();
    ~StringBuffer();
    StringBuffer(const StringBuffer&) = delete;
    StringBuffer& operator=(const StringBuffer&) = delete;

    MaaStringBuffer* get() const { return handle_; }
    std::string str() const;

private:
    MaaStringBuffer* handle_ = nullptr;
};

std::filesystem::path executable_dir();
std::filesystem::path project_root();
std::filesystem::path resource_base_path();
cv::Mat image_buffer_to_mat(const MaaImageBuffer* image);
bool capture_frame(MaaController* controller, cv::Mat& frame);
bool wait_controller(MaaController* controller, MaaCtrlId id);
bool task_stopping(MaaContext* context);
void sleep_interruptible(MaaContext* context, std::chrono::milliseconds duration);
void start_parent_process_watcher();
void setup_runtime_dll_search_path();

} // namespace navi
