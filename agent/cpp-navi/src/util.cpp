#include "util.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ranges>
#include <thread>
#include <vector>

#include <MaaFramework/Utility/MaaBuffer.h>
#include <MaaUtils/Logger.h>

#ifdef _WIN32
#include <MaaUtils/SafeWindows.hpp>
#include <TlHelp32.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace navi
{

ImageBuffer::ImageBuffer()
    : handle_(MaaImageBufferCreate())
{
}

ImageBuffer::~ImageBuffer()
{
    MaaImageBufferDestroy(handle_);
}

StringBuffer::StringBuffer()
    : handle_(MaaStringBufferCreate())
{
}

StringBuffer::~StringBuffer()
{
    MaaStringBufferDestroy(handle_);
}

std::string StringBuffer::str() const
{
    const char* value = MaaStringBufferGet(handle_);
    return value == nullptr ? std::string {} : std::string(value);
}

fs::path executable_dir()
{
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    return fs::path(buffer).parent_path();
#else
    std::array<char, 4096> buffer {};
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    return length > 0 ? fs::path(std::string(buffer.data(), static_cast<size_t>(length))).parent_path() : fs::current_path();
#endif
}

fs::path project_root()
{
    for (fs::path current : { executable_dir(), fs::current_path() }) {
        for (; !current.empty(); current = current.parent_path()) {
            if (fs::exists(current / "assets" / "resource" / "base") || fs::exists(current / "resource" / "base")) {
                return current;
            }
            if (current == current.root_path()) {
                break;
            }
        }
    }
    return fs::current_path();
}

fs::path resource_base_path()
{
    const fs::path root = project_root();
    const fs::path development = root / "assets" / "resource" / "base";
    if (fs::exists(development)) {
        return development;
    }
    const fs::path installed = root / "resource" / "base";
    if (fs::exists(installed)) {
        return installed;
    }
    throw std::runtime_error("Unable to locate resource/base directory");
}

cv::Mat image_buffer_to_mat(const MaaImageBuffer* image)
{
    if (image == nullptr || MaaImageBufferIsEmpty(image)) {
        return {};
    }
    return cv::Mat(
               MaaImageBufferHeight(image),
               MaaImageBufferWidth(image),
               MaaImageBufferType(image),
               MaaImageBufferGetRawData(image))
        .clone();
}

bool wait_controller(MaaController* controller, MaaCtrlId id)
{
    return id != MaaInvalidId && MaaControllerWait(controller, id) == MaaStatus_Succeeded;
}

bool capture_frame(MaaController* controller, cv::Mat& frame)
{
    if (controller == nullptr || !wait_controller(controller, MaaControllerPostScreencap(controller))) {
        return false;
    }
    ImageBuffer buffer;
    if (!MaaControllerCachedImage(controller, buffer.get())) {
        return false;
    }
    frame = image_buffer_to_mat(buffer.get());
    return !frame.empty();
}

bool task_stopping(MaaContext* context)
{
    return context == nullptr || MaaTaskerStopping(MaaContextGetTasker(context));
}

void sleep_interruptible(MaaContext* context, std::chrono::milliseconds duration)
{
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (!task_stopping(context) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::min(std::chrono::milliseconds(50), std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())));
    }
}

void setup_runtime_dll_search_path()
{
#ifdef _WIN32
    const fs::path deps_bin = project_root() / "deps" / "bin";
    const fs::path installed_maafw = executable_dir().parent_path() / "maafw";
    const fs::path dll_dir = fs::exists(deps_bin) ? deps_bin : installed_maafw;
    if (fs::exists(dll_dir)) {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        AddDllDirectory(dll_dir.c_str());
    }
#endif
}

void start_parent_process_watcher()
{
#ifdef _WIN32
    const DWORD self = GetCurrentProcessId();
    DWORD parent = 0;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W entry {};
        entry.dwSize = sizeof(entry);
        if (Process32FirstW(snapshot, &entry)) {
            do {
                if (entry.th32ProcessID == self) {
                    parent = entry.th32ParentProcessID;
                    break;
                }
            } while (Process32NextW(snapshot, &entry));
        }
        CloseHandle(snapshot);
    }
    if (parent == 0) {
        return;
    }
    std::thread([parent]() {
        const HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, parent);
        if (process == nullptr) {
            return;
        }
        WaitForSingleObject(process, INFINITE);
        CloseHandle(process);
        std::_Exit(0);
    }).detach();
#else
    const pid_t parent = getppid();
    std::thread([parent]() {
        while (getppid() == parent) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::_Exit(0);
    }).detach();
#endif
}

} // namespace navi
