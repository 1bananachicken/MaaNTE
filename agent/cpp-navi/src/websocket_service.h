#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <boost/asio.hpp>

#include "route.h"
#include "types.h"

namespace navi
{

class NavigationWebSocketService
{
public:
    using SourceSizeCallback = std::function<Size()>;
    using CurrentPointCallback = std::function<std::optional<Point>()>;

    NavigationWebSocketService(RouteSession& route, uint16_t port, SourceSizeCallback source_size, CurrentPointCallback current_point);
    ~NavigationWebSocketService();

    void start();
    void stop();
    void publish_frame(const FrameState& frame);
    void publish_route();

private:
    class Session;
    void accept_loop();
    void add_session(const std::shared_ptr<Session>& session);
    void remove_session(const Session* session);
    void broadcast_latest();
    std::string handle_message(const std::string& message);
    std::string serialize_state() const;

    RouteSession& route_;
    uint16_t port_ = 14514;
    SourceSizeCallback source_size_;
    CurrentPointCallback current_point_;
    mutable std::mutex state_mutex_;
    FrameState frame_;
    double timestamp_ = 0.0;
    mutable std::mutex sessions_mutex_;
    std::condition_variable sessions_changed_;
    std::vector<std::shared_ptr<Session>> sessions_;
    boost::asio::io_context io_context_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
    std::thread accept_thread_;
    std::atomic_bool stopping_ = false;
};

} // namespace navi
