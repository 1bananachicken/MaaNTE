#include "websocket_service.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <span>

#include <MaaUtils/Logger.h>

namespace navi
{

using tcp = boost::asio::ip::tcp;

namespace
{

double unix_now()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string optional_number(std::optional<double> value)
{
    if (!value || !std::isfinite(*value)) {
        return "null";
    }
    std::ostringstream output;
    output.precision(12);
    output << *value;
    return output.str();
}

std::string escape_json(const std::string& text)
{
    std::string result;
    for (const char ch : text) {
        switch (ch) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += ch; break;
        }
    }
    return result;
}

std::array<uint8_t, 20> sha1(std::string_view input)
{
    std::vector<uint8_t> message(input.begin(), input.end());
    const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) {
        message.push_back(0);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<uint8_t>(bit_length >> shift));
    }
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    for (size_t offset = 0; offset < message.size(); offset += 64) {
        std::array<uint32_t, 80> words {};
        for (size_t index = 0; index < 16; ++index) {
            const size_t pos = offset + index * 4;
            words[index] = (static_cast<uint32_t>(message[pos]) << 24) | (static_cast<uint32_t>(message[pos + 1]) << 16)
                           | (static_cast<uint32_t>(message[pos + 2]) << 8) | message[pos + 3];
        }
        for (size_t index = 16; index < words.size(); ++index) {
            words[index] = std::rotl(words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16], 1);
        }
        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        for (size_t index = 0; index < words.size(); ++index) {
            uint32_t f = 0;
            uint32_t k = 0;
            if (index < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            }
            else if (index < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            }
            else if (index < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            }
            else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            const uint32_t temp = std::rotl(a, 5) + f + e + k + words[index];
            e = d;
            d = c;
            c = std::rotl(b, 30);
            b = a;
            a = temp;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }
    std::array<uint8_t, 20> output {};
    const std::array<uint32_t, 5> values { h0, h1, h2, h3, h4 };
    for (size_t index = 0; index < values.size(); ++index) {
        for (int byte = 0; byte < 4; ++byte) {
            output[index * 4 + byte] = static_cast<uint8_t>(values[index] >> (24 - byte * 8));
        }
    }
    return output;
}

std::string base64(std::span<const uint8_t> input)
{
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    for (size_t index = 0; index < input.size(); index += 3) {
        const uint32_t value = (static_cast<uint32_t>(input[index]) << 16)
                               | (index + 1 < input.size() ? static_cast<uint32_t>(input[index + 1]) << 8 : 0)
                               | (index + 2 < input.size() ? input[index + 2] : 0);
        output += alphabet[(value >> 18) & 63];
        output += alphabet[(value >> 12) & 63];
        output += index + 1 < input.size() ? alphabet[(value >> 6) & 63] : '=';
        output += index + 2 < input.size() ? alphabet[value & 63] : '=';
    }
    return output;
}

std::string trim(std::string value)
{
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

} // namespace

class NavigationWebSocketService::Session : public std::enable_shared_from_this<Session>
{
public:
    Session(tcp::socket socket, NavigationWebSocketService& owner)
        : socket_(std::move(socket))
        , owner_(owner)
    {
    }

    void start()
    {
        owner_.add_session(shared_from_this());
        std::thread([self = shared_from_this()]() { self->run(); }).detach();
    }

    bool send(const std::string& payload)
    {
        std::scoped_lock lock(write_mutex_);
        if (!open_) {
            return false;
        }
        boost::system::error_code error;
        write_frame(0x1, payload, error);
        if (error) {
            open_ = false;
        }
        return !error;
    }

    void close()
    {
        std::scoped_lock lock(write_mutex_);
        boost::system::error_code error;
        if (open_) {
            write_frame(0x8, {}, error);
        }
        open_ = false;
        if (socket_.is_open()) {
            socket_.shutdown(tcp::socket::shutdown_both, error);
            socket_.close(error);
        }
    }

private:
    void run()
    {
        boost::system::error_code error;
        if (!handshake(error)) {
            owner_.remove_session(this);
            return;
        }
        open_ = true;
        send(owner_.serialize_state());
        while (open_ && !owner_.stopping_) {
            uint8_t opcode = 0;
            std::string message;
            if (!read_frame(opcode, message, error) || opcode == 0x8) {
                break;
            }
            if (opcode == 0x9) {
                std::scoped_lock lock(write_mutex_);
                write_frame(0xA, message, error);
            }
            else if (opcode == 0x1) {
                send(owner_.handle_message(message));
            }
        }
        open_ = false;
        owner_.remove_session(this);
    }

    bool handshake(boost::system::error_code& error)
    {
        boost::asio::streambuf request;
        boost::asio::read_until(socket_, request, "\r\n\r\n", error);
        if (error) {
            return false;
        }
        std::istream stream(&request);
        std::string line;
        std::string key;
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const auto separator = line.find(':');
            if (separator == std::string::npos) {
                continue;
            }
            std::string name = line.substr(0, separator);
            std::ranges::transform(name, name.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (name == "sec-websocket-key") {
                key = trim(line.substr(separator + 1));
            }
        }
        if (key.empty()) {
            return false;
        }
        const auto digest = sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        const std::string response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "
                                     + base64(digest) + "\r\n\r\n";
        boost::asio::write(socket_, boost::asio::buffer(response), error);
        return !error;
    }

    bool read_frame(uint8_t& opcode, std::string& payload, boost::system::error_code& error)
    {
        std::array<uint8_t, 2> header {};
        boost::asio::read(socket_, boost::asio::buffer(header), error);
        if (error) {
            return false;
        }
        opcode = header[0] & 0x0F;
        const bool masked = (header[1] & 0x80U) != 0;
        uint64_t length = header[1] & 0x7FU;
        if (length == 126) {
            std::array<uint8_t, 2> extended {};
            boost::asio::read(socket_, boost::asio::buffer(extended), error);
            length = (static_cast<uint64_t>(extended[0]) << 8) | extended[1];
        }
        else if (length == 127) {
            std::array<uint8_t, 8> extended {};
            boost::asio::read(socket_, boost::asio::buffer(extended), error);
            length = 0;
            for (const uint8_t byte : extended) {
                length = (length << 8) | byte;
            }
        }
        if (error || length > 16 * 1024 * 1024) {
            return false;
        }
        std::array<uint8_t, 4> mask {};
        if (masked) {
            boost::asio::read(socket_, boost::asio::buffer(mask), error);
        }
        payload.resize(static_cast<size_t>(length));
        if (length > 0) {
            boost::asio::read(socket_, boost::asio::buffer(payload), error);
        }
        if (error) {
            return false;
        }
        if (masked) {
            for (size_t index = 0; index < payload.size(); ++index) {
                payload[index] = static_cast<char>(static_cast<uint8_t>(payload[index]) ^ mask[index % 4]);
            }
        }
        return true;
    }

    void write_frame(uint8_t opcode, std::string_view payload, boost::system::error_code& error)
    {
        std::vector<uint8_t> frame { static_cast<uint8_t>(0x80U | opcode) };
        if (payload.size() < 126) {
            frame.push_back(static_cast<uint8_t>(payload.size()));
        }
        else if (payload.size() <= 0xFFFF) {
            frame.push_back(126);
            frame.push_back(static_cast<uint8_t>(payload.size() >> 8));
            frame.push_back(static_cast<uint8_t>(payload.size()));
        }
        else {
            frame.push_back(127);
            for (int shift = 56; shift >= 0; shift -= 8) {
                frame.push_back(static_cast<uint8_t>(static_cast<uint64_t>(payload.size()) >> shift));
            }
        }
        frame.insert(frame.end(), payload.begin(), payload.end());
        boost::asio::write(socket_, boost::asio::buffer(frame), error);
    }

    tcp::socket socket_;
    NavigationWebSocketService& owner_;
    std::mutex write_mutex_;
    std::atomic_bool open_ = false;
};

NavigationWebSocketService::NavigationWebSocketService(
    RouteSession& route,
    uint16_t port,
    SourceSizeCallback source_size,
    CurrentPointCallback current_point)
    : route_(route)
    , port_(port)
    , source_size_(std::move(source_size))
    , current_point_(std::move(current_point))
{
}

NavigationWebSocketService::~NavigationWebSocketService()
{
    stop();
}

void NavigationWebSocketService::start()
{
    if (accept_thread_.joinable()) {
        return;
    }
    stopping_ = false;
    const tcp::endpoint endpoint(tcp::v4(), port_);
    acceptor_ = std::make_unique<tcp::acceptor>(io_context_);
    acceptor_->open(endpoint.protocol());
    acceptor_->set_option(boost::asio::socket_base::reuse_address(true));
    acceptor_->bind(endpoint);
    acceptor_->listen();
    accept_thread_ = std::thread([this]() { accept_loop(); });
}

void NavigationWebSocketService::stop()
{
    stopping_ = true;
    if (acceptor_) {
        boost::system::error_code error;
        acceptor_->close(error);
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    std::vector<std::shared_ptr<Session>> sessions;
    {
        std::scoped_lock lock(sessions_mutex_);
        sessions = sessions_;
    }
    for (const auto& session : sessions) {
        session->close();
    }
    {
        std::unique_lock lock(sessions_mutex_);
        sessions_changed_.wait(lock, [this]() { return sessions_.empty(); });
    }
    acceptor_.reset();
}

void NavigationWebSocketService::accept_loop()
{
    while (!stopping_) {
        boost::system::error_code error;
        tcp::socket socket(io_context_);
        acceptor_->accept(socket, error);
        if (error) {
            if (!stopping_) {
                LogWarn << "Navi WebSocket accept failed" << VAR(error.message());
            }
            continue;
        }
        if (stopping_) {
            socket.close(error);
            break;
        }
        std::make_shared<Session>(std::move(socket), *this)->start();
    }
}

void NavigationWebSocketService::add_session(const std::shared_ptr<Session>& session)
{
    std::scoped_lock lock(sessions_mutex_);
    sessions_.push_back(session);
}

void NavigationWebSocketService::remove_session(const Session* session)
{
    {
        std::scoped_lock lock(sessions_mutex_);
        std::erase_if(sessions_, [session](const auto& item) { return item.get() == session; });
    }
    sessions_changed_.notify_all();
}

void NavigationWebSocketService::publish_frame(const FrameState& frame)
{
    {
        std::scoped_lock lock(state_mutex_);
        frame_ = frame;
        timestamp_ = unix_now();
    }
    broadcast_latest();
}

void NavigationWebSocketService::publish_route()
{
    {
        std::scoped_lock lock(state_mutex_);
        timestamp_ = unix_now();
    }
    broadcast_latest();
}

void NavigationWebSocketService::broadcast_latest()
{
    const std::string payload = serialize_state();
    std::vector<std::shared_ptr<Session>> sessions;
    {
        std::scoped_lock lock(sessions_mutex_);
        sessions = sessions_;
    }
    for (const auto& session : sessions) {
        session->send(payload);
    }
}

std::string NavigationWebSocketService::handle_message(const std::string& message)
{
    try {
        const auto parsed = json::parse(message);
        if (!parsed) {
            throw std::invalid_argument("invalid json");
        }
        return handle_route_message(*parsed, route_, source_size_(), current_point_());
    }
    catch (const std::exception& error) {
        return std::string("{\"type\":\"navi-error\",\"message\":\"") + escape_json(error.what()) + "\"}";
    }
}

std::string NavigationWebSocketService::serialize_state() const
{
    FrameState frame;
    double timestamp = 0.0;
    {
        std::scoped_lock lock(state_mutex_);
        frame = frame_;
        timestamp = timestamp_;
    }
    const Size source = source_size_();
    std::ostringstream output;
    output.precision(12);
    output << "{\"type\":\"navi-state\",\"version\":1,\"position\":";
    if (frame.location.point || frame.location.raw_pose) {
        output << '{';
        bool comma = false;
        if (frame.location.raw_pose) {
            output << "\"x\":" << frame.location.raw_pose->x << ",\"y\":" << frame.location.raw_pose->y;
            comma = true;
            if (std::isfinite(frame.location.raw_pose->z)) {
                output << ",\"z\":" << frame.location.raw_pose->z;
            }
        }
        if (frame.location.point) {
            if (comma) {
                output << ',';
            }
            output << "\"pixelX\":" << frame.location.point->x << ",\"pixelY\":" << frame.location.point->y
                   << ",\"sourceWidth\":" << source.width << ",\"sourceHeight\":" << source.height;
            comma = true;
        }
        if (comma) {
            output << ',';
        }
        output << "\"score\":" << frame.location.score << ",\"mode\":\"" << escape_json(frame.location.mode) << "\"}";
    }
    else {
        output << "null";
    }
    output << ",\"angle\":" << optional_number(frame.angle.angle) << ",\"pitch\":" << optional_number(frame.location.camera_pitch)
           << ",\"angleConfidence\":" << frame.angle.confidence << ",\"route\":" << route_payload_json(route_.snapshot())
           << ",\"timestamp\":" << timestamp << '}';
    return output.str();
}

} // namespace navi
