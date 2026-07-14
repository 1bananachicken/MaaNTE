#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "types.h"

namespace navi
{

struct PacketFlow
{
    std::string source;
    uint16_t source_port = 0;
    std::string destination;
    uint16_t destination_port = 0;
    std::string protocol;

    friend bool operator==(const PacketFlow&, const PacketFlow&) = default;
};

class CoordinateTransform
{
public:
    static constexpr Size MapSize { 11264, 11264 };

    static std::optional<Point> to_map(double x, double y, std::optional<double> z = std::nullopt);
    static std::optional<std::pair<double, double>> to_raw_xy(Point point);
};

class CoordinateDecoder
{
public:
    std::optional<RawPose> decode(std::span<const uint8_t> payload, double timestamp, const PacketFlow& flow);

private:
    struct Candidate
    {
        float client_time = 0.0F;
        size_t bit_offset = 0;
        std::array<double, 3> acceleration {};
        std::array<double, 3> location {};
    };

    std::vector<Candidate> candidates(std::span<const uint8_t> payload) const;
    std::optional<Candidate> new_flow_candidate(const std::vector<Candidate>& values) const;
    std::optional<Candidate> confirm_flow(const PacketFlow& flow, const Candidate& candidate, double timestamp);
    Candidate fresh(const std::vector<Candidate>& values) const;
    void clear_pending();

    std::optional<PacketFlow> flow_;
    std::optional<size_t> last_offset_;
    std::optional<double> last_capture_;
    std::optional<double> last_time_;
    std::optional<std::array<double, 3>> last_location_;
    std::optional<PacketFlow> pending_flow_;
    std::optional<Candidate> pending_candidate_;
    int pending_seen_ = 0;
    std::optional<double> pending_at_;
};

class CoordinateCapture
{
public:
    explicit CoordinateCapture(std::string backend);
    ~CoordinateCapture();
    CoordinateCapture(const CoordinateCapture&) = delete;
    CoordinateCapture& operator=(const CoordinateCapture&) = delete;

    void start();
    std::optional<RawPose> read(std::chrono::duration<double> max_age) const;
    std::string stats() const;
    void close();

private:
    class Backend;
    void accept_packet(std::span<const uint8_t> payload, double timestamp, PacketFlow flow);

    std::string backend_name_;
    std::unique_ptr<Backend> backend_;
    CoordinateDecoder decoder_;
    mutable std::mutex mutex_;
    std::optional<RawPose> sample_;
    double sample_timestamp_ = 0.0;
    uint64_t packet_count_ = 0;
    uint64_t payload_count_ = 0;
    uint64_t s2c_count_ = 0;
    uint64_t sample_count_ = 0;
    std::chrono::system_clock::time_point last_packet_wall_ {};
    std::chrono::system_clock::time_point last_payload_wall_ {};
    std::chrono::system_clock::time_point last_sample_wall_ {};
};

} // namespace navi
