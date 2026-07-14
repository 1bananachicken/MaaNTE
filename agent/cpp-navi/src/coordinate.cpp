#include "coordinate.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <MaaUtils/Logger.h>

#include "util.h"

#ifdef _WIN32
#include <MaaUtils/SafeWindows.hpp>
#include <Ws2tcpip.h>
#endif

namespace navi
{

namespace
{

// BEGIN GENERATED NAVI COORDINATE TRANSFORM
constexpr std::array<int, 2> kCalibrationAxes { 0, 1 };
constexpr double kCalibrationA = 0.016394586684750773;
constexpr double kCalibrationB = 5.693519256055879e-08;
constexpr double kCalibrationTx = 6293.474380746091;
constexpr double kCalibrationTy = 3472.664390686138;
// END GENERATED NAVI COORDINATE TRANSFORM
constexpr std::array<double, 3> kNorth { -0.013752068070295848, -0.9999054358407049, 0.0 };
constexpr std::array<double, 3> kEast { 0.9999054358407049, -0.01375206807029585, 0.0 };
constexpr double kPi = 3.14159265358979323846;

uint64_t bits(std::span<const uint8_t> data, size_t offset, size_t count)
{
    if (count > 64 || offset + count > data.size() * 8) {
        throw std::out_of_range("bit range is outside payload");
    }
    uint64_t result = 0;
    for (size_t index = 0; index < count; ++index) {
        const size_t bit = offset + index;
        result |= static_cast<uint64_t>((data[bit / 8] >> (bit % 8)) & 1U) << index;
    }
    return result;
}

struct VectorRead
{
    std::array<double, 3> values {};
    size_t offset = 0;
    int width = 0;
    bool scaled = false;
};

VectorRead read_vector(std::span<const uint8_t> data, size_t offset, int scale)
{
    const int header = static_cast<int>(bits(data, offset, 7));
    offset += 7;
    const int width = header & 63;
    const bool scaled = (header >> 6) != 0;
    if (width == 0 || width > 63) {
        throw std::invalid_argument("unsupported vector width");
    }
    VectorRead output;
    output.width = width;
    output.scaled = scaled;
    const uint64_t sign = uint64_t { 1 } << (width - 1);
    const uint64_t modulus = uint64_t { 1 } << width;
    for (double& value : output.values) {
        uint64_t raw = bits(data, offset, static_cast<size_t>(width));
        offset += static_cast<size_t>(width);
        const int64_t signed_value = (raw & sign) != 0 ? static_cast<int64_t>(raw - modulus) : static_cast<int64_t>(raw);
        value = scaled ? static_cast<double>(signed_value) / scale : static_cast<double>(signed_value);
    }
    output.offset = offset;
    return output;
}

std::pair<std::array<double, 3>, size_t> read_rotator(std::span<const uint8_t> data, size_t offset)
{
    std::array<double, 3> values {};
    for (double& value : values) {
        const bool present = bits(data, offset, 1) != 0;
        ++offset;
        const uint64_t compressed = present ? bits(data, offset, 16) : 0;
        offset += present ? 16 : 0;
        value = static_cast<double>(compressed) * 360.0 / 65536.0;
        if (value > 180.0) {
            value -= 360.0;
        }
    }
    return { values, offset };
}

double distance_sq(const std::array<double, 3>& left, const std::array<double, 3>& right)
{
    double result = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        result += std::pow(left[index] - right[index], 2.0);
    }
    return result;
}

RawPose make_pose(const std::array<double, 3>& location, const std::array<double, 3>& rotation)
{
    const double pitch = rotation[0];
    const double pitch_radians = pitch * kPi / 180.0;
    const double yaw_radians = rotation[1] * kPi / 180.0;
    const std::array<double, 3> view {
        std::cos(pitch_radians) * std::cos(yaw_radians),
        std::cos(pitch_radians) * std::sin(yaw_radians),
        std::sin(pitch_radians),
    };
    const double north = std::inner_product(view.begin(), view.end(), kNorth.begin(), 0.0);
    const double east = std::inner_product(view.begin(), view.end(), kEast.begin(), 0.0);
    double heading = std::atan2(east, north) * 180.0 / kPi;
    heading = std::fmod(heading + 360.0, 360.0);
    return { location[0], location[1], location[2], pitch, heading };
}

enum class Direction
{
    ClientToServer,
    ServerToClient,
    Unknown,
};

#ifdef _WIN32
bool ipv4_local(uint32_t address)
{
    const uint32_t host = ntohl(address);
    const uint8_t a = static_cast<uint8_t>(host >> 24);
    const uint8_t b = static_cast<uint8_t>(host >> 16);
    return a == 10 || a == 127 || (a == 169 && b == 254) || (a == 172 && b >= 16 && b <= 31) || (a == 192 && b == 168)
           || (a == 198 && (b == 18 || b == 19)) || a == 0 || a >= 224;
}

bool address_local(const std::string& value)
{
    IN_ADDR v4 {};
    if (InetPtonA(AF_INET, value.c_str(), &v4) == 1) {
        return ipv4_local(v4.S_un.S_addr);
    }
    IN6_ADDR v6 {};
    if (InetPtonA(AF_INET6, value.c_str(), &v6) == 1) {
        return IN6_IS_ADDR_LOOPBACK(&v6) || IN6_IS_ADDR_LINKLOCAL(&v6) || (v6.u.Byte[0] & 0xFEU) == 0xFCU;
    }
    return false;
}
#else
bool address_local(const std::string&)
{
    return false;
}
#endif

Direction packet_direction(const PacketFlow& flow)
{
    if (flow.source.empty() || flow.destination.empty()) {
        return Direction::Unknown;
    }
    const bool source_local = address_local(flow.source);
    const bool destination_local = address_local(flow.destination);
    if (source_local && !destination_local) {
        return Direction::ClientToServer;
    }
    if (destination_local && !source_local) {
        return Direction::ServerToClient;
    }
    return Direction::Unknown;
}

double unix_now()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

std::optional<Point> CoordinateTransform::to_map(double x, double y, std::optional<double> z)
{
    const std::array<double, 3> point { x, y, z.value_or(0.0) };
    const double raw_x = point[kCalibrationAxes[0]];
    const double raw_y = point[kCalibrationAxes[1]];
    const double map_x = kCalibrationA * raw_x - kCalibrationB * raw_y + kCalibrationTx;
    const double map_y = kCalibrationB * raw_x + kCalibrationA * raw_y + kCalibrationTy;
    if (!std::isfinite(map_x) || !std::isfinite(map_y)) {
        return std::nullopt;
    }
    return Point { static_cast<int>(std::lround(map_x)), static_cast<int>(std::lround(map_y)) };
}

std::optional<std::pair<double, double>> CoordinateTransform::to_raw_xy(Point point)
{
    if (kCalibrationAxes != std::array<int, 2> { 0, 1 }) {
        return std::nullopt;
    }
    const double denominator = kCalibrationA * kCalibrationA + kCalibrationB * kCalibrationB;
    if (denominator <= 1e-12) {
        return std::nullopt;
    }
    const double dx = point.x - kCalibrationTx;
    const double dy = point.y - kCalibrationTy;
    return std::pair {
        (kCalibrationA * dx + kCalibrationB * dy) / denominator,
        (-kCalibrationB * dx + kCalibrationA * dy) / denominator,
    };
}

std::optional<RawPose> CoordinateDecoder::decode(std::span<const uint8_t> payload, double timestamp, const PacketFlow& flow)
{
    const auto values = candidates(payload);
    if (values.empty()) {
        return std::nullopt;
    }

    Candidate selected;
    if (flow_ && flow != *flow_) {
        const auto candidate = new_flow_candidate(values);
        if (!candidate) {
            return std::nullopt;
        }
        const auto confirmed = confirm_flow(flow, *candidate, timestamp);
        if (!confirmed) {
            return std::nullopt;
        }
        selected = *confirmed;
        clear_pending();
        flow_ = flow;
    }
    else if (!last_time_ || !last_capture_) {
        const auto candidate = new_flow_candidate(values);
        if (!candidate) {
            return std::nullopt;
        }
        selected = *candidate;
        clear_pending();
        flow_ = flow;
    }
    else {
        const double gap = std::max(0.0, timestamp - *last_capture_);
        const double expected = *last_time_ + gap;
        std::vector<Candidate> aligned;
        if (last_offset_) {
            std::ranges::copy_if(values, std::back_inserter(aligned), [this](const Candidate& item) { return item.bit_offset == *last_offset_; });
        }
        const auto& tracking = aligned.empty() ? values : aligned;
        selected = *std::ranges::min_element(tracking, [expected](const Candidate& left, const Candidate& right) {
            return std::abs(left.client_time - expected) < std::abs(right.client_time - expected);
        });
        if (std::abs(selected.client_time - expected) > 1.0) {
            std::vector<Candidate> plausible;
            std::ranges::copy_if(tracking, std::back_inserter(plausible), [](const Candidate& item) {
                const double max_acceleration = std::ranges::max(item.acceleration | std::views::transform([](double value) { return std::abs(value); }));
                const double max_location = std::ranges::max(item.location | std::views::transform([](double value) { return std::abs(value); }));
                return item.client_time >= 0.05F && item.bit_offset <= 260 && max_acceleration <= 10000.0 && max_location <= 500000.0;
            });
            if (plausible.empty()) {
                return std::nullopt;
            }
            selected = fresh(plausible);
            clear_pending();
            flow_ = flow;
        }
        else {
            clear_pending();
        }
    }

    try {
        const auto acceleration = read_vector(payload, selected.bit_offset + 32, 10);
        const auto location = read_vector(payload, acceleration.offset, 100);
        const auto [rotation, unused] = read_rotator(payload, location.offset);
        (void)unused;
        last_time_ = selected.client_time;
        last_offset_ = selected.bit_offset;
        last_capture_ = timestamp;
        last_location_ = selected.location;
        return make_pose(selected.location, rotation);
    }
    catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<CoordinateDecoder::Candidate> CoordinateDecoder::candidates(std::span<const uint8_t> payload) const
{
    std::vector<Candidate> output;
    if (payload.size() * 8 <= 257) {
        return output;
    }
    const size_t end = std::min<size_t>(320, payload.size() * 8 - 60);
    for (size_t offset = 197; offset < end; ++offset) {
        try {
            const uint32_t raw_time = static_cast<uint32_t>(bits(payload, offset, 32));
            const float client_time = std::bit_cast<float>(raw_time);
            const auto acceleration = read_vector(payload, offset + 32, 10);
            const auto location = read_vector(payload, acceleration.offset, 100);
            const double max_acceleration = std::ranges::max(acceleration.values | std::views::transform([](double value) { return std::abs(value); }));
            const double max_location = std::ranges::max(location.values | std::views::transform([](double value) { return std::abs(value); }));
            if (!std::isfinite(client_time) || client_time < 0.0F || client_time >= 100000.0F || !acceleration.scaled || !location.scaled
                || acceleration.width < 1 || acceleration.width > 24 || location.width < 16 || location.width > 40
                || max_acceleration >= 50000.0 || max_location >= 20000000.0) {
                continue;
            }
            output.push_back({ client_time, offset, acceleration.values, location.values });
        }
        catch (const std::exception&) {
        }
    }
    return output;
}

std::optional<CoordinateDecoder::Candidate> CoordinateDecoder::new_flow_candidate(const std::vector<Candidate>& values) const
{
    std::vector<Candidate> valid;
    std::ranges::copy_if(values, std::back_inserter(valid), [](const Candidate& item) {
        const double max_acceleration = std::ranges::max(item.acceleration | std::views::transform([](double value) { return std::abs(value); }));
        const double max_location = std::ranges::max(item.location | std::views::transform([](double value) { return std::abs(value); }));
        return item.client_time >= 0.05F && item.bit_offset <= 260 && max_acceleration <= 10000.0 && max_location <= 500000.0;
    });
    if (valid.empty()) {
        return std::nullopt;
    }
    return *std::ranges::max_element(valid, {}, &Candidate::client_time);
}

std::optional<CoordinateDecoder::Candidate> CoordinateDecoder::confirm_flow(
    const PacketFlow& flow,
    const Candidate& candidate,
    double timestamp)
{
    if (!pending_flow_ || *pending_flow_ != flow || !pending_candidate_) {
        pending_flow_ = flow;
        pending_candidate_ = candidate;
        pending_seen_ = 1;
        pending_at_ = timestamp;
        return std::nullopt;
    }
    const Candidate previous = *pending_candidate_;
    const double gap = std::max(0.0, timestamp - pending_at_.value_or(timestamp));
    const double time_delta = candidate.client_time - previous.client_time;
    const bool time_ok = time_delta >= 0.001 && std::abs(time_delta - gap) <= 0.5;
    const bool offset_ok = candidate.bit_offset == previous.bit_offset;
    const bool step_ok = distance_sq(candidate.location, previous.location) <= 6400000000.0;
    pending_seen_ = time_ok && offset_ok && step_ok ? pending_seen_ + 1 : 1;
    pending_candidate_ = candidate;
    pending_at_ = timestamp;
    return pending_seen_ >= 2 ? std::optional(candidate) : std::nullopt;
}

CoordinateDecoder::Candidate CoordinateDecoder::fresh(const std::vector<Candidate>& values) const
{
    if (!last_location_) {
        return *std::ranges::max_element(values, {}, &Candidate::client_time);
    }
    return *std::ranges::min_element(values, [this](const Candidate& left, const Candidate& right) {
        return distance_sq(left.location, *last_location_) < distance_sq(right.location, *last_location_);
    });
}

void CoordinateDecoder::clear_pending()
{
    pending_flow_.reset();
    pending_candidate_.reset();
    pending_seen_ = 0;
    pending_at_.reset();
}

class CoordinateCapture::Backend
{
public:
    using Callback = std::function<void(std::span<const uint8_t>, double, PacketFlow)>;

    Backend(std::string name, Callback callback)
        : name_(std::move(name))
        , callback_(std::move(callback))
    {
    }

    ~Backend() { stop(); }

    void start()
    {
#ifdef _WIN32
        if (name_ == "pcap") {
            start_pcap();
        }
        else if (name_ == "pktmon") {
            start_pktmon();
        }
        else {
            throw std::invalid_argument("capture_backend must be pcap or pktmon");
        }
#else
        throw std::runtime_error("coordinate capture is currently supported on Windows only");
#endif
    }

    void stop()
    {
        stopping_ = true;
#ifdef _WIN32
        for (std::thread& thread : pcap_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        pcap_threads_.clear();
#endif
        if (thread_.joinable()) {
            thread_.join();
        }
#ifdef _WIN32
        if (pcap_close_ != nullptr) {
            for (const PcapCapture& capture : pcap_captures_) {
                pcap_close_(capture.handle);
            }
        }
        pcap_captures_.clear();
        if (pcap_module_ != nullptr) {
            FreeLibrary(pcap_module_);
            pcap_module_ = nullptr;
        }
        if (pktmon_handle_ != nullptr) {
            if (pktmon_stop_ != nullptr) {
                pktmon_stop_(pktmon_handle_);
            }
            if (pktmon_destroy_ != nullptr) {
                pktmon_destroy_(pktmon_handle_);
            }
            pktmon_handle_ = nullptr;
        }
        if (pktmon_module_ != nullptr) {
            FreeLibrary(pktmon_module_);
            pktmon_module_ = nullptr;
        }
#endif
    }

    std::string error() const { return error_; }

private:
#ifdef _WIN32
    struct PcapIf
    {
        PcapIf* next;
        char* name;
        char* description;
        void* addresses;
        uint32_t flags;
    };
    struct PcapHeader
    {
        timeval timestamp;
        uint32_t captured_length;
        uint32_t length;
    };
    struct BpfProgram
    {
        uint32_t length;
        void* instructions;
    };
    struct NativePacket
    {
        uint32_t struct_size;
        double timestamp_unix;
        uint32_t protocol;
        uint16_t source_port;
        uint16_t destination_port;
        char source_address[64];
        char destination_address[64];
        const uint8_t* payload;
        uint32_t payload_size;
    };

    using PcapFindAllDevs = int(__cdecl*)(PcapIf**, char*);
    using PcapFreeAllDevs = void(__cdecl*)(PcapIf*);
    using PcapOpenLive = void*(__cdecl*)(const char*, int, int, int, char*);
    using PcapCompile = int(__cdecl*)(void*, BpfProgram*, const char*, int, uint32_t);
    using PcapSetFilter = int(__cdecl*)(void*, BpfProgram*);
    using PcapFreeCode = void(__cdecl*)(BpfProgram*);
    using PcapNextEx = int(__cdecl*)(void*, PcapHeader**, const uint8_t**);
    using PcapClose = void(__cdecl*)(void*);
    using PcapDataLink = int(__cdecl*)(void*);

    struct PcapCapture
    {
        void* handle = nullptr;
        int link_type = 0;
        std::string name;
    };

    void start_pcap()
    {
        pcap_module_ = LoadLibraryW(L"wpcap.dll");
        if (pcap_module_ == nullptr) {
            throw std::runtime_error("Npcap/WinPcap wpcap.dll is unavailable");
        }
        const auto find_all = reinterpret_cast<PcapFindAllDevs>(GetProcAddress(pcap_module_, "pcap_findalldevs"));
        const auto free_all = reinterpret_cast<PcapFreeAllDevs>(GetProcAddress(pcap_module_, "pcap_freealldevs"));
        const auto open_live = reinterpret_cast<PcapOpenLive>(GetProcAddress(pcap_module_, "pcap_open_live"));
        const auto compile = reinterpret_cast<PcapCompile>(GetProcAddress(pcap_module_, "pcap_compile"));
        const auto set_filter = reinterpret_cast<PcapSetFilter>(GetProcAddress(pcap_module_, "pcap_setfilter"));
        const auto free_code = reinterpret_cast<PcapFreeCode>(GetProcAddress(pcap_module_, "pcap_freecode"));
        pcap_next_ex_ = reinterpret_cast<PcapNextEx>(GetProcAddress(pcap_module_, "pcap_next_ex"));
        pcap_close_ = reinterpret_cast<PcapClose>(GetProcAddress(pcap_module_, "pcap_close"));
        const auto data_link = reinterpret_cast<PcapDataLink>(GetProcAddress(pcap_module_, "pcap_datalink"));
        if (!find_all || !free_all || !open_live || !compile || !set_filter || !free_code || !pcap_next_ex_ || !pcap_close_ || !data_link) {
            throw std::runtime_error("wpcap.dll is missing required exports");
        }
        std::array<char, 256> error {};
        PcapIf* devices = nullptr;
        if (find_all(&devices, error.data()) != 0 || devices == nullptr) {
            throw std::runtime_error(std::string("pcap_findalldevs failed: ") + error.data());
        }
        std::string failures;
        for (PcapIf* device = devices; device != nullptr; device = device->next) {
            if (device->name == nullptr || (device->flags & 1U) != 0) {
                continue;
            }
            error.fill('\0');
            void* handle = open_live(device->name, 65536, 1, 20, error.data());
            if (handle == nullptr) {
                failures += std::string(device->name) + ": " + error.data() + "; ";
                continue;
            }
            BpfProgram program {};
            const bool compiled = compile(handle, &program, "tcp port 30031 or udp", 1, 0xFFFFFFFFU) == 0;
            const bool filtered = compiled && set_filter(handle, &program) == 0;
            if (compiled) {
                free_code(&program);
            }
            if (!filtered) {
                failures += std::string(device->name) + ": failed to apply filter; ";
                pcap_close_(handle);
                continue;
            }
            const int link_type = data_link(handle);
            if (link_type != 1 && link_type != 12 && link_type != 228 && link_type != 229) {
                failures += std::string(device->name) + ": unsupported link type " + std::to_string(link_type) + "; ";
                pcap_close_(handle);
                continue;
            }
            pcap_captures_.push_back({ handle, link_type, device->name });
        }
        free_all(devices);
        if (pcap_captures_.empty()) {
            throw std::runtime_error("pcap could not open a usable interface: " + failures);
        }
        stopping_ = false;
        for (const PcapCapture& capture : pcap_captures_) {
            LogInfo << "Navi pcap interface opened" << VAR(capture.name) << VAR(capture.link_type);
            pcap_threads_.emplace_back([this, handle = capture.handle, link_type = capture.link_type]() { pcap_loop(handle, link_type); });
        }
    }

    static bool parse_packet(std::span<const uint8_t> packet, int link_type, std::span<const uint8_t>& payload, PacketFlow& flow)
    {
        size_t offset = 0;
        uint16_t ether_type = 0;
        if (link_type == 1) {
            if (packet.size() < 14) {
                return false;
            }
            offset = 14;
            ether_type = static_cast<uint16_t>((packet[12] << 8) | packet[13]);
            if ((ether_type == 0x8100 || ether_type == 0x88A8) && packet.size() >= 18) {
                ether_type = static_cast<uint16_t>((packet[16] << 8) | packet[17]);
                offset = 18;
            }
        }
        else if (link_type == 12) {
            if (packet.empty()) {
                return false;
            }
            ether_type = (packet[0] >> 4) == 4 ? 0x0800 : (packet[0] >> 4) == 6 ? 0x86DD : 0;
        }
        else if (link_type == 228 || link_type == 229) {
            ether_type = link_type == 228 ? 0x0800 : 0x86DD;
        }
        else {
            return false;
        }
        uint8_t protocol = 0;
        if (ether_type == 0x0800 && packet.size() >= offset + 20) {
            const size_t header_length = static_cast<size_t>(packet[offset] & 0x0FU) * 4;
            if (header_length < 20 || packet.size() < offset + header_length) {
                return false;
            }
            protocol = packet[offset + 9];
            IN_ADDR source {};
            IN_ADDR destination {};
            std::memcpy(&source, packet.data() + offset + 12, 4);
            std::memcpy(&destination, packet.data() + offset + 16, 4);
            std::array<char, 64> text {};
            InetNtopA(AF_INET, &source, text.data(), static_cast<DWORD>(text.size()));
            flow.source = text.data();
            InetNtopA(AF_INET, &destination, text.data(), static_cast<DWORD>(text.size()));
            flow.destination = text.data();
            offset += header_length;
        }
        else if (ether_type == 0x86DD && packet.size() >= offset + 40) {
            protocol = packet[offset + 6];
            IN6_ADDR source {};
            IN6_ADDR destination {};
            std::memcpy(&source, packet.data() + offset + 8, 16);
            std::memcpy(&destination, packet.data() + offset + 24, 16);
            std::array<char, 64> text {};
            InetNtopA(AF_INET6, &source, text.data(), static_cast<DWORD>(text.size()));
            flow.source = text.data();
            InetNtopA(AF_INET6, &destination, text.data(), static_cast<DWORD>(text.size()));
            flow.destination = text.data();
            offset += 40;
        }
        else {
            return false;
        }
        if (protocol == 6 && packet.size() >= offset + 20) {
            flow.protocol = "TCP";
            flow.source_port = static_cast<uint16_t>((packet[offset] << 8) | packet[offset + 1]);
            flow.destination_port = static_cast<uint16_t>((packet[offset + 2] << 8) | packet[offset + 3]);
            const size_t header_length = static_cast<size_t>(packet[offset + 12] >> 4) * 4;
            if (header_length < 20 || packet.size() < offset + header_length) {
                return false;
            }
            offset += header_length;
        }
        else if (protocol == 17 && packet.size() >= offset + 8) {
            flow.protocol = "UDP";
            flow.source_port = static_cast<uint16_t>((packet[offset] << 8) | packet[offset + 1]);
            flow.destination_port = static_cast<uint16_t>((packet[offset + 2] << 8) | packet[offset + 3]);
            offset += 8;
        }
        else {
            return false;
        }
        payload = packet.subspan(offset);
        return !payload.empty();
    }

    void pcap_loop(void* handle, int link_type)
    {
        while (!stopping_) {
            PcapHeader* header = nullptr;
            const uint8_t* data = nullptr;
            const int status = pcap_next_ex_(handle, &header, &data);
            if (status == 0) {
                continue;
            }
            if (status < 0 || header == nullptr || data == nullptr) {
                break;
            }
            std::span<const uint8_t> payload;
            PacketFlow flow;
            if (parse_packet(std::span(data, header->captured_length), link_type, payload, flow)) {
                const double timestamp =
                    static_cast<double>(header->timestamp.tv_sec) + static_cast<double>(header->timestamp.tv_usec) / 1000000.0;
                callback_(payload, timestamp, std::move(flow));
            }
        }
    }

    using PktmonCreate = void*(__cdecl*)();
    using PktmonDestroy = void(__cdecl*)(void*);
    using PktmonStart = int32_t(__cdecl*)(void*, const char*);
    using PktmonRead = int32_t(__cdecl*)(void*, NativePacket*, uint32_t);
    using PktmonStop = void(__cdecl*)(void*);
    using PktmonLastError = uint32_t(__cdecl*)(void*, char*, uint32_t);

    void start_pktmon()
    {
        const auto path = find_pktmon_backend();
        if (!path) {
            throw std::runtime_error("pktmon_backend.dll is unavailable");
        }
        pktmon_module_ = LoadLibraryW(path->c_str());
        if (pktmon_module_ == nullptr) {
            throw std::runtime_error("failed to load pktmon_backend.dll");
        }
        const auto create = reinterpret_cast<PktmonCreate>(GetProcAddress(pktmon_module_, "PktmonCreate"));
        pktmon_destroy_ = reinterpret_cast<PktmonDestroy>(GetProcAddress(pktmon_module_, "PktmonDestroy"));
        const auto start = reinterpret_cast<PktmonStart>(GetProcAddress(pktmon_module_, "PktmonStart"));
        pktmon_read_ = reinterpret_cast<PktmonRead>(GetProcAddress(pktmon_module_, "PktmonRead"));
        pktmon_stop_ = reinterpret_cast<PktmonStop>(GetProcAddress(pktmon_module_, "PktmonStop"));
        pktmon_last_error_ = reinterpret_cast<PktmonLastError>(GetProcAddress(pktmon_module_, "PktmonLastError"));
        if (!create || !pktmon_destroy_ || !start || !pktmon_read_ || !pktmon_stop_ || !pktmon_last_error_) {
            throw std::runtime_error("pktmon_backend.dll is missing required exports");
        }
        pktmon_handle_ = create();
        if (pktmon_handle_ == nullptr) {
            throw std::runtime_error("PktmonCreate failed");
        }
        if (start(pktmon_handle_, "tcp port 30031 or udp") != 0) {
            throw std::runtime_error("PktmonStart failed: " + pktmon_error());
        }
        stopping_ = false;
        thread_ = std::thread([this]() { pktmon_loop(); });
    }

    std::string pktmon_error() const
    {
        const uint32_t size = pktmon_last_error_(pktmon_handle_, nullptr, 0);
        std::string buffer(std::max<uint32_t>(size, 1), '\0');
        pktmon_last_error_(pktmon_handle_, buffer.data(), static_cast<uint32_t>(buffer.size()));
        buffer.resize(std::strlen(buffer.c_str()));
        return buffer;
    }

    void pktmon_loop()
    {
        while (!stopping_) {
            NativePacket packet {};
            packet.struct_size = sizeof(packet);
            const int32_t status = pktmon_read_(pktmon_handle_, &packet, 20);
            if (status == -8 || status == -5) {
                continue;
            }
            if (status != 0) {
                error_ = pktmon_error();
                break;
            }
            PacketFlow flow {
                packet.source_address,
                packet.source_port,
                packet.destination_address,
                packet.destination_port,
                packet.protocol == 6 ? "TCP" : packet.protocol == 17 ? "UDP" : "",
            };
            callback_(std::span(packet.payload, packet.payload_size), packet.timestamp_unix, std::move(flow));
        }
    }

    HMODULE pcap_module_ = nullptr;
    std::vector<PcapCapture> pcap_captures_;
    std::vector<std::thread> pcap_threads_;
    PcapNextEx pcap_next_ex_ = nullptr;
    PcapClose pcap_close_ = nullptr;
    HMODULE pktmon_module_ = nullptr;
    void* pktmon_handle_ = nullptr;
    PktmonDestroy pktmon_destroy_ = nullptr;
    PktmonRead pktmon_read_ = nullptr;
    PktmonStop pktmon_stop_ = nullptr;
    PktmonLastError pktmon_last_error_ = nullptr;
#endif

    std::string name_;
    Callback callback_;
    std::atomic_bool stopping_ = false;
    std::thread thread_;
    std::string error_;
};

CoordinateCapture::CoordinateCapture(std::string backend)
    : backend_name_(std::move(backend))
{
}

CoordinateCapture::~CoordinateCapture()
{
    close();
}

void CoordinateCapture::start()
{
    if (backend_) {
        return;
    }
    backend_ = std::make_unique<Backend>(backend_name_, [this](std::span<const uint8_t> payload, double timestamp, PacketFlow flow) {
        accept_packet(payload, timestamp, std::move(flow));
    });
    backend_->start();
}

std::optional<RawPose> CoordinateCapture::read(std::chrono::duration<double> max_age) const
{
    std::scoped_lock lock(mutex_);
    if (!sample_ || unix_now() - sample_timestamp_ > max_age.count()) {
        return std::nullopt;
    }
    return sample_;
}

std::string CoordinateCapture::stats() const
{
    std::scoped_lock lock(mutex_);
    std::ostringstream output;
    output << "backend=" << backend_name_ << " packets=" << packet_count_ << " payloads=" << payload_count_ << " s2c=" << s2c_count_
           << " samples=" << sample_count_;
    if (backend_ && !backend_->error().empty()) {
        output << " error=" << backend_->error();
    }
    return output.str();
}

void CoordinateCapture::close()
{
    if (backend_) {
        backend_->stop();
        backend_.reset();
    }
}

void CoordinateCapture::accept_packet(std::span<const uint8_t> payload, double timestamp, PacketFlow flow)
{
    const Direction direction = packet_direction(flow);
    std::scoped_lock lock(mutex_);
    ++packet_count_;
    last_packet_wall_ = std::chrono::system_clock::now();
    if (!payload.empty()) {
        ++payload_count_;
        last_payload_wall_ = last_packet_wall_;
    }
    if (direction == Direction::ServerToClient) {
        ++s2c_count_;
    }
    if (payload.empty() || direction == Direction::ServerToClient) {
        return;
    }
    const auto sample = decoder_.decode(payload, timestamp, flow);
    if (!sample) {
        return;
    }
    sample_ = sample;
    sample_timestamp_ = timestamp;
    ++sample_count_;
    last_sample_wall_ = std::chrono::system_clock::now();
}

} // namespace navi
