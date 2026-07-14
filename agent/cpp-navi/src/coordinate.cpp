#include "coordinate.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <boost/asio.hpp>
#include <meojson/json.hpp>

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

uint16_t bridge_port()
{
    constexpr uint16_t default_port = 14515;
    const char* configured = std::getenv("MAANTE_COORDINATE_BRIDGE_PORT");
    if (configured == nullptr || *configured == '\0') {
        return default_port;
    }
    try {
        const int port = std::stoi(configured);
        if (port > 0 && port <= 65535) {
            return static_cast<uint16_t>(port);
        }
    }
    catch (const std::exception&) {
    }
    throw std::invalid_argument("MAANTE_COORDINATE_BRIDGE_PORT must be between 1 and 65535");
}

std::string response_error(const json::value& response)
{
    if (response.is_object() && response.contains("error") && response.at("error").is_string()) {
        return response.at("error").as<std::string>();
    }
    return "coordinate bridge request failed";
}

} // namespace

class CoordinateCapture::Bridge
{
public:
    explicit Bridge(const std::string& backend)
        : socket_(io_)
    {
        using tcp = boost::asio::ip::tcp;
        tcp::resolver resolver(io_);
        boost::asio::connect(socket_, resolver.resolve("127.0.0.1", std::to_string(bridge_port())));
        const json::object request {
            { "type", "start" },
            { "backend", backend },
        };
        call(request);
    }

    json::value call(const json::object& request)
    {
        const std::string payload = request.dumps() + '\n';
        boost::asio::write(socket_, boost::asio::buffer(payload));
        boost::asio::read_until(socket_, response_buffer_, '\n');

        std::istream stream(&response_buffer_);
        std::string line;
        std::getline(stream, line);
        const auto response = json::parse(line);
        if (!response || !response->is_object()) {
            throw std::runtime_error("coordinate bridge returned invalid JSON");
        }
        if (!response->contains("ok") || !response->at("ok").is_boolean() || !response->at("ok").as<bool>()) {
            throw std::runtime_error(response_error(*response));
        }
        return *response;
    }

    void close()
    {
        if (!socket_.is_open()) {
            return;
        }
        try {
            call(json::object { { "type", "close" } });
        }
        catch (const std::exception&) {
        }
        boost::system::error_code error;
        socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, error);
        socket_.close(error);
    }

private:
    boost::asio::io_context io_;
    boost::asio::ip::tcp::socket socket_;
    boost::asio::streambuf response_buffer_;
};

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
    if (!bridge_) {
        bridge_ = std::make_unique<Bridge>(backend_name_);
    }
}

std::optional<RawPose> CoordinateCapture::read(std::chrono::duration<double> max_age)
{
    if (!bridge_) {
        throw std::runtime_error("coordinate bridge is not started");
    }
    const json::value response = bridge_->call(json::object {
        { "type", "read" },
        { "max_age", max_age.count() },
    });
    if (!response.contains("pose") || response.at("pose").is_null()) {
        return std::nullopt;
    }
    if (!response.at("pose").is_array()) {
        throw std::runtime_error("coordinate bridge returned invalid pose");
    }
    const auto& pose = response.at("pose").as_array();
    if (pose.size() < 5 || !pose[0].is_number() || !pose[1].is_number() || !pose[2].is_number() || !pose[3].is_number()
        || !pose[4].is_number()) {
        throw std::runtime_error("coordinate bridge returned invalid pose");
    }
    return RawPose {
        pose[0].as<double>(),
        pose[1].as<double>(),
        pose[2].as<double>(),
        pose[3].as<double>(),
        pose[4].as<double>(),
    };
}

std::string CoordinateCapture::stats() const
{
    return "backend=" + backend_name_ + " source=protected-bridge";
}

void CoordinateCapture::close()
{
    if (bridge_) {
        bridge_->close();
        bridge_.reset();
    }
}

} // namespace navi
