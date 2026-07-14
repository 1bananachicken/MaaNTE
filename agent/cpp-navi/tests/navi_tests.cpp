#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <boost/asio.hpp>
#include <meojson/json.hpp>

#include "coordinate.h"
#include "route.h"
#include "websocket_service.h"

int main()
{
    using namespace navi;

    const auto map_point = CoordinateTransform::to_map(-134394.56, 199913.53, 11416.17);
    assert(map_point.has_value());
    const auto raw_point = CoordinateTransform::to_raw_xy(*map_point);
    assert(raw_point.has_value());
    assert(std::abs(raw_point->first - (-134394.56)) < 40.0);
    assert(std::abs(raw_point->second - 199913.53) < 40.0);

    const auto parsed = json::parse(R"({
        "routes": [{
            "name": "main",
            "segments": [{
                "points": [
                    {"pixelX": 1000, "pixelY": 2000},
                    {"lat": 51.8, "lng": -29.0}
                ]
            }]
        }]
    })");
    assert(parsed.has_value());
    const auto route = parse_route_segment(*parsed, "main", 1, { 11264, 11264 }, { 11264, 11264 });
    assert(route.size() == 2);
    assert((route[0] == Point { 1000, 2000 }));

    RouteSession session;
    session.reset({ { 10, 10 }, { 100, 100 }, { 200, 200 } }, true, Point { 90, 95 });
    auto snapshot = session.snapshot();
    assert(snapshot.active);
    assert(snapshot.current_index == 1);
    session.advance();
    assert(session.snapshot().current_index == 2);
    session.advance();
    assert(session.snapshot().status == "arrived");

    boost::asio::io_context io;
    boost::asio::ip::tcp::acceptor bridge_acceptor(
        io,
        boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    const auto bridge_port = bridge_acceptor.local_endpoint().port();
#ifdef _WIN32
    _putenv_s("MAANTE_COORDINATE_BRIDGE_PORT", std::to_string(bridge_port).c_str());
#else
    setenv("MAANTE_COORDINATE_BRIDGE_PORT", std::to_string(bridge_port).c_str(), 1);
#endif
    std::thread bridge_thread([&bridge_acceptor]() {
        boost::asio::io_context bridge_io;
        boost::asio::ip::tcp::socket socket(bridge_io);
        bridge_acceptor.accept(socket);
        boost::asio::streambuf input;
        for (int index = 0; index < 3; ++index) {
            boost::asio::read_until(socket, input, '\n');
            std::istream stream(&input);
            std::string request;
            std::getline(stream, request);
            const std::string response = index == 1
                                             ? R"({"ok":true,"pose":[1.0,2.0,3.0,4.0,5.0]})" "\n"
                                             : R"({"ok":true})" "\n";
            boost::asio::write(socket, boost::asio::buffer(response));
        }
    });
    CoordinateCapture capture("pcap");
    capture.start();
    const auto pose = capture.read(std::chrono::seconds(1));
    assert(pose.has_value());
    assert(pose->x == 1.0 && pose->y == 2.0 && pose->z == 3.0);
    assert(pose->pitch == 4.0 && pose->heading == 5.0);
    capture.close();
    bridge_thread.join();
    bridge_acceptor.close();

    boost::asio::ip::tcp::acceptor probe(
        io,
        boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0));
    const auto port = probe.local_endpoint().port();
    probe.close();

    NavigationWebSocketService websocket(
        session,
        port,
        []() { return Size { 11264, 11264 }; },
        []() -> std::optional<Point> { return Point { 100, 200 }; });
    websocket.start();

    boost::asio::ip::tcp::socket client(io);
    client.connect({ boost::asio::ip::address_v4::loopback(), port });
    const std::string request =
        "GET / HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    boost::asio::write(client, boost::asio::buffer(request));
    boost::asio::streambuf response;
    boost::asio::read_until(client, response, "\r\n\r\n");
    std::istream response_stream(&response);
    const std::string response_text {
        std::istreambuf_iterator<char>(response_stream),
        std::istreambuf_iterator<char>()
    };
    assert(response_text.contains("101 Switching Protocols"));
    assert(response_text.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    client.close();

    boost::asio::ip::tcp::socket pending_client(io);
    pending_client.connect({ boost::asio::ip::address_v4::loopback(), port });
    websocket.stop();
    pending_client.close();

    std::cout << "cpp-navi tests passed" << std::endl;
    return 0;
}
