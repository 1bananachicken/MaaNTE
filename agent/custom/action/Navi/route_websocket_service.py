from typing import Any, Callable

from .route_model import RouteSession, SourceSize, Waypoint, handle_route_message
from .navigation_server import NavigationWebSocketServer


class RouteWebSocketService:
    def __init__(
        self,
        route: RouteSession,
        *,
        port: int,
        get_source_size: Callable[[], SourceSize],
        get_current_point: Callable[[], Waypoint | None],
    ) -> None:
        self.route = route
        self.get_source_size = get_source_size
        self.get_current_point = get_current_point
        self.websocket = NavigationWebSocketServer(
            port=port,
            message_handler=self.handle_message,
        )

    def start(self) -> None:
        self.websocket.start()

    def stop(self) -> None:
        self.websocket.stop()

    def publish_frame(self, location: Any, angle: Any) -> None:
        self.websocket.publish_state(
            location.raw_coordinate,
            map_point=location.point,
            score=location.score,
            mode=location.mode,
            source_size=self.get_source_size(),
            angle=angle.angle if angle.found else None,
            angle_confidence=angle.confidence,
        )

    def publish_route(self) -> None:
        payload = self.route.payload()
        self.websocket.publish_route(
            payload["waypoints"],
            active=payload["active"],
            current_index=payload["currentIndex"],
            status=payload["status"],
        )

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return handle_route_message(
            message,
            self.route,
            self.get_source_size(),
            self.get_current_point(),
        )
