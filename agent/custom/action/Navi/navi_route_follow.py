import threading
import time
from dataclasses import dataclass, field
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from ..Common.logger import get_logger
from .map_locator import MapLocator
from .path_navigator import PathNavigator, load_params, parse_bool
from .websocket_backend import NavigationWebSocketPublisher

logger = get_logger(__name__)


@dataclass
class RouteState:
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    active: bool = False
    current_index: int = 0
    status: str = "waiting"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "waypoints": [
                    {"pixelX": int(x), "pixelY": int(y)} for x, y in self.waypoints
                ],
                "active": self.active,
                "currentIndex": self.current_index,
                "status": self.status,
            }

    def reset(
        self,
        waypoints: list[tuple[int, int]],
        start: bool,
        current_point: tuple[int, int] | None,
    ) -> None:
        with self.lock:
            self.waypoints = waypoints
            self.active = bool(start and waypoints)
            self.current_index = (
                self.nearest_index(current_point)
                if self.active and current_point is not None
                else 0
            )
            self.status = "running" if self.active else "ready"

    def start(self, current_point: tuple[int, int] | None) -> None:
        with self.lock:
            if not self.waypoints:
                self.active = False
                self.status = "empty"
                return
            self.current_index = (
                self.nearest_index(current_point)
                if current_point is not None
                else min(self.current_index, len(self.waypoints) - 1)
            )
            self.active = True
            self.status = "running"

    def advance(self) -> None:
        with self.lock:
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.active = False
                self.status = "arrived"

    def nearest_index(self, current_point: tuple[int, int]) -> int:
        current_x, current_y = current_point
        return min(
            range(len(self.waypoints)),
            key=lambda index: (self.waypoints[index][0] - current_x) ** 2
            + (self.waypoints[index][1] - current_y) ** 2,
        )


@AgentServer.custom_action("navi_route_follow")
class NaviRouteFollowAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            params = load_params(argv.custom_action_param)
            host = str(params.get("host", "0.0.0.0"))
            port = int(params.get("port", 14514))
            tolerance = float(params.get("tolerance", 80.0))
            angle_backend = str(params.get("angle_backend", "auto"))
            debug = parse_bool(params.get("debug", False))
        except ValueError as exc:
            logger.error("Navi route param invalid: %s", exc)
            return CustomAction.RunResult(success=False)

        route = RouteState()
        source_size = [MapLocator.MAP_SIZE]
        current_point: list[tuple[int, int] | None] = [None]
        moving_target: list[tuple[int, tuple[int, int]] | None] = [None]
        lock = threading.Lock()

        def publish_route() -> None:
            payload = route.payload()
            websocket.publish_route(
                payload["waypoints"],
                active=payload["active"],
                current_index=payload["currentIndex"],
                status=payload["status"],
            )

        def on_frame(location: Any, angle: Any) -> None:
            websocket.publish_state(
                location.point,
                score=location.score,
                mode=location.mode,
                source_size=source_size[0],
                angle=angle.angle if angle.found else None,
                angle_confidence=angle.confidence,
            )
            if location.found and location.point is not None:
                with lock:
                    current_point[0] = location.point

        def should_cancel() -> bool:
            with lock:
                target = moving_target[0]
            if target is None:
                return False
            target_index, target_point = target
            with route.lock:
                return (
                    not route.active
                    or route.current_index != target_index
                    or target_index >= len(route.waypoints)
                    or route.waypoints[target_index] != target_point
                )

        def handle_message(message: dict[str, Any]) -> dict[str, Any]:
            with lock:
                latest_point = current_point[0]
            return self.handle_route_message(
                message, route, source_size[0], latest_point
            )

        websocket = NavigationWebSocketPublisher(host, port, message_handler=handle_message)
        navigator: PathNavigator | None = None

        try:
            websocket.start()
            navigator = PathNavigator(
                context,
                angle_backend=angle_backend,
                tolerance=tolerance,
                debug=debug,
                on_frame=on_frame,
                should_cancel=should_cancel,
            )
            source_size[0] = (navigator.locator.origin_w, navigator.locator.origin_h)
            logger.info("Navi route service started: ws://%s:%s", host, port)

            while not context.tasker.stopping:
                publish_route()
                payload = route.payload()
                if not payload["active"]:
                    started = time.perf_counter()
                    navigator.update()
                    navigator.sleep_remaining(started)
                    continue

                with route.lock:
                    current_index = route.current_index
                    waypoints = list(route.waypoints)
                if current_index >= len(waypoints):
                    with route.lock:
                        route.active = False
                        route.status = "arrived"
                    continue

                target = waypoints[current_index]
                with lock:
                    moving_target[0] = (current_index, target)
                arrived = navigator.move_to(target)
                with lock:
                    moving_target[0] = None
                if arrived:
                    route.advance()

            return CustomAction.RunResult(success=False)
        except Exception as exc:
            logger.error("Navi route failed: %s", exc)
            return CustomAction.RunResult(success=False)
        finally:
            if navigator is not None:
                navigator.close()
            websocket.stop()

    @classmethod
    def handle_route_message(
        cls,
        message: dict[str, Any],
        route: RouteState,
        source_size: tuple[int, int],
        current_point: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        message_type = str(message.get("type", "")).strip()
        if message_type in ("navi-route-set", "route-set"):
            values = message.get("waypoints")
            if not isinstance(values, list):
                raise ValueError("waypoints must be a list")
            message_source_size = cls.parse_source_size(message, source_size)
            route.reset(
                [
                    cls.parse_waypoint(value, message_source_size, source_size)
                    for value in values
                ],
                bool(message.get("start", False)),
                current_point,
            )
        elif message_type in ("navi-route-add", "route-add"):
            with route.lock:
                route.waypoints.append(
                    cls.parse_waypoint(
                        message,
                        cls.parse_source_size(message, source_size),
                        source_size,
                    )
                )
                if not route.active:
                    route.status = "ready"
        elif message_type in ("navi-route-clear", "route-clear"):
            with route.lock:
                route.waypoints.clear()
                route.current_index = 0
                route.active = False
                route.status = "cleared"
        elif message_type in ("navi-route-start", "route-start"):
            route.start(current_point)
        elif message_type in ("navi-route-stop", "route-stop"):
            with route.lock:
                route.active = False
                route.status = "stopped"
        else:
            return {"type": "navi-route-ack", "ok": False, "message": "unknown type"}

        return {"type": "navi-route-ack", "ok": True, "route": route.payload()}

    @staticmethod
    def parse_waypoint(
        value: Any,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> tuple[int, int]:
        if not isinstance(value, dict):
            raise ValueError("waypoint must be an object")
        if "pixelX" in value and "pixelY" in value:
            x = float(value["pixelX"])
            y = float(value["pixelY"])
        elif "target_x" in value and "target_y" in value:
            x = float(value["target_x"])
            y = float(value["target_y"])
        elif "x" in value and "y" in value and value.get("coordinate") != "online":
            x = float(value["x"])
            y = float(value["y"])
        else:
            raise ValueError("waypoint needs pixelX/pixelY or x/y")

        source_size = NaviRouteFollowAction.parse_source_size(value, source_size)
        source_w, source_h = source_size
        target_w, target_h = target_size
        if source_w <= 0 or source_h <= 0:
            raise ValueError("waypoint source size must be positive")
        return int(round(x * target_w / source_w)), int(round(y * target_h / source_h))

    @staticmethod
    def parse_source_size(
        value: dict[str, Any], default: tuple[int, int]
    ) -> tuple[int, int]:
        if "sourceWidth" in value and "sourceHeight" in value:
            return int(value["sourceWidth"]), int(value["sourceHeight"])
        source_size = value.get("sourceSize")
        if isinstance(source_size, (list, tuple)) and len(source_size) >= 2:
            return int(source_size[0]), int(source_size[1])
        return default
