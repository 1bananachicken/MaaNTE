import threading
import time
from typing import Any, Callable

from maa.context import Context

from .map_locator import MapLocator
from .waypoint_navigator import WaypointNavigator
from .route_model import RouteSession, SourceSize, Waypoint


class RouteRunner:
    """按顺序执行可变路线会话中的路径点。"""

    def __init__(
        self,
        context: Context,
        route: RouteSession,
        *,
        angle_backend: str = "auto",
        tolerance: float = 80.0,
        frame_interval: float = 0.1,
        debug: bool = False,
        on_frame: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.context = context
        self.route = route
        self.angle_backend = angle_backend
        self.tolerance = tolerance
        self.frame_interval = max(0.05, float(frame_interval))
        self.debug = debug
        self.on_frame = on_frame
        self.navigator: WaypointNavigator | None = None
        self._source_size: SourceSize = MapLocator.MAP_SIZE
        self._current_point: Waypoint | None = None
        self._moving_target: tuple[int, Waypoint] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.navigator is not None:
            return
        self.navigator = WaypointNavigator(
            self.context,
            angle_backend=self.angle_backend,
            tolerance=self.tolerance,
            frame_interval=self.frame_interval,
            debug=self.debug,
            on_frame=self._on_frame,
            should_cancel=self._should_cancel,
        )
        self._source_size = (
            self.navigator.locator.origin_w,
            self.navigator.locator.origin_h,
        )

    def run_until_stopped(
        self,
        *,
        on_tick: Callable[[], None] | None = None,
        stop_when_route_done: bool = False,
    ) -> str:
        self.start()
        assert self.navigator is not None

        while not self.context.tasker.stopping:
            if on_tick is not None:
                on_tick()
            payload = self.route.payload()
            if not payload["active"]:
                if stop_when_route_done and payload["status"] in {
                    "arrived",
                    "cleared",
                    "empty",
                    "stopped",
                }:
                    return str(payload["status"])
                started = time.perf_counter()
                self.navigator.update()
                self.navigator.sleep_remaining(started)
                continue

            with self.route.lock:
                current_index = self.route.current_index
                waypoints = list(self.route.waypoints)
            if current_index >= len(waypoints):
                with self.route.lock:
                    self.route.active = False
                    self.route.status = "arrived"
                continue

            target = waypoints[current_index]
            with self._lock:
                self._moving_target = (current_index, target)
            arrived = self.navigator.move_to(target)
            with self._lock:
                self._moving_target = None
            if arrived:
                self.route.advance()

        return "stopped"

    def close(self) -> None:
        if self.navigator is not None:
            self.navigator.close()
            self.navigator = None

    def update_current_frame(self) -> Waypoint | None:
        self.start()
        assert self.navigator is not None
        self.navigator.update()
        return self.current_point()

    def source_size(self) -> SourceSize:
        return self._source_size

    def current_point(self) -> Waypoint | None:
        with self._lock:
            return self._current_point

    def _on_frame(self, location: Any, angle: Any) -> None:
        if location.found and location.point is not None:
            with self._lock:
                self._current_point = location.point
        if self.on_frame is not None:
            self.on_frame(location, angle)

    def _should_cancel(self) -> bool:
        with self._lock:
            target = self._moving_target
        if target is None:
            return False
        target_index, target_point = target
        with self.route.lock:
            return (
                not self.route.active
                or self.route.current_index != target_index
                or target_index >= len(self.route.waypoints)
                or self.route.waypoints[target_index] != target_point
            )
