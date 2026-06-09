import json
import math
import time
from typing import Any, Callable

import cv2

from maa.context import Context

from ..Common.logger import get_logger
from .angle_predictor import AnglePredictor
from .map_locator import MapLocator

logger = get_logger(__name__)

_KEY_W = 87


class PathNavigator:
    def __init__(
        self,
        context: Context,
        *,
        angle_backend: str = "auto",
        tolerance: float = 80.0,
        max_duration: float | None = None,
        debug: bool = False,
        on_frame: Callable[[Any, Any], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.context = context
        self.controller = context.tasker.controller
        self.tolerance = tolerance
        self.max_duration = max_duration
        self.debug = debug
        self.on_frame = on_frame
        self.should_cancel = should_cancel
        self.frame_interval = 0.1
        self.turn_pixels_per_degree = 10.0
        self.max_turn_degrees = 35.0
        self.align_threshold = 12.0
        self.move_pulse = 0.12
        self.w_down = False
        self.current_point: tuple[int, int] | None = None
        self.locator = MapLocator(debug=debug)
        self.predictor = AnglePredictor(
            backend=angle_backend,
            threshold=0.0,
            debug=debug,
        )

    def update(self) -> tuple[Any, Any] | None:
        frame = self.controller.post_screencap().wait().get()
        if frame is None:
            return None
        location = self.locator.locate(frame)
        angle = self.predictor.predict(frame)
        if location.found and location.point is not None:
            self.current_point = location.point
        if self.on_frame is not None:
            self.on_frame(location, angle)
        return location, angle

    def move_to(self, target: tuple[int, int]) -> bool:
        target_x, target_y = target
        deadline = (
            time.monotonic() + self.max_duration
            if self.max_duration is not None
            else None
        )
        last_log_time = 0.0

        while not self.context.tasker.stopping:
            if deadline is not None and time.monotonic() >= deadline:
                logger.info("PathNavigator timeout: target=%s", target)
                self.release()
                return False
            if self.should_cancel is not None and self.should_cancel():
                self.release()
                return False

            started = time.perf_counter()
            result = self.update()
            if result is None:
                self.sleep_remaining(started)
                continue
            location, angle = result
            if (
                not location.found
                or location.point is None
                or not angle.found
                or angle.angle is None
            ):
                self.sleep_remaining(started)
                continue

            current_x, current_y = location.point
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.hypot(dx, dy)
            if distance <= self.tolerance:
                logger.info(
                    "PathNavigator arrived: target=%s current=%s distance=%.1f",
                    target,
                    location.point,
                    distance,
                )
                self.release()
                return True

            desired_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
            angle_delta = (desired_angle - angle.angle + 540.0) % 360.0 - 180.0
            turn_degrees = max(
                -self.max_turn_degrees,
                min(self.max_turn_degrees, angle_delta),
            )
            if abs(angle_delta) <= self.align_threshold:
                turn_degrees = 0.0
            turn_dx = int(round(turn_degrees * self.turn_pixels_per_degree))

            if not self.w_down:
                self.controller.post_key_down(_KEY_W).wait()
                self.w_down = True
            if turn_dx != 0:
                self.controller.post_relative_move(turn_dx, 0).wait()
            if not self.sleep_interruptible(self.move_pulse):
                self.release()
                return False

            now = time.monotonic()
            if now - last_log_time >= 2.0:
                logger.info(
                    "PathNavigator moving: target=%s current=%s distance=%.1f "
                    "angle_delta=%.1f",
                    target,
                    location.point,
                    distance,
                    angle_delta,
                )
                last_log_time = now

            if self.debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                self.release()
                return False
            self.sleep_remaining(started)

        self.release()
        return False

    def release(self) -> None:
        if self.w_down:
            self.controller.post_key_up(_KEY_W).wait()
            self.w_down = False

    def close(self) -> None:
        try:
            self.release()
        finally:
            if self.debug:
                cv2.destroyAllWindows()

    def sleep_remaining(self, started: float) -> None:
        sleep_time = self.frame_interval - (time.perf_counter() - started)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def sleep_interruptible(self, duration: float) -> bool:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self.context.tasker.stopping:
                return False
            time.sleep(min(0.05, abs(deadline - time.monotonic())))
        return True


def load_params(custom_action_param: Any) -> dict[str, Any]:
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    try:
        params = json.loads(custom_action_param)
    except Exception as exc:
        logger.warning("Parse custom_action_param failed: %s", exc)
        return {}
    return params if isinstance(params, dict) else {}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
