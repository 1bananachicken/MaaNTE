import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from maa.context import Context

from ..Common.logger import get_logger
from .angle_predictor import AnglePredictor
from .coordinate_position import CoordinatePositionProvider
from .debug_windows import close_debug_windows, pump_debug_windows
from .map_locator import MapLocator

logger = get_logger(__name__)

_KEY_W = 87


@dataclass
class AnglePidController:
    """将角度误差平滑转换为受限的鼠标转向指令。"""

    kp: float = 0.85
    ki: float = 0.04
    kd: float = 0.10
    output_limit: float = 35.0
    integral_limit: float = 120.0
    deadband: float = 4.0
    max_dt: float = 0.25
    _integral: float = 0.0
    _last_error: float | None = None
    _last_time: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def update(self, error: float, now: float) -> float:
        if abs(error) <= self.deadband:
            self.reset()
            return 0.0

        if self._last_time is None:
            dt = self.max_dt
            derivative = 0.0
        else:
            dt = max(1e-3, min(self.max_dt, now - self._last_time))
            derivative = (
                0.0 if self._last_error is None else (error - self._last_error) / dt
            )

        if self._last_error is not None and error * self._last_error < 0:
            self._integral = 0.0
        self._integral += error * dt
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral),
        )

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))

        self._last_error = error
        self._last_time = now
        return output


class WaypointNavigator:
    """根据实时位置和朝向，将角色移动到单个地图路径点。"""

    def __init__(
        self,
        context: Context,
        *,
        angle_backend: str = "auto",
        position_backend: str = "map",
        tolerance: float = 80.0,
        max_duration: float | None = None,
        frame_interval: float = 0.1,
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
        self.frame_interval = max(0.05, float(frame_interval))
        self.turn_pixels_per_degree = 10.0
        self.max_turn_degrees = 35.0
        self.align_threshold = 4.0
        self.move_pulse = 0.12
        self.key_refresh_interval = 0.5
        self.turn_pid = AnglePidController(
            output_limit=self.max_turn_degrees,
            deadband=self.align_threshold,
        )
        self.w_down = False
        self._last_w_down_at = 0.0
        self.current_point: tuple[int, int] | None = None
        self.locator: MapLocator | None = None
        self.position_provider: CoordinatePositionProvider | None = None
        self.predictor: AnglePredictor | None = None
        try:
            self.position_provider = CoordinatePositionProvider(
                position_backend,
                debug=debug,
            )
            if self.position_provider.uses_visual_positioning():
                self.locator = MapLocator(debug=debug)
            self.predictor = AnglePredictor(
                backend=angle_backend,
                threshold=0.0,
                debug=debug,
            )
        except Exception:
            self.close()
            raise

    def update(self) -> tuple[Any, Any] | None:
        frame = self.controller.post_screencap().wait().get()
        if frame is None:
            if self.debug:
                pump_debug_windows()
            return None
        assert self.position_provider is not None
        assert self.predictor is not None
        location = self.position_provider.locate(self.locator, frame)
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
        self.turn_pid.reset()

        while not self.context.tasker.stopping:
            if deadline is not None and time.monotonic() >= deadline:
                logger.info("WaypointNavigator timeout: target=%s", target)
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
                self.turn_pid.reset()
                self.release()
                self.sleep_remaining(started)
                continue

            current_x, current_y = location.point
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.hypot(dx, dy)
            if distance <= self.tolerance:
                logger.info(
                    "WaypointNavigator arrived: target=%s current=%s distance=%.1f",
                    target,
                    location.point,
                    distance,
                )
                self.release()
                return True

            desired_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
            angle_delta = (desired_angle - angle.angle + 540.0) % 360.0 - 180.0
            turn_degrees = self.turn_pid.update(angle_delta, time.monotonic())
            turn_dx = int(round(turn_degrees * self.turn_pixels_per_degree))

            self.press_forward()
            if turn_dx != 0:
                self.controller.post_relative_move(turn_dx, 0).wait()
            if not self.sleep_interruptible(self.move_pulse):
                self.release()
                return False

            now = time.monotonic()
            if now - last_log_time >= 2.0:
                logger.info(
                    "WaypointNavigator moving: target=%s current=%s distance=%.1f "
                    "angle_delta=%.1f turn=%.1f",
                    target,
                    location.point,
                    distance,
                    angle_delta,
                    turn_degrees,
                )
                last_log_time = now

            self.sleep_remaining(started)

        self.release()
        return False

    def press_forward(self) -> None:
        now = time.monotonic()
        if (
            self.w_down
            and now - self._last_w_down_at < self.key_refresh_interval
        ):
            return
        self.controller.post_key_down(_KEY_W).wait()
        self.w_down = True
        self._last_w_down_at = now

    def release(self) -> None:
        self.turn_pid.reset()
        if self.w_down:
            self.controller.post_key_up(_KEY_W).wait()
            self.w_down = False
        self._last_w_down_at = 0.0

    def close(self) -> None:
        try:
            self.release()
        finally:
            try:
                if self.position_provider is not None:
                    self.position_provider.close()
                    self.position_provider = None
            finally:
                try:
                    if self.locator is not None:
                        self.locator.close()
                        self.locator = None
                finally:
                    try:
                        if self.predictor is not None:
                            self.predictor.close()
                            self.predictor = None
                    finally:
                        if self.debug:
                            close_debug_windows()

    def sleep_remaining(self, started: float) -> None:
        sleep_time = self.frame_interval - (time.perf_counter() - started)
        if sleep_time <= 0:
            return
        if not self.debug:
            time.sleep(sleep_time)
            return

        deadline = time.monotonic() + sleep_time
        while time.monotonic() < deadline:
            pump_debug_windows()
            time.sleep(min(0.05, abs(deadline - time.monotonic())))

    def sleep_interruptible(self, duration: float) -> bool:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self.context.tasker.stopping:
                return False
            if self.debug:
                pump_debug_windows()
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
