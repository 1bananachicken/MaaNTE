from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..Common.logger import get_logger
from .map_locator import MapLocationResult

logger = get_logger(__name__)

_RawPoint = tuple[float, float, float]
_RawPose = tuple[float, float, float, float, float]
_MapPoint = tuple[int, int]
COORDINATE_MAP_SIZE = (11264, 11264)
_CORE_MODULE = "nte_coordinate_api"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_THIRDPARTY_DIR = _PROJECT_ROOT / "thirdparty"

# BEGIN GENERATED NAVI COORDINATE TRANSFORM
_CALIBRATION_AXES = (0, 1)
_CALIBRATION_A = 0.016394586684750773
_CALIBRATION_B = 5.693519256055879e-08
_CALIBRATION_TX = 6293.474380746091
_CALIBRATION_TY = 3472.664390686138
_CALIBRATION_ERROR = 0.22031967781665318
# END GENERATED NAVI COORDINATE TRANSFORM


class _CoordinateCapture(Protocol):
    def start(self) -> None: ...

    def read(self, max_age: float = 1.0) -> _RawPose | None: ...

    def close(self) -> None: ...


def _create_capture() -> _CoordinateCapture:
    if not _THIRDPARTY_DIR.is_dir():
        raise RuntimeError("coordinate core directory not found: %s" % _THIRDPARTY_DIR)
    thirdparty_path = str(_THIRDPARTY_DIR)
    if thirdparty_path not in sys.path:
        sys.path.insert(0, thirdparty_path)

    try:
        module = importlib.import_module(_CORE_MODULE)
    except Exception as exc:
        raise RuntimeError(
            "coordinate core import failed: module=%s path=%s "
            "python=%s.%s executable=%s error=%s: %s"
            % (
                _CORE_MODULE,
                _THIRDPARTY_DIR,
                sys.version_info.major,
                sys.version_info.minor,
                sys.executable,
                type(exc).__name__,
                exc,
            )
        ) from exc

    try:
        capture_type: Any = getattr(module, "CoordinateCapture")
    except AttributeError as exc:
        raise RuntimeError(
            "coordinate core loaded from %s but does not export CoordinateCapture"
            % getattr(module, "__file__", "<unknown>")
        ) from exc

    api_version = getattr(module, "API_VERSION", None)
    if api_version != "1.1.1":
        raise RuntimeError(
            "coordinate core API 1.1.1 is required, got %s"
            % (api_version or "<unknown>")
        )

    capture = capture_type(refresh_rate=0)
    for method_name in ("start", "read", "close"):
        if not callable(getattr(capture, method_name, None)):
            raise RuntimeError(
                "coordinate core CoordinateCapture is missing %s()" % method_name
            )
    return capture


@dataclass(frozen=True, slots=True)
class _Transform:
    axes: tuple[int, int]
    a: float
    b: float
    tx: float
    ty: float
    error: float

    def apply(self, point: _RawPoint) -> tuple[float, float]:
        x = point[self.axes[0]]
        y = point[self.axes[1]]
        return (
            self.a * x - self.b * y + self.tx,
            self.b * x + self.a * y + self.ty,
        )

    def invert_xy(self, point: tuple[float, float]) -> tuple[float, float] | None:
        if self.axes != (0, 1):
            return None
        denominator = self.a * self.a + self.b * self.b
        if denominator <= 1e-12:
            return None
        delta_x = float(point[0]) - self.tx
        delta_y = float(point[1]) - self.ty
        return (
            (self.a * delta_x + self.b * delta_y) / denominator,
            (-self.b * delta_x + self.a * delta_y) / denominator,
        )


_COORDINATE_TRANSFORM = _Transform(
    axes=_CALIBRATION_AXES,
    a=_CALIBRATION_A,
    b=_CALIBRATION_B,
    tx=_CALIBRATION_TX,
    ty=_CALIBRATION_TY,
    error=_CALIBRATION_ERROR,
)


def raw_coordinate_to_map(
    x: float,
    y: float,
    z: float | None = None,
) -> _MapPoint | None:
    if 2 in _COORDINATE_TRANSFORM.axes and z is None:
        raise ValueError("raw coordinate z is required by the current calibration")
    point = (float(x), float(y), 0.0 if z is None else float(z))
    map_x, map_y = _COORDINATE_TRANSFORM.apply(point)
    if not math.isfinite(map_x) or not math.isfinite(map_y):
        return None
    return int(round(map_x)), int(round(map_y))


def _map_from_raw(raw: _RawPoint) -> _MapPoint | None:
    return raw_coordinate_to_map(raw[0], raw[1], raw[2])


def _raw_xy_from_map(point: tuple[float, float]) -> tuple[float, float] | None:
    raw = _COORDINATE_TRANSFORM.invert_xy(point)
    if raw is None or not all(math.isfinite(value) for value in raw):
        return None
    return raw


class CoordinatePositionProvider:
    def __init__(
        self,
        backend: str,
        debug: bool = False,
    ) -> None:
        normalized = backend.strip().lower()
        if normalized not in {"map", "auto", "coordinate"}:
            raise ValueError("position_backend must be map, auto, or coordinate")
        self._capture: _CoordinateCapture | None = None
        self._coordinate_active = False
        self._debug = bool(debug)
        self._last_map_point: _MapPoint | None = None
        self._last_raw_coordinate: _RawPoint | None = None
        self._last_camera_pitch: float | None = None
        self._last_camera_heading: float | None = None

        if normalized == "map":
            return

        capture: _CoordinateCapture | None = None
        try:
            capture = _create_capture()
            capture.start()
        except Exception as exc:
            if capture is not None:
                capture.close()
            if normalized == "coordinate":
                raise
            logger.warning("Navi coordinate capture unavailable, using map: %s", exc)
            return

        self._capture = capture
        logger.info(
            "Navi coordinate capture started: axes=%s scale=%.9f error=%.2f",
            _COORDINATE_TRANSFORM.axes,
            math.hypot(_COORDINATE_TRANSFORM.a, _COORDINATE_TRANSFORM.b),
            _COORDINATE_TRANSFORM.error,
        )

    def locate(self, locator: Any, frame: Any) -> MapLocationResult:
        capture = self._capture
        if capture is None:
            if locator is None:
                raise RuntimeError("visual map locator is unavailable")
            result = locator.locate(frame)
            if result.point is not None:
                result.raw_coordinate = _raw_xy_from_map(result.point)
            return result

        pose = capture.read()
        if pose is None:
            if self._debug:
                logger.debug("Navi coordinate unavailable; source=coordinate")
            return MapLocationResult(
                found=False,
                point=self._last_map_point,
                raw_point=self._last_map_point,
                score=1.0 if self._last_map_point is not None else 0.0,
                mode="coordinate_stale",
                polygon=None,
                raw_coordinate=self._last_raw_coordinate,
                camera_pitch=self._last_camera_pitch,
                camera_heading=self._last_camera_heading,
            )

        raw = pose[0], pose[1], pose[2]
        camera_pitch = float(pose[3])
        camera_heading = float(pose[4])
        point = _map_from_raw(raw)
        if (
            point is None
            or not math.isfinite(camera_pitch)
            or not math.isfinite(camera_heading)
        ):
            self._coordinate_active = False
            if self._debug:
                logger.debug(
                    "Navi coordinate transform failed: "
                    "raw=(%.2f, %.2f, %.2f) source=coordinate",
                    raw[0],
                    raw[1],
                    raw[2],
                )
            return MapLocationResult(
                found=False,
                point=self._last_map_point,
                raw_point=self._last_map_point,
                score=1.0 if self._last_map_point is not None else 0.0,
                mode="coordinate_invalid",
                polygon=None,
                raw_coordinate=self._last_raw_coordinate,
                camera_pitch=self._last_camera_pitch,
                camera_heading=self._last_camera_heading,
            )

        if not self._coordinate_active:
            logger.info("Navi position source switched to coordinate-only")
            self._coordinate_active = True
        self._last_map_point = point
        self._last_raw_coordinate = raw
        self._last_camera_pitch = camera_pitch
        self._last_camera_heading = camera_heading % 360.0
        if self._debug:
            logger.debug(
                "Navi coordinate position: raw=(%.2f, %.2f, %.2f) "
                "map=(%d, %d) pitch=%.2f heading=%.2f source=coordinate",
                raw[0],
                raw[1],
                raw[2],
                point[0],
                point[1],
                self._last_camera_pitch,
                self._last_camera_heading,
            )
        return MapLocationResult(
            found=True,
            point=point,
            raw_point=point,
            score=1.0,
            mode="coordinate",
            polygon=None,
            raw_coordinate=raw,
            camera_pitch=self._last_camera_pitch,
            camera_heading=self._last_camera_heading,
        )

    def uses_visual_positioning(self) -> bool:
        return self._capture is None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
