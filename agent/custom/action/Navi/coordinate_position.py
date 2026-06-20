from __future__ import annotations

import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..Common.logger import get_logger
from .map_locator import MapLocationResult

logger = get_logger(__name__)

_RawPoint = tuple[float, float, float]
_MapPoint = tuple[int, int]
_PLANE_AXES = ((0, 1), (0, 2), (1, 2))
_CORE_MODULE = "nte_coordinate_api"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_THIRDPARTY_DIR = _PROJECT_ROOT / "thirdparty"
_CALIBRATION_PATH = _PROJECT_ROOT / "config" / "navi_coordinate_calibration.json"
_MAP_WORLD_ORIGIN = (5632.0, 5632.0)
_MAP_PIXELS_PER_WORLD_UNIT = 22.0


class _CoordinateCapture(Protocol):
    def start(self) -> None: ...

    def read(self, max_age: float = 1.0) -> _RawPoint | None: ...

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

    capture = capture_type()
    for method_name in ("start", "read", "close"):
        if not callable(getattr(capture, method_name, None)):
            raise RuntimeError(
                "coordinate core CoordinateCapture is missing %s()" % method_name
            )
    return capture


@dataclass(slots=True)
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


class _CoordinateCalibration:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._samples: list[tuple[_RawPoint, _MapPoint]] = []
        self._transform: _Transform | None = None
        self._persisted = False
        self._load()

    def observe(self, raw: _RawPoint, point: _MapPoint) -> None:
        if self._persisted:
            return
        if self._transform is not None:
            predicted = self._transform.apply(raw)
            residual = math.hypot(predicted[0] - point[0], predicted[1] - point[1])
            if residual > 500.0:
                logger.info(
                    "Navi coordinate calibration reset: residual=%.1f",
                    residual,
                )
                self._samples.clear()
                self._transform = None

        if self._samples:
            previous_raw, previous_point = self._samples[-1]
            raw_step = math.dist(raw, previous_raw)
            map_step = math.dist(point, previous_point)
            if raw_step < 20.0 and map_step < 2.0:
                return

        self._samples.append((raw, point))
        if len(self._samples) > 32:
            del self._samples[0]
        fitted = self._fit()
        if fitted is not None:
            first_calibration = self._transform is None
            self._transform = fitted
            if first_calibration:
                logger.info(
                    "Navi coordinate calibrated: axes=%s scale=%.6f error=%.2f",
                    fitted.axes,
                    math.hypot(fitted.a, fitted.b),
                    fitted.error,
                )
                self._save(fitted)
                self._persisted = True

    def locate(self, raw: _RawPoint) -> _MapPoint | None:
        if self._transform is None:
            return None
        x, y = self._transform.apply(raw)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return int(round(x)), int(round(y))

    def ready(self) -> bool:
        return self._transform is not None

    def persisted(self) -> bool:
        return self._persisted

    def _load(self) -> None:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(value.get("points"), list):
                transform = self._fit_manual_points(value["points"])
            else:
                axes = value["axes"]
                transform = _Transform(
                    axes=(int(axes[0]), int(axes[1])),
                    a=float(value["a"]),
                    b=float(value["b"]),
                    tx=float(value["tx"]),
                    ty=float(value["ty"]),
                    error=float(value.get("error", 0.0)),
                )
            if transform.axes not in _PLANE_AXES:
                raise ValueError("invalid coordinate plane axes")
            coefficients = (
                transform.a,
                transform.b,
                transform.tx,
                transform.ty,
                transform.error,
            )
            if not all(math.isfinite(item) for item in coefficients):
                raise ValueError("non-finite calibration coefficient")
        except FileNotFoundError:
            return
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Navi coordinate calibration file invalid: path=%s error=%s",
                self._path,
                exc,
            )
            return

        self._transform = transform
        self._persisted = True
        logger.info(
            "Navi coordinate calibration loaded: path=%s axes=%s "
            "scale=%.6f error=%.2f",
            self._path,
            transform.axes,
            math.hypot(transform.a, transform.b),
            transform.error,
        )

    def _fit_manual_points(self, values: list[Any]) -> _Transform:
        samples: list[tuple[_RawPoint, _MapPoint]] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError("manual point %d must be an object" % index)
            raw = value.get("raw")
            point = value.get("map")
            world = value.get("world")
            if point is None and isinstance(world, (list, tuple)) and len(world) == 2:
                latitude = float(world[0])
                longitude = float(world[1])
                point = (
                    _MAP_WORLD_ORIGIN[0]
                    + longitude * _MAP_PIXELS_PER_WORLD_UNIT,
                    _MAP_WORLD_ORIGIN[1]
                    - latitude * _MAP_PIXELS_PER_WORLD_UNIT,
                )
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 3
                or not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                raise ValueError(
                    "manual point %d needs raw=[x,y,z] and "
                    "map=[x,y] or world=[lat,lng]" % index
                )
            samples.append(
                (
                    (float(raw[0]), float(raw[1]), float(raw[2])),
                    (int(round(float(point[0]))), int(round(float(point[1])))),
                )
            )
        if len(samples) < 3:
            raise ValueError("manual calibration needs at least 3 points")

        previous_samples = self._samples
        try:
            self._samples = samples
            transform = self._fit(minimum_samples=3)
        finally:
            self._samples = previous_samples
        if transform is None:
            raise ValueError("manual calibration points cannot produce a valid transform")
        return transform

    def _save(self, transform: _Transform) -> None:
        value = {
            "version": 1,
            "axes": list(transform.axes),
            "a": transform.a,
            "b": transform.b,
            "tx": transform.tx,
            "ty": transform.ty,
            "error": transform.error,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(value, ensure_ascii=False, indent=4) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, self._path)
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                "failed to save coordinate calibration: %s" % self._path
            ) from exc
        logger.info("Navi coordinate calibration saved: path=%s", self._path)

    def _fit(self, minimum_samples: int = 4) -> _Transform | None:
        if len(self._samples) < minimum_samples:
            return None

        map_points = np.asarray([item[1] for item in self._samples], dtype=np.float64)
        if (
            float(np.ptp(map_points[:, 0])) < 5.0
            and float(np.ptp(map_points[:, 1])) < 5.0
        ):
            return None

        candidates: list[_Transform] = []
        for axes in _PLANE_AXES:
            raw_points = np.asarray(
                [[item[0][axes[0]], item[0][axes[1]]] for item in self._samples],
                dtype=np.float64,
            )
            if (
                max(float(np.ptp(raw_points[:, 0])), float(np.ptp(raw_points[:, 1])))
                < 50
            ):
                continue

            matrix: list[list[float]] = []
            target: list[float] = []
            for (raw_x, raw_y), (map_x, map_y) in zip(raw_points, map_points):
                matrix.append([raw_x, -raw_y, 1.0, 0.0])
                target.append(map_x)
                matrix.append([raw_y, raw_x, 0.0, 1.0])
                target.append(map_y)
            coefficients, _, rank, _ = np.linalg.lstsq(
                np.asarray(matrix, dtype=np.float64),
                np.asarray(target, dtype=np.float64),
                rcond=None,
            )
            if rank < 4:
                continue

            transform = _Transform(
                axes=axes,
                a=float(coefficients[0]),
                b=float(coefficients[1]),
                tx=float(coefficients[2]),
                ty=float(coefficients[3]),
                error=0.0,
            )
            scale = math.hypot(transform.a, transform.b)
            if not 0.0001 <= scale <= 100.0:
                continue
            errors = [
                math.dist(transform.apply(raw), point) for raw, point in self._samples
            ]
            transform.error = math.sqrt(
                sum(error * error for error in errors) / len(errors)
            )
            candidates.append(transform)

        if not candidates:
            return None
        selected = min(candidates, key=lambda item: item.error)
        return selected if selected.error <= 80.0 else None


class CoordinatePositionProvider:
    def __init__(
        self,
        backend: str,
        debug: bool = False,
    ) -> None:
        normalized = backend.strip().lower()
        if normalized not in {"map", "auto", "coordinate"}:
            raise ValueError("position_backend must be map, auto, or coordinate")
        self._backend = normalized
        self._capture: _CoordinateCapture | None = None
        self._calibration = _CoordinateCalibration(_CALIBRATION_PATH)
        self._coordinate_active = False
        self._debug = bool(debug)

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

        if not self._calibration.ready():
            capture.close()
            message = (
                "coordinate positioning requires a valid calibration file: %s"
                % _CALIBRATION_PATH
            )
            if normalized == "coordinate":
                raise RuntimeError(message)
            logger.warning("%s; using map", message)
            return

        self._capture = capture
        logger.info("Navi coordinate capture started")

    def locate(self, locator: Any, frame: Any) -> MapLocationResult:
        capture = self._capture
        if capture is None:
            if locator is None:
                raise RuntimeError("visual map locator is unavailable")
            return locator.locate(frame)

        raw = capture.read()
        if raw is None:
            if self._debug:
                logger.debug("Navi coordinate unavailable; source=coordinate")
            return MapLocationResult(
                found=False,
                point=None,
                raw_point=None,
                score=0.0,
                mode="coordinate_unavailable",
                polygon=None,
            )

        point = self._calibration.locate(raw)
        if point is None:
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
                point=None,
                raw_point=None,
                score=0.0,
                mode="coordinate_invalid",
                polygon=None,
            )
        if not self._coordinate_active:
            logger.info("Navi position source switched to coordinate-only")
            self._coordinate_active = True
        if self._debug:
            logger.debug(
                "Navi coordinate position: raw=(%.2f, %.2f, %.2f) "
                "map=(%d, %d) source=coordinate",
                raw[0],
                raw[1],
                raw[2],
                point[0],
                point[1],
            )
        return MapLocationResult(
            found=True,
            point=point,
            raw_point=point,
            score=1.0,
            mode="coordinate",
            polygon=None,
        )

    def uses_visual_positioning(self) -> bool:
        return self._capture is None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
