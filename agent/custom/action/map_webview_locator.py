import importlib.util
import itertools
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from .Common.logger import get_logger
from .map_locator_ncc import MapLocationNccResult, MapLocatorNcc
from .predict_angle import AnglePredictionResult, AnglePredictor

logger = get_logger(__name__)

DEFAULT_MAP_URL = "https://www.ghzs666.com/yh-map#/"
DEFAULT_CALIBRATION_PATH = "config/map_webview_calibration.json"
MAX_TRANSFORM_CONDITION = 12.0
MAX_RELATIVE_RMSE = 0.18
MAX_INLIER_ERROR = 2.0


def _parse_params(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid map webview params, use defaults: {exc}")
    return {}


def _positive_float(value: Any, default: float) -> float:
    try:
        return max(0.01, float(value))
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MapCoordinateTransform:
    """Affine transform from NCC map.jpg pixels to Leaflet latitude/longitude."""

    coefficients: tuple[float, float, float, float, float, float]
    inlier_count: int = 0
    total_count: int = 0
    rmse: float = 0.0

    def apply(self, point: tuple[int, int]) -> tuple[float, float]:
        a, b, c, d, e, f = self.coefficients
        x, y = point
        return a * x + b * y + c, d * x + e * y + f

    def condition_number(self) -> float:
        a, b, _, d, e, _ = self.coefficients
        singular_values = np.linalg.svd(
            np.asarray([[a, b], [d, e]], dtype=np.float64),
            compute_uv=False,
        )
        if singular_values[-1] <= 1e-12:
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    def validate(self) -> "MapCoordinateTransform":
        condition_number = self.condition_number()
        if condition_number > MAX_TRANSFORM_CONDITION:
            raise ValueError(
                f"Calibration transform is ill-conditioned: "
                f"condition={condition_number:.2f}, max={MAX_TRANSFORM_CONDITION:.2f}. "
                "Collect local movement samples in two different directions."
            )
        return self

    @classmethod
    def _fit_similarity(
        cls,
        local_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> "MapCoordinateTransform":
        matrix = []
        values = []
        for (x, y), (latitude, longitude) in zip(local_matrix, targets):
            matrix.extend(([x, -y, 1.0, 0.0], [y, x, 0.0, 1.0]))
            values.extend((latitude, longitude))
        solved, _, rank, _ = np.linalg.lstsq(
            np.asarray(matrix, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            rcond=None,
        )
        if rank < 4:
            raise ValueError("Calibration points must contain distinct locations")

        scale_cos, scale_sin, latitude_offset, longitude_offset = solved
        return cls(
            (
                float(scale_cos),
                float(-scale_sin),
                float(latitude_offset),
                float(scale_sin),
                float(scale_cos),
                float(longitude_offset),
            )
        ).validate()

    @classmethod
    def fit(cls, pairs: list[dict[str, Any]]) -> "MapCoordinateTransform":
        if len(pairs) < 3:
            raise ValueError("At least three calibration pairs are required")

        local_points = []
        online_points = []
        for pair in pairs:
            local = pair.get("local")
            online = pair.get("online")
            if (
                not isinstance(local, list | tuple)
                or not isinstance(online, list | tuple)
                or len(local) != 2
                or len(online) != 2
            ):
                raise ValueError(f"Invalid calibration pair: {pair!r}")
            local_points.append([float(local[0]), float(local[1])])
            online_points.append([float(online[0]), float(online[1])])

        local_matrix = np.asarray(local_points, dtype=np.float64)
        targets = np.asarray(online_points, dtype=np.float64)
        best_inliers = None
        best_rmse = float("inf")
        for first, second in itertools.combinations(range(len(pairs)), 2):
            if np.linalg.norm(local_matrix[first] - local_matrix[second]) <= 1e-6:
                continue
            try:
                candidate = cls._fit_similarity(
                    local_matrix[[first, second]],
                    targets[[first, second]],
                )
            except ValueError:
                continue
            predicted = np.asarray(
                [candidate.apply(tuple(point)) for point in local_matrix]
            )
            residuals = np.linalg.norm(predicted - targets, axis=1)
            inliers = np.flatnonzero(residuals <= MAX_INLIER_ERROR)
            if len(inliers) < 3:
                continue
            rmse = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
            if (
                best_inliers is None
                or len(inliers) > len(best_inliers)
                or (len(inliers) == len(best_inliers) and rmse < best_rmse)
            ):
                best_inliers = inliers
                best_rmse = rmse

        if best_inliers is None:
            raise ValueError(
                "Calibration points do not contain three consistent samples. "
                "Reset and collect nearby movement samples in two directions."
            )

        transform = cls._fit_similarity(
            local_matrix[best_inliers],
            targets[best_inliers],
        )
        predicted = np.asarray(
            [transform.apply(tuple(point)) for point in local_matrix]
        )
        residuals = np.linalg.norm(predicted - targets, axis=1)
        inliers = np.flatnonzero(residuals <= MAX_INLIER_ERROR)
        if len(inliers) < 3:
            raise ValueError("Calibration refinement left fewer than three inliers")

        rmse = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
        centered_targets = targets[inliers] - np.mean(targets[inliers], axis=0)
        spread = float(np.sqrt(np.mean(np.sum(np.square(centered_targets), axis=1))))
        relative_rmse = rmse / spread if spread > 1e-12 else float("inf")
        if relative_rmse > MAX_RELATIVE_RMSE:
            raise ValueError(
                f"Calibration points disagree with one map transform: "
                f"relative_rmse={relative_rmse:.3f}, max={MAX_RELATIVE_RMSE:.3f}. "
                "Reset and collect nearby movement samples in two directions."
            )
        return cls(
            transform.coefficients,
            inlier_count=len(inliers),
            total_count=len(pairs),
            rmse=rmse,
        )


def _resolve_project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _calibration_path(params: dict[str, Any]) -> Path | None:
    configured_path = params.get("calibration_path", DEFAULT_CALIBRATION_PATH)
    return _resolve_project_path(configured_path) if configured_path else None


def _load_calibration_pairs(calibration_path: Path | None) -> list[dict[str, Any]]:
    if calibration_path is None or not calibration_path.exists():
        return []
    value = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid calibration JSON object: {calibration_path}")
    pairs = value.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(
            f"Calibration file does not contain a pairs list: {calibration_path}"
        )
    return pairs


def _load_transform(
    params: dict[str, Any],
    calibration_path: Path | None,
    pairs: list[dict[str, Any]],
) -> tuple[MapCoordinateTransform | None, bool, str | None]:
    coefficients = params.get("online_transform")
    if coefficients is not None:
        if not isinstance(coefficients, list | tuple) or len(coefficients) != 6:
            raise ValueError("online_transform must contain six numbers")
        return (
            MapCoordinateTransform(
                tuple(float(value) for value in coefficients)
            ).validate(),
            True,
            None,
        )

    if len(pairs) < 3:
        return None, False, None
    try:
        transform = MapCoordinateTransform.fit(pairs)
    except ValueError as exc:
        logger.warning(f"Map webview calibration is not usable: {exc}")
        return None, False, str(exc)
    logger.info(
        f"Map webview calibration loaded: path={calibration_path}, "
        f"inliers={transform.inlier_count}/{transform.total_count}, "
        f"rmse={transform.rmse:.3f}, condition={transform.condition_number():.2f}, "
        f"coefficients={transform.coefficients}"
    )
    return transform, False, None


def _normalize_calibration_pair(pair: Any) -> dict[str, list[float]]:
    if not isinstance(pair, dict):
        raise ValueError("Calibration pair must be an object")
    local = pair.get("local")
    online = pair.get("online")
    if (
        not isinstance(local, list | tuple)
        or not isinstance(online, list | tuple)
        or len(local) != 2
        or len(online) != 2
    ):
        raise ValueError(
            "Calibration pair must contain local and online coordinate pairs"
        )
    return {
        "local": [float(value) for value in local],
        "online": [float(value) for value in online],
    }


def _location_payload(
    locator: MapLocatorNcc,
    result: MapLocationNccResult,
    angle_result: AnglePredictionResult,
    transform: MapCoordinateTransform | None,
    calibration_count: int,
    calibration_issue: str | None,
) -> dict[str, Any]:
    online_point = transform.apply(result.point) if transform and result.point else None
    return {
        "point": result.point,
        "rawPoint": result.raw_point,
        "onlinePoint": online_point,
        "score": result.score,
        "mode": result.mode,
        "angle": angle_result.angle,
        "angleConfidence": angle_result.confidence,
        "angleFound": angle_result.found,
        "mapSize": [locator.origin_w, locator.origin_h],
        "calibrated": transform is not None,
        "calibrationCount": calibration_count,
        "calibrationIssue": calibration_issue,
        "calibrationInliers": transform.inlier_count if transform else 0,
        "calibrationRmse": transform.rmse if transform else None,
    }


class _MapStateHandler(BaseHTTPRequestHandler):
    server: "_MapStateServer"

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/state.json":
            self.send_error(404)
            return
        content = self.server.read_state()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/calibration/reset.json":
            content = self.server.reset_calibration()
        elif path == "/calibration.json":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                pair = json.loads(self.rfile.read(length))
                content = self.server.add_calibration_pair(pair)
            except Exception as exc:
                logger.warning(f"Map webview calibration rejected: {exc}")
                self.send_error(400, str(exc))
                return
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _MapStateServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        calibration_path: Path | None,
        pairs: list[dict[str, Any]],
        transform: MapCoordinateTransform | None,
        transform_locked: bool,
        calibration_issue: str | None,
    ):
        self._state_lock = threading.Lock()
        self._calibration_path = calibration_path
        self._pairs = pairs
        self._transform = transform
        self._transform_locked = transform_locked
        self._calibration_issue = calibration_issue
        self._state = json.dumps(
            {
                "point": None,
                "onlinePoint": None,
                "score": 0.0,
                "mode": "init",
                "angle": None,
                "angleConfidence": 0.0,
                "angleFound": False,
                "calibrated": transform is not None,
                "calibrationCount": len(pairs),
                "calibrationIssue": calibration_issue,
                "calibrationInliers": transform.inlier_count if transform else 0,
                "calibrationRmse": transform.rmse if transform else None,
            }
        ).encode("utf-8")
        super().__init__(("127.0.0.1", 0), _MapStateHandler)

    @property
    def state_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/state.json"

    def update_location(
        self,
        locator: MapLocatorNcc,
        result: MapLocationNccResult,
        angle_result: AnglePredictionResult,
    ) -> None:
        with self._state_lock:
            self._state = json.dumps(
                _location_payload(
                    locator,
                    result,
                    angle_result,
                    self._transform,
                    len(self._pairs),
                    self._calibration_issue,
                ),
                ensure_ascii=False,
            ).encode("utf-8")

    def read_state(self) -> bytes:
        with self._state_lock:
            return self._state

    def add_calibration_pair(self, pair: dict[str, Any]) -> bytes:
        normalized = _normalize_calibration_pair(pair)
        with self._state_lock:
            replaced = False
            for index, existing in enumerate(self._pairs):
                distance = np.linalg.norm(
                    np.asarray(existing["local"], dtype=np.float64)
                    - np.asarray(normalized["local"], dtype=np.float64)
                )
                if distance <= 20.0:
                    self._pairs[index] = normalized
                    replaced = True
                    break
            if not replaced:
                self._pairs.append(normalized)

            if not self._transform_locked and len(self._pairs) >= 3:
                try:
                    self._transform = MapCoordinateTransform.fit(self._pairs)
                    self._calibration_issue = None
                except ValueError as exc:
                    self._transform = None
                    self._calibration_issue = str(exc)
                    logger.warning(f"Map webview calibration is not usable yet: {exc}")

            if self._calibration_path is not None:
                self._calibration_path.parent.mkdir(parents=True, exist_ok=True)
                self._calibration_path.write_text(
                    json.dumps({"pairs": self._pairs}, indent=2, ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )

            response = {
                "calibrated": self._transform is not None,
                "calibrationCount": len(self._pairs),
                "calibrationIssue": self._calibration_issue,
                "calibrationInliers": (
                    self._transform.inlier_count if self._transform else 0
                ),
                "calibrationRmse": self._transform.rmse if self._transform else None,
            }
            logger.info(
                f"Map webview calibration saved: path={self._calibration_path}, "
                f"pairs={len(self._pairs)}, calibrated={response['calibrated']}"
            )
            return json.dumps(response).encode("utf-8")

    def reset_calibration(self) -> bytes:
        with self._state_lock:
            self._pairs = []
            if not self._transform_locked:
                self._transform = None
            self._calibration_issue = None
            state = json.loads(self._state)
            point = state.get("point")
            state.update(
                {
                    "onlinePoint": (
                        self._transform.apply(tuple(point))
                        if self._transform is not None and point
                        else None
                    ),
                    "calibrated": self._transform is not None,
                    "calibrationCount": 0,
                    "calibrationIssue": None,
                    "calibrationInliers": 0,
                    "calibrationRmse": None,
                }
            )
            self._state = json.dumps(state, ensure_ascii=False).encode("utf-8")
            if self._calibration_path is not None:
                self._calibration_path.parent.mkdir(parents=True, exist_ok=True)
                self._calibration_path.write_text('{"pairs": []}\n', encoding="utf-8")
            logger.info(f"Map webview calibration reset: path={self._calibration_path}")
            return json.dumps(
                {
                    "calibrated": self._transform is not None,
                    "calibrationCount": 0,
                    "calibrationIssue": None,
                    "calibrationInliers": 0,
                    "calibrationRmse": None,
                }
            ).encode("utf-8")


def _start_viewer(
    server: _MapStateServer, map_url: str, params: dict[str, Any]
) -> subprocess.Popen:
    viewer_script = Path(__file__).with_name("map_webview_window.py")
    command = [
        sys.executable,
        str(viewer_script),
        "--url",
        map_url,
        "--state-url",
        server.state_url,
        "--title",
        str(params.get("title") or "MaaNTE Online Map"),
        "--width",
        str(_positive_int(params.get("width"), 1280)),
        "--height",
        str(_positive_int(params.get("height"), 820)),
    ]
    if params.get("webview_debug"):
        command.append("--debug")
    if params.get("pointer_image"):
        command.extend(["--pointer-path", str(params["pointer_image"])])

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    return subprocess.Popen(command, creationflags=creationflags)


def _stop_viewer(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


@AgentServer.custom_action("map_webview_locator")
class MapWebViewLocatorAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = _parse_params(argv.custom_action_param)
        if importlib.util.find_spec("webview") is None:
            logger.error(
                "Map webview requires pywebview. Install requirements.txt and retry."
            )
            return CustomAction.RunResult(success=False)

        try:
            calibration_path = _calibration_path(params)
            pairs = _load_calibration_pairs(calibration_path)
            locator = MapLocatorNcc(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                debug=False,
            )
            predictor = AnglePredictor(
                backend=params.get("angle_backend") or params.get("backend"),
                pointer_roi=params.get("pointer_roi") or None,
                threshold=float(params.get("angle_threshold", 0.0)),
                debug=False,
            )
            angle_provider = predictor.provider_name()
            transform, transform_locked, calibration_issue = _load_transform(
                params,
                calibration_path,
                pairs,
            )
        except Exception as exc:
            logger.error(f"Map webview locator init failed: {exc}")
            return CustomAction.RunResult(success=False)

        map_url = str(params.get("map_url") or DEFAULT_MAP_URL)
        update_interval = _positive_float(params.get("update_interval"), 0.1)
        try:
            server = _MapStateServer(
                calibration_path,
                pairs,
                transform,
                transform_locked,
                calibration_issue,
            )
        except Exception as exc:
            logger.error(f"Map webview state server init failed: {exc}")
            return CustomAction.RunResult(success=False)
        server_thread = threading.Thread(
            target=server.serve_forever,
            name="map-webview-state-server",
            daemon=True,
        )
        server_thread.start()

        try:
            process = _start_viewer(server, map_url, params)
        except Exception as exc:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2.0)
            logger.error(f"Map webview process failed to start: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(
            f"Map webview locator started: url={map_url}, map={locator.big_map_path}, "
            f"calibrated={transform is not None}, angle_provider={angle_provider}"
        )
        controller = context.tasker.controller
        exit_code = None
        try:
            while not context.tasker.stopping and process.poll() is None:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is not None:
                    result = locator.locate(frame)
                    angle_result = predictor.predict(frame)
                    server.update_location(locator, result, angle_result)
                    logger.debug(
                        f"Map webview location: point={result.point}, raw={result.raw_point}, "
                        f"score={result.score:.3f}, mode={result.mode}, "
                        f"angle={angle_result.angle}, "
                        f"angle_confidence={angle_result.confidence:.3f}"
                    )

                sleep_time = update_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            exit_code = process.poll()
        except Exception as exc:
            logger.error(f"Map webview locator failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            _stop_viewer(process)
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2.0)

        if exit_code not in (None, 0):
            logger.error(f"Map webview process exited unexpectedly: code={exit_code}")
            return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=True)
