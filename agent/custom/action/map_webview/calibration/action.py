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

from ...Common.logger import get_logger
from ...Navi.map_locator_ncc import MapLocationNccResult, MapLocatorNcc

logger = get_logger(__name__)

DEFAULT_MAP_URL = "https://www.ghzs666.com/yh-map#/"
DEFAULT_CALIBRATION_PATH = "config/map_webview_calibration.json"
MAX_TRANSFORM_CONDITION = 12.0
MAX_RELATIVE_RMSE = 0.18
MAX_INLIER_ERROR = 2.0
PAIR_REPLACE_RADIUS = 20.0


def parse_params(raw: Any) -> dict[str, Any]:
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

        normalized = [normalize_pair(pair) for pair in pairs]
        local_matrix = np.asarray([pair["local"] for pair in normalized], dtype=np.float64)
        targets = np.asarray([pair["online"] for pair in normalized], dtype=np.float64)
        best_inliers = None
        best_rmse = float("inf")
        for first, second in itertools.combinations(range(len(normalized)), 2):
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

        transform = cls._fit_similarity(local_matrix[best_inliers], targets[best_inliers])
        predicted = np.asarray([transform.apply(tuple(point)) for point in local_matrix])
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
            total_count=len(normalized),
            rmse=rmse,
        )


def resolve_project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[5] / path


def calibration_path(params: dict[str, Any]) -> Path | None:
    configured_path = params.get("calibration_path", DEFAULT_CALIBRATION_PATH)
    return resolve_project_path(configured_path) if configured_path else None


def normalize_pair(pair: Any) -> dict[str, list[float]]:
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
        raise ValueError("Calibration pair must contain local and online coordinate pairs")
    return {
        "local": [float(value) for value in local],
        "online": [float(value) for value in online],
    }


def load_pairs(path: Path | None) -> list[dict[str, list[float]]]:
    if path is None or not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("pairs"), list):
        raise ValueError(f"Calibration file does not contain a pairs list: {path}")
    return [normalize_pair(pair) for pair in value["pairs"]]


def save_pairs(path: Path, pairs: list[dict[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pairs": pairs}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def merge_pairs(
    existing: list[dict[str, list[float]]],
    incoming: list[Any],
) -> list[dict[str, list[float]]]:
    pairs = list(existing)
    for value in incoming:
        pair = normalize_pair(value)
        for index, current in enumerate(pairs):
            distance = np.linalg.norm(
                np.asarray(current["local"], dtype=np.float64)
                - np.asarray(pair["local"], dtype=np.float64)
            )
            if distance <= PAIR_REPLACE_RADIUS:
                pairs[index] = pair
                break
        else:
            pairs.append(pair)
    return pairs


def load_transform(params: dict[str, Any]) -> MapCoordinateTransform | None:
    coefficients = params.get("online_transform")
    if coefficients is not None:
        if not isinstance(coefficients, list | tuple) or len(coefficients) != 6:
            raise ValueError("online_transform must contain six numbers")
        return MapCoordinateTransform(
            tuple(float(value) for value in coefficients)
        ).validate()

    path = calibration_path(params)
    pairs = load_pairs(path)
    if len(pairs) < 3:
        logger.warning(f"Map webview calibration is missing or incomplete: path={path}")
        return None
    try:
        transform = MapCoordinateTransform.fit(pairs)
    except ValueError as exc:
        logger.warning(f"Map webview calibration is not usable: {exc}")
        return None
    logger.info(
        f"Map webview calibration loaded: path={path}, "
        f"inliers={transform.inlier_count}/{transform.total_count}, "
        f"rmse={transform.rmse:.3f}, condition={transform.condition_number():.2f}, "
        f"coefficients={transform.coefficients}"
    )
    return transform


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


def _calibration_summary(pairs: list[dict[str, list[float]]]) -> dict[str, Any]:
    summary = {
        "calibrated": False,
        "calibrationCount": len(pairs),
        "calibrationIssue": None,
    }
    if len(pairs) < 3:
        return summary
    try:
        transform = MapCoordinateTransform.fit(pairs)
    except ValueError as exc:
        summary["calibrationIssue"] = str(exc)
        return summary
    summary["calibrated"] = True
    return summary


class _CalibrationStateHandler(BaseHTTPRequestHandler):
    server: "_CalibrationStateServer"

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/state.json":
            self.send_error(404)
            return
        self._send_json(self.server.read_state())

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path == "/calibration.json":
                length = int(self.headers.get("Content-Length", "0"))
                content = self.server.add_pair(json.loads(self.rfile.read(length)))
            elif path == "/calibration/reset.json":
                content = self.server.reset()
            else:
                self.send_error(404)
                return
        except Exception as exc:
            logger.warning(f"Map webview calibration request rejected: {exc}")
            self.send_error(400, str(exc))
            return
        self._send_json(content)

    def _send_json(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _CalibrationStateServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        path: Path,
        pairs: list[dict[str, list[float]]],
    ):
        self._state_lock = threading.Lock()
        self._path = path
        self._pairs = pairs
        self._state = {"point": None, **_calibration_summary(pairs)}
        super().__init__(("127.0.0.1", 0), _CalibrationStateHandler)

    @property
    def state_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/state.json"

    def update_location(self, result: MapLocationNccResult) -> None:
        with self._state_lock:
            self._state["point"] = result.point

    def read_state(self) -> bytes:
        with self._state_lock:
            return json.dumps(self._state, ensure_ascii=False).encode("utf-8")

    def add_pair(self, pair: Any) -> bytes:
        with self._state_lock:
            self._pairs = merge_pairs(self._pairs, [pair])
            save_pairs(self._path, self._pairs)
            self._state.update(_calibration_summary(self._pairs))
            logger.info(
                f"Map webview calibration point saved: path={self._path}, "
                f"pairs={len(self._pairs)}"
            )
            return json.dumps(self._state, ensure_ascii=False).encode("utf-8")

    def reset(self) -> bytes:
        with self._state_lock:
            self._pairs = []
            save_pairs(self._path, self._pairs)
            self._state.update(_calibration_summary(self._pairs))
            logger.info(f"Map webview calibration reset: path={self._path}")
            return json.dumps(self._state, ensure_ascii=False).encode("utf-8")


def _start_viewer(
    server: _CalibrationStateServer,
    params: dict[str, Any],
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).with_name("window.py")),
        "--url",
        str(params.get("map_url") or DEFAULT_MAP_URL),
        "--state-url",
        server.state_url,
        "--title",
        str(params.get("title") or "MaaNTE Map Calibration"),
        "--width",
        str(_positive_int(params.get("width"), 1280)),
        "--height",
        str(_positive_int(params.get("height"), 820)),
    ]
    if params.get("webview_debug"):
        command.append("--debug")
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


def _update_file(path: Path, params: dict[str, Any]) -> None:
    if params.get("reset"):
        save_pairs(path, [])
        logger.info(f"Map webview calibration reset: path={path}")
        return

    incoming = params.get("pairs", [])
    if not isinstance(incoming, list):
        raise ValueError("pairs must be a list")
    if params.get("pair") is not None:
        incoming = [*incoming, params["pair"]]

    pairs = [] if params.get("replace") else load_pairs(path)
    pairs = merge_pairs(pairs, incoming)
    save_pairs(path, pairs)
    summary = _calibration_summary(pairs)
    logger.info(
        f"Map webview calibration saved: path={path}, "
        f"pairs={summary['calibrationCount']}, calibrated={summary['calibrated']}"
    )


@AgentServer.custom_action("map_webview_calibration")
class MapWebViewCalibrationAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = parse_params(argv.custom_action_param)
        path = calibration_path(params)
        if path is None:
            logger.error("Map webview calibration path is disabled")
            return CustomAction.RunResult(success=False)

        try:
            if any(key in params for key in ("pair", "pairs", "replace", "reset")):
                _update_file(path, params)
                return CustomAction.RunResult(success=True)

            if importlib.util.find_spec("webview") is None:
                raise RuntimeError(
                    "Map webview calibration requires pywebview. "
                    "Install requirements.txt and retry."
                )
            locator = MapLocatorNcc(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                debug=False,
            )
            server = _CalibrationStateServer(path, load_pairs(path))
            server_thread = threading.Thread(
                target=server.serve_forever,
                name="map-webview-calibration-state-server",
                daemon=True,
            )
            server_thread.start()
            try:
                process = _start_viewer(server, params)
            except Exception:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2.0)
                raise

            update_interval = _positive_float(params.get("update_interval"), 0.1)
            controller = context.tasker.controller
            exit_code = None
            try:
                while not context.tasker.stopping and process.poll() is None:
                    started = time.perf_counter()
                    frame = controller.post_screencap().wait().get()
                    if frame is not None:
                        server.update_location(locator.locate(frame))
                    sleep_time = update_interval - (time.perf_counter() - started)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                exit_code = process.poll()
            finally:
                _stop_viewer(process)
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2.0)

            if exit_code not in (None, 0):
                raise RuntimeError(
                    f"Map webview calibration process exited unexpectedly: "
                    f"code={exit_code}"
                )
            return CustomAction.RunResult(success=True)
        except Exception as exc:
            logger.error(f"Map webview calibration failed: {exc}")
            return CustomAction.RunResult(success=False)
