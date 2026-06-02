import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from ..Common.logger import get_logger
from .resources import resource_base_path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

logger = get_logger(__name__)


@dataclass
class AnglePredictionResult:
    found: bool
    angle: float | None
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    tip: tuple[int, int] | None = None
    left: tuple[int, int] | None = None
    right: tuple[int, int] | None = None


class AnglePredictor:
    def __init__(
        self,
        model_path: Path | None = None,
        backend: str | None = None,
        pointer_roi: list[int] | None = None,
        threshold: float = 0.0,
        debug: bool = False,
    ):
        if model_path is None:
            model_path = resource_base_path() / "model/navi/pointer_model.onnx"

        self.model_path = Path(model_path)
        self.backend = self._resolve_backend(backend)
        self.pointer_roi = pointer_roi or [73, 60, 64, 64]
        self.threshold = threshold
        self.debug = debug
        self._session_cache = {}
        self._provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }

    def predict(self, frame: np.ndarray) -> AnglePredictionResult:
        session, _provider_name = self._get_session()
        input_name = session.get_inputs()[0].name

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        x, y, w, h = self.pointer_roi
        img_crop = frame[y : y + h, x : x + w].copy()
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

        img_input = (img_rgb / 255.0).transpose(2, 0, 1).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)

        output = session.run(None, {input_name: img_input})[0][0]
        confidence = output[:, 4]
        best_idx = int(np.argmax(confidence))
        best_pred = output[best_idx]
        max_conf = float(confidence[best_idx])

        result = AnglePredictionResult(found=False, angle=None, confidence=max_conf)
        if max_conf > self.threshold:
            kpts = best_pred[6:].reshape(3, 3)
            tip = kpts[0][:2]
            left = kpts[1][:2]
            right = kpts[2][:2]
            tail_center = (left + right) / 2

            dx = tip[0] - tail_center[0]
            dy = tip[1] - tail_center[1]
            angle = math.degrees(math.atan2(dx, -dy)) % 360

            x1, y1, x2, y2 = best_pred[0:4]
            result = AnglePredictionResult(
                found=True,
                angle=float(angle),
                confidence=max_conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                tip=(int(tip[0]), int(tip[1])),
                left=(int(left[0]), int(left[1])),
                right=(int(right[0]), int(right[1])),
            )

        if self.debug:
            self.show_debug(img_crop, result)

        return result

    def show_debug(self, img_crop: np.ndarray, result: AnglePredictionResult) -> None:
        display_img = img_crop.copy()
        if result.found and result.bbox and result.tip and result.left and result.right:
            cv2.rectangle(
                display_img,
                (result.bbox[0], result.bbox[1]),
                (result.bbox[2], result.bbox[3]),
                (0, 255, 0),
                1,
            )
            tail = (
                int((result.left[0] + result.right[0]) / 2),
                int((result.left[1] + result.right[1]) / 2),
            )
            cv2.line(display_img, tail, result.tip, (255, 0, 255), 2)
            cv2.circle(display_img, result.tip, 2, (0, 0, 255), -1)
            cv2.circle(display_img, result.left, 2, (255, 255, 0), -1)
            cv2.circle(display_img, result.right, 2, (255, 255, 0), -1)

        display_img = cv2.resize(display_img, (400, 400), interpolation=cv2.INTER_CUBIC)
        if result.found and result.angle is not None:
            cv2.putText(
                display_img,
                f"Angle: {result.angle:05.1f} deg",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_img,
                f"Conf:  {result.confidence:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display_img,
                "NO TARGET",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Angle Predictor", display_img)

    def close_debug(self) -> None:
        if self.debug:
            cv2.destroyWindow("Angle Predictor")

    def provider_name(self) -> str:
        _session, provider_name = self._get_session()
        return provider_name

    def _resolve_backend(self, backend: str | None) -> str:
        backend = (
            str(backend or os.environ.get("MAA_ONNX_BACKEND", "cpu")).strip().lower()
        )
        if backend == "auto":
            available = onnxruntime.get_available_providers()
            if "DmlExecutionProvider" in available:
                return "directml"
            return "cpu"

        provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }
        if backend not in provider_name_map:
            logger.warning(f"Unknown inference backend {backend}, fallback to CPU")
            return "cpu"
        return backend

    def _get_session(self):
        backend = self.backend
        if backend in self._session_cache:
            return self._session_cache[backend]

        if not self.model_path.exists():
            raise FileNotFoundError(f"Angle model not found: {self.model_path}")

        provider_name = self._provider_name_map[backend]
        available = onnxruntime.get_available_providers()
        if provider_name not in available:
            logger.warning(
                f"Requested provider {provider_name} is unavailable, available providers: {available}; fallback to CPU"
            )
            backend = "cpu"
            self.backend = backend
            provider_name = self._provider_name_map[backend]

        provider_options = (
            [{"device_id": 0}] if provider_name == "DmlExecutionProvider" else None
        )
        session = onnxruntime.InferenceSession(
            str(self.model_path),
            sess_options=onnxruntime.SessionOptions(),
            providers=[provider_name],
            provider_options=provider_options,
        )
        self._session_cache[backend] = (session, provider_name)
        return self._session_cache[backend]


@AgentServer.custom_action("predict_angle")
class AnglePredictorTestAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = _load_params(argv.custom_action_param)
        debug = bool(params.get("debug", False))
        frame_interval = float(params.get("frame_interval", 0.016))
        threshold = float(params.get("threshold", 0.0))
        pointer_roi = params.get("pointer_roi") or None
        backend = params.get("backend")

        try:
            predictor = AnglePredictor(
                backend=backend,
                pointer_roi=pointer_roi,
                threshold=threshold,
                debug=debug,
            )
            provider_name = predictor.provider_name()
        except Exception as exc:
            logger.error(f"Angle predictor init failed: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(
            f"Angle predictor started: backend={predictor.backend}, provider={provider_name}, debug={debug}"
        )
        controller = context.tasker.controller
        last_result = AnglePredictionResult(found=False, angle=None, confidence=0.0)

        try:
            while not context.tasker.stopping:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    continue

                last_result = predictor.predict(frame)

                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                sleep_time = frame_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Angle predictor failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            if debug:
                cv2.destroyAllWindows()

        return CustomAction.RunResult(success=last_result.found)


def _load_params(custom_action_param) -> dict:
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    try:
        params = json.loads(custom_action_param)
        return params if isinstance(params, dict) else {}
    except Exception as exc:
        logger.warning(f"Parse custom_action_param failed, use defaults: {exc}")
        return {}
