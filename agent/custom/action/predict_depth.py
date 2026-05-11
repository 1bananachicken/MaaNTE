import cv2
import json
import os
import time
import numpy as np
import onnxruntime

from pathlib import Path
from .Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = get_logger(__name__)


@AgentServer.custom_action("predict_depth")
class PredictDepth(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            self.model_path = abs_path / "assets/resource/base/model/depth/depth_anything_v2_vits_dynamic.onnx"
        else:
            self.model_path = abs_path / "resource/base/model/depth/depth_anything_v2_vits_dynamic.onnx"
        self._session_cache = {}

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        if not self.model_path.exists():
            logger.error(f"Depth model not found: {self.model_path}")
            return CustomAction.RunResult(success=False)

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                if not isinstance(params, dict):
                    params = {}
            except Exception as exc:
                logger.warning(f"Parse custom_action_param failed, use defaults: {exc}")

        backend = str(params.get("backend") or os.environ.get("MAA_ONNX_BACKEND", "cpu")).strip().lower()
        provider_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }
        if backend == "auto":
            backend = "directml" if "DmlExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
        if backend not in provider_map:
            logger.warning(f"Unknown backend {backend}, fallback to CPU")
            backend = "cpu"

        provider_name = provider_map[backend]
        if provider_name not in onnxruntime.get_available_providers():
            logger.warning(f"Provider {provider_name} is unavailable, fallback to CPU")
            backend = "cpu"
            provider_name = provider_map[backend]

        if backend not in self._session_cache:
            provider_options = [{"device_id": 0}] if provider_name == "DmlExecutionProvider" else None
            self._session_cache[backend] = (
                onnxruntime.InferenceSession(
                    str(self.model_path),
                    sess_options=onnxruntime.SessionOptions(),
                    providers=[provider_name],
                    provider_options=provider_options,
                ),
                provider_name,
            )

        session, provider_name = self._session_cache[backend]
        input_name = session.get_inputs()[0].name
        controller = context.tasker.controller
        masks = [
            [11, 8, 205, 163],
            [882, 10, 382, 63],
            [882, 10, 382, 63],
            [922, 605, 335, 103],
            [462, 653, 369, 46],
            [0, 647, 199, 73],
            [17, 182, 169, 51],
            [1142, 125, 128, 337]
        ]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        logger.info(f"Depth estimation started: {provider_name}, press Q to quit")

        while True:
            if context.tasker.stopping:
                break

            started = time.perf_counter()
            frame = controller.post_screencap().wait().get()
            if frame is None:
                continue
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            h, w = frame.shape[:2]
            masked = frame.copy()
            valid_mask = np.ones((h, w), dtype=np.uint8)
            sx, sy = w / 1280.0, h / 720.0
            for x, y, rw, rh in masks:
                x1, y1 = max(0, int(x * sx)), max(0, int(y * sy))
                x2, y2 = min(w, int((x + rw) * sx)), min(h, int((y + rh) * sy))
                masked[y1:y2, x1:x2] = 0
                valid_mask[y1:y2, x1:x2] = 0

            img = cv2.cvtColor(cv2.resize(masked, (518, 518), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            img = ((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]

            depth = session.run(None, {input_name: img})[0].squeeze()
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
            valid_depth = depth[valid_mask > 0]
            depth_min = float(valid_depth.min()) if valid_depth.size else float(depth.min())
            depth_max = float(valid_depth.max()) if valid_depth.size else float(depth.max())
            depth = np.clip((depth - depth_min) * 255.0 / max(depth_max - depth_min, 1e-6), 0, 255).astype(np.uint8)
            depth[valid_mask == 0] = 0
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            depth_color[valid_mask == 0] = 0

            fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            cv2.putText(depth_color, f"FPS {fps:.1f}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            view_w = 640
            view_h = max(1, int(h * view_w / w))
            view = np.hstack((
                cv2.resize(masked, (view_w, view_h), interpolation=cv2.INTER_AREA),
                cv2.resize(depth_color, (view_w, view_h), interpolation=cv2.INTER_AREA),
            ))
            cv2.imshow("Depth Anything V2", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)
