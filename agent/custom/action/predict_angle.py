import cv2
import json
import math
import os
import time
import numpy as np
import onnxruntime

from pathlib import Path

from .Common.utils import get_image

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("predict_angle")
class PredictAngle(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            model_path = abs_path / "assets/resource/base/model/navi/pointer_model.onnx"
        else:
            model_path = abs_path / "resource/base/model/navi/pointer_model.onnx"

        self.model_path = model_path
        self.pointer_roi = [73, 60, 64, 64]
        self.threshold = 0.5
        self._session_cache = {}
        self._provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller
        pointer_roi = self.pointer_roi
        backend = self._resolve_backend(argv.custom_action_param)
        session, provider_name = self._get_session(backend)
        input_name = session.get_inputs()[0].name

        print("启动新的 YOLO-Pose 实时角度预测... (在弹出的窗口上按 'Q' 键退出)")
        print(f"当前推理后端: {backend.upper()} ({provider_name})")

        while True:
            if context.tasker.stopping:
                cv2.destroyAllWindows()
                break

            img_original = controller.post_screencap().wait().get()[..., ::-1]
            x, y, w, h = pointer_roi
            img_crop = img_original[y:y+h, x:x+w].copy()

            img_input = img_crop / 255.0    
            img_input = img_input.transpose(2, 0, 1).astype(np.float32) 
            img_input = np.expand_dims(img_input, axis=0)  

            result = session.run(None, {input_name: img_input})
            output = result[0][0]

            confidence = output[:, 4]
            best_idx = np.argmax(confidence)
            best_pred = output[best_idx]
            max_conf = confidence[best_idx]
          
            if max_conf > 0.5:            
                kpts = best_pred[6:].reshape(3, 3)
                tip = kpts[0][:2]    
                left = kpts[1][:2]   
                right = kpts[2][:2]  
             
                tail_center = (left + right) / 2
            
                dx = tip[0] - tail_center[0]
                dy = tip[1] - tail_center[1]
                angle = math.degrees(math.atan2(dx, -dy)) % 360
                
                print(f"预测角度: {angle:05.1f}° | 置信度: {max_conf:.2f}")

                x1, y1, x2, y2 = best_pred[0:4]
                tl = (int(x1), int(y1))
                br = (int(x2), int(y2))
                cv2.rectangle(img_crop, tl, br, (0, 255, 0), 1)

                pt0 = (int(tip[0]), int(tip[1]))
                pt1 = (int(left[0]), int(left[1]))
                pt2 = (int(right[0]), int(right[1]))
                tail = (int(tail_center[0]), int(tail_center[1]))

                cv2.line(img_crop, tail, pt0, (255, 0, 255), 2)
                cv2.circle(img_crop, pt0, 2, (0, 0, 255), -1)
                cv2.circle(img_crop, pt1, 2, (255, 255, 0), -1)
                cv2.circle(img_crop, pt2, 2, (255, 255, 0), -1)

            else:
                print(f"未检测到指针，最高置信度仅为: {max_conf:.2f}")

            display_img = cv2.cvtColor(img_crop.copy(), cv2.COLOR_RGB2BGR)
            target_size = (400, 400)
            display_img = cv2.resize(display_img, target_size, interpolation=cv2.INTER_CUBIC)

            if max_conf > 0.5:
                angle_text = f"Angle: {angle:05.1f} deg"
                conf_text = f"Conf:  {max_conf:.2f}"

                cv2.putText(display_img, angle_text, (10, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display_img, conf_text, (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_img, "NO TARGET", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Arrow Tracker", cv2.resize(display_img, (400, 400), interpolation=cv2.INTER_NEAREST))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.5) 

        return CustomAction.RunResult(success=True)

    def _resolve_backend(self, custom_action_param: str) -> str:
        backend = os.environ.get("MAA_ONNX_BACKEND", "cpu")

        if custom_action_param:
            try:
                params = json.loads(custom_action_param)
                if isinstance(params, dict) and params.get("backend"):
                    backend = str(params["backend"])
            except Exception as exc:
                print(f"解析 custom_action_param 失败，将使用默认后端: {exc}")

        backend = backend.strip().lower()
        if backend == "auto":
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return "cuda"
            if "DmlExecutionProvider" in available:
                return "directml"
            return "cpu"

        if backend not in self._provider_name_map:
            print(f"未知推理后端 {backend}，将回退到 CPU")
            return "cpu"
        return backend

    def _get_session(self, backend: str):
        if backend in self._session_cache:
            return self._session_cache[backend]

        provider_name = self._provider_name_map[backend]
        available = onnxruntime.get_available_providers()

        if provider_name not in available:
            print(
                f"请求的后端 {backend.upper()} 不可用，当前可用 Providers: {available}，已回退到 CPU"
            )
            backend = "cpu"
            provider_name = self._provider_name_map[backend]

        session_options = onnxruntime.SessionOptions()
        providers = [provider_name]
        provider_options = None

        if provider_name == "CUDAExecutionProvider":
            provider_options = [{"device_id": 0}]
        elif provider_name == "DmlExecutionProvider":
            provider_options = [{"device_id": 0}]

        if provider_options is None:
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers,
            )
        else:
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )

        self._session_cache[backend] = (session, provider_name)
        return self._session_cache[backend]
