import cv2
import math
import time
import numpy as np
import onnxruntime  

from pathlib import Path

from .Common.utils import get_image

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("predict_angle_new")
class PredictAngleNew(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            model_path = abs_path / "assets/resource/base/model/navi/pointer_model.onnx"
        else:
            model_path = abs_path / "resource/base/model/navi/pointer_model.onnx"
        # self.pointer_roi = [84, 71, 40, 40]
        self.pointer_roi = [73, 60, 64, 64]
        self.session = onnxruntime.InferenceSession(model_path)  
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        controller = context.tasker.controller
        pointer_roi = self.pointer_roi
        
        print("启动新的实时角度预测... (在弹出的窗口上按 'Q' 键退出)")

        while True:
            if context.tasker.stopping:
                break

            img = get_image(controller)
            x, y, w, h = pointer_roi
            img = img[y:y+h, x:x+w]
            cv2.imshow("Arrow Tracker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            img = img[..., ::-1]  # BGR 转 RGB
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = np.expand_dims(img, axis=0)  # 添加批次维度
            input_name = self.session.get_inputs()[0].name
            result = self.session.run(None, {input_name: img})  
            print(f"模型输出: {result}")
            pred_deg = math.degrees(math.atan2(result[0][0][1], result[0][0][0])) % 360
            print(f"预测角度: {pred_deg:.2f}°")

            time.sleep(1)
            
        return CustomAction.RunResult(success=True)
