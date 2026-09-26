"""
自动排球 - space 高亮检测（HSV + 连通块面积筛选）
用于 NewVolleyballSpikeJump 节点，替代纯 ColorMatch，
防止场地蓝色边框进入 ROI 时误判。
"""
import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context


@AgentServer.custom_recognition("volleyball_spike_jump_detect")
class VolleyballSpikeJumpDetect(CustomRecognition):
    # HSV 范围：space 高亮是半透明浅蓝色/青色
    LOWER = np.array([80, 40, 120])
    UPPER = np.array([115, 220, 255])
    # 最大连通块面积阈值：按钮约5200，边框约520
    MIN_AREA = 3000

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> CustomRecognition.AnalyzeResult | None:
        image = argv.image
        if image is None:
            return None

        # 解析参数（兼容对象和字符串）
        params = {}
        raw = argv.custom_recognition_param
        if raw:
            if isinstance(raw, dict):
                params = raw
            else:
                try:
                    import json
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        params = parsed
                except Exception:
                    pass

        roi = params.get("roi")
        if roi and len(roi) == 4:
            x, y, w, h = [int(v) for v in roi]
            image = image[y:y + h, x:x + w]

        # HSV 筛选
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER, self.UPPER)

        # 连通块面积筛选
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        max_area = 0
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > max_area:
                max_area = area

        if max_area >= self.MIN_AREA:
            return CustomRecognition.AnalyzeResult(
                box=[0, 0, image.shape[1], image.shape[0]],
                detail={"area": int(max_area)},
            )
        return None
