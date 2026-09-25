import json
import math

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils.logger import logger

# 主控腰部 ROI（检测蓝色底线）
SERVE_ROI = [380, 335, 520, 50]
# HSV 蓝色范围（V下限100，覆盖底线蓝色）
HSV_LOWER = np.array([90, 30, 100])
HSV_UPPER = np.array([130, 255, 255])
# 底线连通块最小宽度（衣物花纹最大约34px，底线115~221px）
MIN_LINE_WIDTH = 80


@AgentServer.custom_recognition("volleyball_serve_detect")
class VolleyballServeDetect(CustomRecognition):
    """检测主控是否站在蓝色底线上（即主控发球状态）。

    在主控腰部ROI内做HSV蓝色筛选+连通块分析，若存在宽度>MIN_LINE_WIDTH的
    水平连通块，则判定为蓝色底线穿过主控身体 → 主控发球。

    custom_recognition_param (JSON, 可选):
      roi: [x,y,w,h]         覆盖默认ROI
      min_line_width: int    覆盖默认最小宽度
    """

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> CustomRecognition.AnalyzeResult | None:
        roi = SERVE_ROI
        min_width = MIN_LINE_WIDTH
        if argv.custom_recognition_param:
            try:
                p = json.loads(argv.custom_recognition_param)
                if "roi" in p:
                    roi = p["roi"]
                if "min_line_width" in p:
                    min_width = int(p["min_line_width"])
            except Exception:
                pass

        image = argv.image
        h, w = image.shape[:2]
        rx, ry, rw, rh = roi
        x1 = max(0, rx)
        y1 = max(0, ry)
        x2 = min(w, rx + rw)
        y2 = min(h, ry + rh)
        if x2 <= x1 or y2 <= y1:
            return None

        try:
            roi_img = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            best = None
            best_w = 0
            for i in range(1, num):
                bw = stats[i, cv2.CC_STAT_WIDTH]
                if bw > best_w:
                    best_w = bw
                    best = i

            if best is not None and best_w >= min_width:
                bx = stats[best, cv2.CC_STAT_LEFT]
                by = stats[best, cv2.CC_STAT_TOP]
                bw = stats[best, cv2.CC_STAT_WIDTH]
                bh = stats[best, cv2.CC_STAT_HEIGHT]
                area = stats[best, cv2.CC_STAT_AREA]
                box = [x1 + bx, y1 + by, bw, bh]
                logger.debug(
                    "ServeDetect: found line width=%d area=%d box=%s",
                    bw, area, box,
                )
                return CustomRecognition.AnalyzeResult(
                    box=box,
                    detail={"width": int(bw), "area": int(area)},
                )

            logger.debug("ServeDetect: no line, max_width=%d", best_w)
            return None
        except Exception:
            logger.exception("ServeDetect: failed")
            return None
