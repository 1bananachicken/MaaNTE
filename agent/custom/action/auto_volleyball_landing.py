import json
import math

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils.logger import logger


@AgentServer.custom_recognition("volleyball_landing_detect")
class VolleyballLandingDetect(CustomRecognition):
    """排球落点检测：HSV筛选 + 连通块面积+圆形度筛选，只返回符合落点特征的连通块。

    custom_recognition_param (JSON):
      roi: [x, y, w, h]       检测区域（必填）
      hsv_lower: [h,s,v]      HSV下限，默认 [90, 30, 150]
      hsv_upper: [h,s,v]      HSV上限，默认 [130, 255, 255]
      min_area: int           最小面积，默认 20
      max_area: int           最大面积，默认 300
      min_circ: float         最小圆形度，默认 0.3
    """

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> CustomRecognition.AnalyzeResult | None:
        # 解析参数（兼容对象和字符串两种格式）
        params = {}
        raw = argv.custom_recognition_param
        if raw:
            if isinstance(raw, dict):
                params = raw
            else:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        params = parsed
                except (json.JSONDecodeError, TypeError):
                    logger.error("VolleyballLandingDetect: param parse failed")
                    return None

        roi = params.get("roi")
        if not roi or len(roi) != 4:
            logger.error("VolleyballLandingDetect: missing roi param")
            return None

        hsv_lower = np.array(params.get("hsv_lower", [90, 30, 150]))
        hsv_upper = np.array(params.get("hsv_upper", [130, 255, 255]))
        min_area = int(params.get("min_area", 20))
        max_area = int(params.get("max_area", 300))
        min_circ = float(params.get("min_circ", 0.3))

        rx, ry, rw, rh = roi
        image = argv.image  # BGR numpy array

        try:
            # 裁剪ROI
            h, w = image.shape[:2]
            x1 = max(0, int(rx))
            y1 = max(0, int(ry))
            x2 = min(w, int(rx + rw))
            y2 = min(h, int(ry + rh))
            if x2 <= x1 or y2 <= y1:
                return None
            roi_img = image[y1:y2, x1:x2]

            # HSV筛选
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

            # 连通块分析
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            best = None
            best_area = 0
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area or area > max_area:
                    continue

                # 圆形度
                blob_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(
                    blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter <= 0:
                    continue
                circ = 4 * math.pi * area / (perimeter * perimeter)
                if circ < min_circ:
                    continue

                # 选面积最大的符合条件的连通块
                if area > best_area:
                    bx = stats[i, cv2.CC_STAT_LEFT]
                    by = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    best = (bx, by, bw, bh, area, circ)
                    best_area = area

            if best is None:
                return None

            bx, by, bw, bh, area, circ = best
            # 坐标映射回全图
            box = [x1 + bx, y1 + by, bw, bh]
            logger.debug(
                "VolleyballLandingDetect: found area=%d circ=%.3f box=%s",
                area, circ, box,
            )
            return CustomRecognition.AnalyzeResult(
                box=box,
                detail={"area": int(area), "circularity": round(circ, 3)},
            )
        except Exception:
            logger.exception("VolleyballLandingDetect: failed")
            return None
