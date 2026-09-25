import json
import math
import time

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

# 方向键 VK 码
KEY_W = 87
KEY_A = 65
KEY_S = 83
KEY_D = 68
KEY_LBUTTON = 1

# 区域划分（基于1280x720）
CENTER_X1, CENTER_Y1, CENTER_X2, CENTER_Y2 = 580, 300, 700, 420
# 大检测ROI：覆盖所有方向
DETECT_ROI = [470, 190, 340, 340]

# HSV 落点筛选
HSV_LOWER = np.array([90, 30, 150])
HSV_UPPER = np.array([130, 255, 255])
MIN_AREA = 20
MAX_AREA = 300
MIN_CIRC = 0.3

# 过场动画检测
TRANSITION_ROI = [400, 200, 480, 320]
TRANSITION_BLUE_RATIO = 0.3
TRANSITION_HSV_LOWER = np.array([90, 30, 100])
TRANSITION_HSV_UPPER = np.array([130, 255, 255])


def _is_transition(image):
    """检测是否为过场动画（全屏蓝色）。"""
    h, w = image.shape[:2]
    rx, ry, rw, rh = TRANSITION_ROI
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return False
    roi_img = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TRANSITION_HSV_LOWER, TRANSITION_HSV_UPPER)
    total = mask.size
    if total == 0:
        return False
    ratio = float(np.count_nonzero(mask)) / total
    return ratio >= TRANSITION_BLUE_RATIO


def _detect_landing(image):
    """在大ROI内检测落点，返回落点中心坐标(cx, cy)或None。"""
    h, w = image.shape[:2]
    rx, ry, rw, rh = DETECT_ROI
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return None
    roi_img = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best = None
    best_area = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        blob_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue
        circ = 4 * math.pi * area / (perimeter * perimeter)
        if circ < MIN_CIRC:
            continue
        if area > best_area:
            best_area = area
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
            cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
            best = (x1 + cx, y1 + cy)
    return best


def _classify_direction(cx, cy):
    """根据落点中心坐标判断方向。"""
    if CENTER_X1 <= cx <= CENTER_X2 and CENTER_Y1 <= cy <= CENTER_Y2:
        return "center"
    if cx < CENTER_X1:
        return "left"
    if cx > CENTER_X2:
        return "right"
    if cy < CENTER_Y1:
        return "up"
    return "down"


def _press_key(controller, key, duration=0.15):
    """长按方向键 duration 秒。"""
    controller.post_key_down(key).wait()
    time.sleep(duration)
    controller.post_key_up(key).wait()


@AgentServer.custom_action("volleyball_move_to_landing")
class VolleyballMoveToLanding(CustomAction):
    """闭环移动：持续检测落点位置，长按方向键移动，直到落点进入中心区域后按左键传球。

    custom_action_param (JSON):
      move_duration: float  单次方向键长按秒数，默认 0.15
      wait_after_move: float  移动后等待秒数，默认 0.1
      timeout: float  总超时秒数，默认 3.0
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        move_duration = 0.15
        wait_after_move = 0.1
        timeout = 3.0
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param)
                move_duration = float(p.get("move_duration", move_duration))
                wait_after_move = float(p.get("wait_after_move", wait_after_move))
                timeout = float(p.get("timeout", timeout))
            except Exception:
                pass

        start = time.time()
        try:
            # 第一次检测：没有落点说明球在对方半场，直接返回不移动
            controller.post_screencap().wait()
            image = controller.cached_image
            if image is None:
                return CustomAction.RunResult(success=True)

            # 过场动画出现，立即返回
            if _is_transition(image):
                logger.debug("MoveToLanding: transition detected, skip")
                return CustomAction.RunResult(success=True)

            landing = _detect_landing(image)
            if landing is None:
                return CustomAction.RunResult(success=True)

            # 检测到落点，进入闭环移动
            while time.time() - start < timeout:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                cx, cy = landing
                direction = _classify_direction(cx, cy)
                logger.debug(
                    "MoveToLanding: landing=(%.0f,%.0f) dir=%s", cx, cy, direction
                )

                if direction == "center":
                    controller.post_key_down(KEY_LBUTTON).wait()
                    time.sleep(0.05)
                    controller.post_key_up(KEY_LBUTTON).wait()
                    time.sleep(0.3)
                    return CustomAction.RunResult(success=True)

                key_map = {
                    "left": KEY_A,
                    "right": KEY_D,
                    "up": KEY_W,
                    "down": KEY_S,
                }
                _press_key(controller, key_map[direction], move_duration)
                time.sleep(wait_after_move)

                controller.post_screencap().wait()
                image = controller.cached_image
                if image is not None:
                    # 过场动画出现，立即返回
                    if _is_transition(image):
                        logger.debug("MoveToLanding: transition during move, skip")
                        return CustomAction.RunResult(success=True)
                    landing = _detect_landing(image)
                    if landing is None:
                        return CustomAction.RunResult(success=True)

            logger.warning("MoveToLanding: timeout after %.1fs", timeout)
            return CustomAction.RunResult(success=True)
        except Exception:
            logger.exception("MoveToLanding: failed")
            return CustomAction.RunResult(success=False)
