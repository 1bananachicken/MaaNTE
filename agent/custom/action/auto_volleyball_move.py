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

_KEY_NAMES = {KEY_W: "W", KEY_A: "A", KEY_S: "S", KEY_D: "D"}

# 屏幕中心（1280x720）
SCREEN_CENTER_X = 640
SCREEN_CENTER_Y = 360

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


def _direction_to_keys(dx, dy):
    """根据向量(dx, dy)计算八向按键列表。

    扇区划分（每45°一个）：
      0=右(D)  1=右下(S+D)  2=下(S)  3=左下(A+S)
      4=左(A)  5=左上(W+A)  6=上(W)  7=右上(W+D)
    """
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += 2 * math.pi
    sector = int((angle + math.pi / 8) / (math.pi / 4)) % 8
    key_map = [
        [KEY_D],
        [KEY_S, KEY_D],
        [KEY_S],
        [KEY_A, KEY_S],
        [KEY_A],
        [KEY_W, KEY_A],
        [KEY_W],
        [KEY_W, KEY_D],
    ]
    return key_map[sector]


def _press_keys(controller, keys, duration):
    """同时长按多个方向键 duration 秒。"""
    for key in keys:
        controller.post_key_down(key).wait()
    try:
        time.sleep(duration)
    finally:
        for key in keys:
            controller.post_key_up(key).wait()


@AgentServer.custom_action("volleyball_move_to_landing")
class VolleyballMoveToLanding(CustomAction):
    """开环八向移动：检测落点坐标，以屏幕中心为基准计算向量距离，
    按移动速度推算按键时延，一次移动到位后按左键传球。

    custom_action_param (JSON):
      speed: float          移动速度 px/s，默认 325
      center_radius: float  到位判定半径（像素），默认 50
      min_hold: float       最小按键时延秒，默认 0.05
      max_hold: float       最大按键时延秒，默认 1.0
      auto_pass: bool       移动到位后是否自动按左键传球，默认 true
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        speed = 325.0
        center_radius = 50.0
        min_hold = 0.05
        max_hold = 1.0
        auto_pass = True
        if argv.custom_action_param:
            try:
                p = (
                    json.loads(argv.custom_action_param)
                    if isinstance(argv.custom_action_param, str)
                    else argv.custom_action_param
                )
                speed = float(p.get("speed", speed))
                center_radius = float(p.get("center_radius", center_radius))
                min_hold = float(p.get("min_hold", min_hold))
                max_hold = float(p.get("max_hold", max_hold))
                auto_pass = bool(p.get("auto_pass", auto_pass))
            except Exception:
                pass

        try:
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

            cx, cy = landing
            dx = cx - SCREEN_CENTER_X
            dy = cy - SCREEN_CENTER_Y
            distance = math.hypot(dx, dy)

            logger.debug(
                "MoveToLanding: landing=(%.0f,%.0f) vector=(%.0f,%.0f) dist=%.1f",
                cx,
                cy,
                dx,
                dy,
                distance,
            )

            # 落点在中心半径内，直接传球
            if distance <= center_radius:
                logger.debug("MoveToLanding: already in center radius, pass directly")
            else:
                keys = _direction_to_keys(dx, dy)
                hold_time = max(min_hold, min(max_hold, distance / speed))
                logger.debug(
                    "MoveToLanding: keys=%s hold=%.3fs",
                    [_KEY_NAMES.get(k, str(k)) for k in keys],
                    hold_time,
                )
                _press_keys(controller, keys, hold_time)

            if auto_pass:
                controller.post_key_down(KEY_LBUTTON).wait()
                time.sleep(0.05)
                controller.post_key_up(KEY_LBUTTON).wait()
                time.sleep(0.3)

            return CustomAction.RunResult(success=True)
        except Exception:
            logger.exception("MoveToLanding: failed")
            return CustomAction.RunResult(success=False)
