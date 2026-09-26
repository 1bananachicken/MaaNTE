"""排球比赛内动作与识别集合（落点检测、八向移动、发球检测、起跳扣球、跳过剧情、回合等待、结算点击、视角初始化）。"""

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils.logger import logger


# 跳过提示 ROI（右上角，与 VolleyballSkipStory 一致）
SKIP_ROI = [1223, 29, 28, 26]
# 跳过确认弹窗：勾选框 ROI（与 VolleyballSkipCheckbox 一致）
CHECKBOX_ROI = [566, 361, 23, 24]
# 跳过确认弹窗：确认按钮 ROI（与 VolleyballSkipConfirm 一致）
CONFIRM_ROI = [734, 426, 173, 37]
# 模板匹配阈值
MATCH_THRESHOLD = 0.7
# 黑屏亮度阈值（平均灰度）
BLACK_THRESHOLD = 25
# 跳过提示检测超时
DETECT_TIMEOUT = 13.0
# 弹窗处理 + 黑屏等待超时
SETTLE_TIMEOUT = 8.0
# 轮询间隔
POLL_INTERVAL = 0.2
# 跳过提示对应的按键：Esc（与 VolleyballSkipStory 的 ClickKey 27 一致）
SKIP_KEY = 27


def _load_template(rel_path):
    """按仓库/发行版两种目录布局加载模板。"""
    _abs_path = Path(__file__).parents[3]
    if Path.exists(_abs_path / "assets"):
        _path = _abs_path / "assets/resource/base/image" / rel_path
    else:
        _path = _abs_path / "resource/base/image" / rel_path
    return cv2.imread(str(_path), cv2.IMREAD_COLOR)


_skip_template = _load_template("Volleyball/SkipButton.png")
_checkbox_template = _load_template("Volleyball/SkipCheckbox.png")
_confirm_template = _load_template("Volleyball/SkipConfirmButton.png")


def _match(image, template, roi, label):
    """在指定 ROI 内匹配模板，返回 (命中, 中心坐标)。"""
    if template is None:
        return False, 0, 0
    h, w = image.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return False, 0, 0
    roi_img = image[y1:y2, x1:x2]
    if roi_img.shape[0] < template.shape[0] or roi_img.shape[1] < template.shape[1]:
        return False, 0, 0
    result = cv2.matchTemplate(roi_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= MATCH_THRESHOLD:
        cx = x1 + max_loc[0] + template.shape[1] // 2
        cy = y1 + max_loc[1] + template.shape[0] // 2
        logger.debug("SkipIntro: %s matched val=%.3f center=(%d,%d)", label, max_val, cx, cy)
        return True, cx, cy
    return False, 0, 0


def _is_black(image):
    """判断画面是否为黑屏（平均灰度低于阈值）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < BLACK_THRESHOLD


def _screencap(controller):
    """截取当前画面，失败返回 None。"""
    controller.post_screencap().wait()
    return controller.cached_image


@AgentServer.custom_action("volleyball_skip_intro")
class VolleyballSkipIntro(CustomAction):
    """入口前置：等跳过提示→按 Esc→处理跳过确认弹窗→等黑屏结束。

    custom_action_param (JSON):
      detect_timeout: float   跳过提示检测超时，默认13秒
      settle_timeout: float   弹窗处理+黑屏等待超时，默认8秒
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info(
            "SkipIntro: run started, templates=%s",
            [_skip_template is not None, _checkbox_template is not None, _confirm_template is not None],
        )
        try:
            return self._run_impl(context, argv)
        except Exception as e:
            logger.exception("SkipIntro: exception: %s", e)
            return CustomAction.RunResult(success=False)

    def _run_impl(self, context, argv):
        controller = context.tasker.controller

        detect_timeout = DETECT_TIMEOUT
        settle_timeout = SETTLE_TIMEOUT
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param) if isinstance(argv.custom_action_param, str) else argv.custom_action_param
                detect_timeout = float(p.get("detect_timeout", detect_timeout))
                settle_timeout = float(p.get("settle_timeout", settle_timeout))
            except Exception:
                pass

        # 阶段1：等跳过提示出现，按 Esc 跳过（与 VolleyballSkipStory 一致）
        start = time.time()
        skipped = False
        while time.time() - start < detect_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            image = _screencap(controller)
            if image is None:
                time.sleep(POLL_INTERVAL)
                continue

            hit, cx, cy = _match(image, _skip_template, SKIP_ROI, "skip hint")
            if hit:
                logger.info("SkipIntro: skip hint found at (%d,%d), press Esc", cx, cy)
                controller.post_click_key(SKIP_KEY).wait()
                skipped = True
                break

            time.sleep(POLL_INTERVAL)

        if not skipped:
            logger.warning("SkipIntro: skip hint not found within %.1fs, proceed", detect_timeout)
            return CustomAction.RunResult(success=True)

        # 阶段2：处理跳过确认弹窗（勾选框→确认按钮），并等黑屏结束
        time.sleep(0.5)  # 按 Esc 后稍等，弹窗/动画开始
        checkbox_done = False
        confirm_done = False
        black_seen = False
        start = time.time()
        while time.time() - start < settle_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            image = _screencap(controller)
            if image is None:
                time.sleep(POLL_INTERVAL)
                continue

            if not checkbox_done:
                hit, cx, cy = _match(image, _checkbox_template, CHECKBOX_ROI, "skip checkbox")
                if hit:
                    logger.info("SkipIntro: skip checkbox found at (%d,%d), click", cx, cy)
                    controller.post_click(cx, cy).wait()
                    checkbox_done = True
                    time.sleep(POLL_INTERVAL)
                    continue

            if not confirm_done:
                hit, cx, cy = _match(image, _confirm_template, CONFIRM_ROI, "skip confirm")
                if hit:
                    logger.info("SkipIntro: skip confirm found at (%d,%d), click", cx, cy)
                    controller.post_click(cx, cy).wait()
                    confirm_done = True
                    time.sleep(POLL_INTERVAL)
                    continue

            if _is_black(image):
                black_seen = True
                logger.debug("SkipIntro: black screen detected")
            elif black_seen:
                # 已经见过黑屏，现在亮度恢复，动画结束
                logger.info("SkipIntro: black screen ended, wait 0.5s")
                time.sleep(0.5)
                return CustomAction.RunResult(success=True)

            time.sleep(POLL_INTERVAL)

        logger.warning(
            "SkipIntro: settle wait timeout (checkbox=%s confirm=%s), proceed",
            checkbox_done,
            confirm_done,
        )
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_view_setup")
class VolleyballViewSetup(CustomAction):
    """一次性视角初始化：鼠标下移到俯视角度（Z视角切换有记忆，仅重登后需手动切）。"""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        dy = 5500
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param) if isinstance(argv.custom_action_param, str) else argv.custom_action_param
                dy = int(p.get("dy", dy))
            except Exception:
                pass

        try:
            # 前置等待：上一步可能点了跳过键触发黑屏，黑屏未结束前鼠标操作无效
            time.sleep(0.5)
            controller.post_relative_move(0, dy).wait()
            time.sleep(0.5)
            return CustomAction.RunResult(success=True)
        except Exception:
            logger.exception("VolleyballViewSetup: failed")
            return CustomAction.RunResult(success=False)


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
MOVE_HSV_LOWER = np.array([90, 30, 150])
MOVE_HSV_UPPER = np.array([130, 255, 255])
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
    mask = cv2.inRange(hsv, MOVE_HSV_LOWER, MOVE_HSV_UPPER)
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


# 主控腰部 ROI（检测蓝色底线）
SERVE_ROI = [380, 335, 520, 50]
# HSV 蓝色范围（V下限100，覆盖底线蓝色）
SERVE_HSV_LOWER = np.array([90, 30, 100])
SERVE_HSV_UPPER = np.array([130, 255, 255])
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
            mask = cv2.inRange(hsv, SERVE_HSV_LOWER, SERVE_HSV_UPPER)
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


# 结算画面检测：左下角离开按钮
RESULT_ROI = [0, 600, 300, 120]
RESULT_THRESHOLD = 0.9

# 加载离开按钮模板
_abs_path = Path(__file__).parents[3]
if Path.exists(_abs_path / "assets"):
    _leave_btn_path = _abs_path / "assets/resource/base/image/Volleyball/ExitButton.png"
else:
    _leave_btn_path = _abs_path / "resource/base/image/Volleyball/ExitButton.png"
_leave_template = cv2.imread(str(_leave_btn_path), cv2.IMREAD_COLOR)


def _blue_ratio(image, roi):
    """计算ROI内蓝色像素占比。"""
    h, w = image.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi_img = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TRANSITION_HSV_LOWER, TRANSITION_HSV_UPPER)
    total = mask.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / total


def _is_result_screen(image):
    """检测是否为结算画面（左下角离开按钮模板匹配）。"""
    if _leave_template is None:
        return False
    h, w = image.shape[:2]
    rx, ry, rw, rh = RESULT_ROI
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return False
    roi_img = image[y1:y2, x1:x2]
    if (
        roi_img.shape[0] < _leave_template.shape[0]
        or roi_img.shape[1] < _leave_template.shape[1]
    ):
        return False
    result = cv2.matchTemplate(roi_img, _leave_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    logger.debug("RoundWait: result match=%.3f at %s", max_val, max_loc)
    return max_val >= RESULT_THRESHOLD


@AgentServer.custom_action("volleyball_round_wait")
class VolleyballRoundWait(CustomAction):
    """回合间等待：等过场动画出现→消失→角色回位，然后返回。

    custom_action_param (JSON):
      blue_ratio: float       过场蓝色占比阈值，默认 0.3
      wait_after: float       过场消失后等待秒数，默认 1.5
      timeout: float          总超时秒数，默认 10.0
      poll_interval: float    轮询间隔秒数，默认 0.1
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        blue_threshold = 0.3
        wait_after = 1.5
        timeout = 10.0
        poll_interval = 0.1
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param) if isinstance(argv.custom_action_param, str) else argv.custom_action_param
                blue_threshold = float(p.get("blue_ratio", blue_threshold))
                wait_after = float(p.get("wait_after", wait_after))
                timeout = float(p.get("timeout", timeout))
                poll_interval = float(p.get("poll_interval", poll_interval))
            except Exception:
                pass

        start = time.time()
        transition_seen = False

        try:
            while time.time() - start < timeout:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                controller.post_screencap().wait()
                image = controller.cached_image
                if image is None:
                    time.sleep(poll_interval)
                    continue

                ratio = _blue_ratio(image, TRANSITION_ROI)
                logger.debug(
                    "RoundWait: blue_ratio=%.3f seen=%s", ratio, transition_seen
                )

                if not transition_seen:
                    if ratio >= blue_threshold:
                        transition_seen = True
                        logger.info("RoundWait: transition started")
                else:
                    if ratio < blue_threshold:
                        # 过场消失，等角色回位
                        logger.info(
                            "RoundWait: transition ended, wait %.1fs", wait_after
                        )
                        time.sleep(wait_after)
                        return CustomAction.RunResult(success=True)

                # 结算画面检测：胜利结算蓝色占比与过场重叠，需单独判断
                if _is_result_screen(image):
                    logger.info("RoundWait: result screen detected, exit early")
                    return CustomAction.RunResult(success=True)

                time.sleep(poll_interval)

            # 超时：如果已经看到过场但没等到消失，也返回成功（避免卡死）
            if transition_seen:
                logger.warning("RoundWait: timeout but transition was seen, proceed")
                time.sleep(wait_after)
            else:
                logger.warning("RoundWait: timeout, no transition seen")
            return CustomAction.RunResult(success=True)
        except Exception:
            logger.exception("RoundWait: failed")
            return CustomAction.RunResult(success=False)


@AgentServer.custom_action("volleyball_click_result")
class VolleyballClickResult(CustomAction):
    """结算画面点击。

    custom_action_param (JSON):
      x: int    点击横坐标
      y: int    点击纵坐标
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        x = 140
        y = 610
        if argv.custom_action_param:
            try:
                p = (
                    json.loads(argv.custom_action_param)
                    if isinstance(argv.custom_action_param, str)
                    else argv.custom_action_param
                )
                x = int(p.get("x", x))
                y = int(p.get("y", y))
            except Exception:
                pass

        controller = context.tasker.controller
        controller.post_click(x, y).wait()
        return CustomAction.RunResult(success=True)
