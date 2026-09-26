"""
自动排球 - 入口前置：检测并点击右上角跳过按钮，等待黑屏结束。
进入排球比赛时会播放开场动画，右上角有跳过按钮。
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

# 跳过按钮 ROI（右上角）
SKIP_ROI = [1150, 10, 100, 80]
# 模板匹配阈值
SKIP_THRESHOLD = 0.7
# 黑屏亮度阈值（平均灰度）
BLACK_THRESHOLD = 25
# 检测超时
DETECT_TIMEOUT = 13.0
# 黑屏等待超时
BLACK_TIMEOUT = 3.0
# 点击偏移（模板中心与实际按钮中心的偏差）
CLICK_OFFSET_X = 30
CLICK_OFFSET_Y = 0
# 轮询间隔
POLL_INTERVAL = 0.2

# 加载跳过按钮模板
_abs_path = Path(__file__).parents[3]
if Path.exists(_abs_path / "assets"):
    _skip_path = _abs_path / "assets/resource/base/image/VolleyballWeekly/SkipButton.png"
else:
    _skip_path = _abs_path / "resource/base/image/VolleyballWeekly/SkipButton.png"
_skip_template = cv2.imread(str(_skip_path), cv2.IMREAD_COLOR)


def _match_skip(image):
    """在右上角ROI内匹配跳过按钮，返回 (命中, 中心坐标)。"""
    if _skip_template is None:
        return False, 0, 0
    h, w = image.shape[:2]
    rx, ry, rw, rh = SKIP_ROI
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return False, 0, 0
    roi_img = image[y1:y2, x1:x2]
    if roi_img.shape[0] < _skip_template.shape[0] or roi_img.shape[1] < _skip_template.shape[1]:
        return False, 0, 0
    result = cv2.matchTemplate(roi_img, _skip_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= SKIP_THRESHOLD:
        cx = x1 + max_loc[0] + _skip_template.shape[1] // 2
        cy = y1 + max_loc[1] + _skip_template.shape[0] // 2
        return True, cx, cy
    return False, 0, 0


def _is_black(image):
    """判断画面是否为黑屏（平均灰度低于阈值）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < BLACK_THRESHOLD


@AgentServer.custom_action("volleyball_skip_intro")
class VolleyballSkipIntro(CustomAction):
    """入口前置：检测跳过按钮→点击→等黑屏结束。

    custom_action_param (JSON):
      detect_timeout: float   跳过按钮检测超时，默认13秒
      black_timeout: float    黑屏等待超时，默认3秒
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("SkipIntro: run started, template_loaded=%s", _skip_template is not None)
        try:
            return self._run_impl(context, argv)
        except Exception as e:
            logger.exception("SkipIntro: exception: %s", e)
            return CustomAction.RunResult(success=True)

    def _run_impl(self, context, argv):
        controller = context.tasker.controller
        logger.info("SkipIntro: controller ready")

        detect_timeout = DETECT_TIMEOUT
        black_timeout = BLACK_TIMEOUT
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param)
                detect_timeout = float(p.get("detect_timeout", detect_timeout))
                black_timeout = float(p.get("black_timeout", black_timeout))
            except Exception:
                pass

        # 阶段1：循环检测跳过按钮
        start = time.time()
        clicked = False
        while time.time() - start < detect_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            controller.post_screencap().wait()
            image = controller.cached_image
            if image is None:
                time.sleep(POLL_INTERVAL)
                continue

            hit, cx, cy = _match_skip(image)
            if hit:
                logger.info("SkipIntro: skip button found at (%d,%d), click (%d,%d)", cx, cy, cx + CLICK_OFFSET_X, cy + CLICK_OFFSET_Y)
                controller.post_click(cx + CLICK_OFFSET_X, cy + CLICK_OFFSET_Y).wait()
                clicked = True
                break

            time.sleep(POLL_INTERVAL)

        if not clicked:
            logger.warning("SkipIntro: skip button not found within %.1fs, proceed", detect_timeout)
            return CustomAction.RunResult(success=True)

        # 阶段2：等黑屏出现后结束
        time.sleep(0.5)  # 点击后稍等，动画开始
        black_seen = False
        start = time.time()
        while time.time() - start < black_timeout:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            controller.post_screencap().wait()
            image = controller.cached_image
            if image is None:
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

        logger.warning("SkipIntro: black screen wait timeout, proceed")
        return CustomAction.RunResult(success=True)
