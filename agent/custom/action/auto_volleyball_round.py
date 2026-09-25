import json
import time

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger

# 过场动画检测 ROI（画面中心区域）
TRANSITION_ROI = [400, 200, 480, 320]
# HSV 蓝色范围
HSV_LOWER = np.array([90, 30, 100])
HSV_UPPER = np.array([130, 255, 255])


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
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    total = mask.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / total


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
                p = json.loads(argv.custom_action_param)
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
                logger.debug("RoundWait: blue_ratio=%.3f seen=%s", ratio, transition_seen)

                if not transition_seen:
                    if ratio >= blue_threshold:
                        transition_seen = True
                        logger.info("RoundWait: transition started")
                else:
                    if ratio < blue_threshold:
                        # 过场消失，等角色回位
                        logger.info("RoundWait: transition ended, wait %.1fs", wait_after)
                        time.sleep(wait_after)
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
            return CustomAction.RunResult(success=True)
