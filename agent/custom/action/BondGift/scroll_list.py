"""
可滚动列表通用工具 — 滚动+匹配+到底检测
"""
import time
import numpy as np
import cv2
from typing import Optional

from maa.context import Context
from .logger import get_logger

logger = get_logger(__name__)

MAX_SCROLLS = 15
SCROLL_DURATION = 500   # ms
SCROLL_WAIT = 0.4       # s — 等待滚动动画


def _run_swipe(context: Context, begin_y: int, end_y: int):
    """在列表区域中心 X 位置上下滑动"""
    override = {
        "_BondGiftSwipe": {
            "action": {
                "type": "Swipe",
                "param": {
                    "begin": [640, begin_y],
                    "end": [640, end_y],
                    "duration": SCROLL_DURATION,
                },
            }
        }
    }
    context.run_task("_BondGiftSwipe", pipeline_override=override)


def scroll_down(context: Context):
    """向下滚动一页"""
    _run_swipe(context, 600, 200)


def scroll_up(context: Context):
    """向上滚动一页"""
    _run_swipe(context, 200, 600)


def snap_roi(controller, roi: list) -> np.ndarray:
    """截取列表区域的 np 数组"""
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    x, y, w, h = roi
    return img[y : y + h, x : x + w].copy()


def is_stuck(prev: np.ndarray, curr: np.ndarray) -> bool:
    """两帧画面是否基本一致（到底了）"""
    diff = np.abs(curr.astype(int) - prev.astype(int)).mean()
    return diff < 3


def find_in_list(
    controller,
    context: Context,
    template: np.ndarray,
    list_roi: list,
    threshold: float = 0.7,
) -> Optional[tuple]:
    """
    在可滚动列表中匹配模板，自动滚动直到找到

    Returns:
        (global_x, global_y) 或 None
    """
    prev = None

    for i in range(MAX_SCROLLS):
        # 1. 截图
        img = controller.cached_image or (
            controller.post_screencap().wait() or controller.cached_image
        )
        job = controller.post_screencap()
        job.wait()
        img = controller.cached_image

        # 2. 模板匹配
        x1, y1, w, h = list_roi
        x2, y2 = x1 + w, y1 + h
        roi = img[y1:y2, x1:x2]
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return None

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return (x1 + max_loc[0] + template.shape[1] // 2,
                    y1 + max_loc[1] + template.shape[0] // 2)

        # 3. 滚一页
        scroll_down(context)
        time.sleep(SCROLL_WAIT)

        # 4. 检测到底
        curr = snap_roi(controller, list_roi)
        if prev is not None and is_stuck(prev, curr):
            break
        prev = curr.copy()

    return None
