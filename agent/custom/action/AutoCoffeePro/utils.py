import time
import cv2
import numpy as np

from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT


def get_image(controller):
    """Take a screenshot using the Maa controller API."""
    job = controller.post_screencap()
    job.wait()
    return controller.cached_image


def click_rect(controller, rect, delay=0.05):
    """Click the center of a rectangle [x, y, w, h]."""
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    controller.post_touch_down(cx, cy).wait()
    time.sleep(delay)
    controller.post_touch_up().wait()


def click_rect_multiple(controller, rect, repeat=3):
    """Click multiple times for reliability."""
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    for _ in range(repeat):
        controller.post_touch_down(cx, cy).wait()
        time.sleep(0.05)
        controller.post_touch_up().wait()


def press_key_f(controller):
    """Press and release the F key."""
    KEY_F = 70
    controller.post_key_down(KEY_F).wait()
    time.sleep(0.1)
    controller.post_key_up(KEY_F).wait()


def press_key_esc(controller):
    """Press and release the Escape key."""
    KEY_ESC = 27
    controller.post_key_down(KEY_ESC).wait()
    time.sleep(0.1)
    controller.post_key_up(KEY_ESC).wait()


def match_template_in_region(
    img, region, template, min_similarity=0.8, green_mask=False
):
    """OpenCV template matching within a region.

    Args:
        img: Full-screen image (numpy BGR array).
        region: [x, y, w, h] within the image.
        template: Template image (numpy array).
        min_similarity: Minimum match score threshold.
        green_mask: Whether to use green-channel masking.

    Returns:
        (hit, score, x, y) where (x, y) is the top-left in region coordinates.
    """
    if img is None or not isinstance(img, np.ndarray):
        return False, 0.0, 0, 0

    x1, y1, w, h = region
    x2, y2 = x1 + w, y1 + h

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False, 0.0, 0, 0

    roi = img[y1:y2, x1:x2]

    if len(roi.shape) == 3 and roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

    if green_mask:
        lower_green = np.array([0, 255, 0], dtype=np.uint8)
        upper_green = np.array([0, 255, 0], dtype=np.uint8)
        mask = cv2.bitwise_not(cv2.inRange(template, lower_green, upper_green))
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    np.clip(res, 0.0, 1.0, out=res)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= min_similarity:
        return True, max_val, x1 + max_loc[0], y1 + max_loc[1]
    return False, max_val, 0, 0


def wait_for_condition(
    condition_fn,
    timeout=30.0,
    interval=0.3,
    stopping=None,
):
    """Poll a condition function until it returns True or timeout.

    Args:
        condition_fn: Callable that returns a truthy value.
        timeout: Maximum seconds to wait.
        interval: Poll interval in seconds.
        stopping: Optional callable that returns True when task should stop.

    Returns:
        The truthy value from condition_fn, or None on timeout/stop.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stopping is not None and stopping():
            return None
        result = condition_fn()
        if result:
            return result
        time.sleep(interval)
    return None
