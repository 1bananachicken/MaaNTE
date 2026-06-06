import time
from collections.abc import Sequence

import cv2
import numpy as np

from utils import screen


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    screen.update_frame_size_from_image(img)
    return img


def _rect_to_tuple(rect) -> tuple[int, int, int, int]:
    if hasattr(rect, "x") and hasattr(rect, "y"):
        width = getattr(rect, "w", getattr(rect, "width", None))
        height = getattr(rect, "h", getattr(rect, "height", None))
        if width is not None and height is not None:
            return int(rect.x), int(rect.y), int(width), int(height)

    if len(rect) < 4:
        raise ValueError("rect must contain x, y, w, h")

    x, y, w, h = rect[:4]
    return int(x), int(y), int(w), int(h)


def click_point(controller, point: Sequence[int | float], delay=0.001, move=False):
    x, y = point[:2]
    if move:
        controller.post_touch_move(round(x), round(y)).wait()
    controller.post_touch_down(round(x), round(y)).wait()
    time.sleep(delay)
    controller.post_touch_up().wait()


def click_point_720(controller, point: Sequence[int | float], delay=0.001, move=False):
    click_point(controller, screen.map_point_to_frame(point[0], point[1]), delay, move)


def click_rect(controller, rect, delay=0.001, move=False):
    x, y, w, h = _rect_to_tuple(rect)
    click_point(controller, (x + w / 2, y + h / 2), delay, move)


def click_rect_720(controller, rect, delay=0.001, move=False):
    click_rect(controller, screen.map_rect_to_frame(rect), delay, move)


def swipe(controller, begin: Sequence[int | float], end: Sequence[int | float], duration=200):
    begin_x, begin_y = begin[:2]
    end_x, end_y = end[:2]
    controller.post_swipe(round(begin_x), round(begin_y), round(end_x), round(end_y), duration).wait()


def swipe_720(controller, begin: Sequence[int | float], end: Sequence[int | float], duration=200):
    swipe(controller, screen.map_point_to_frame(begin[0], begin[1]), screen.map_point_to_frame(end[0], end[1]), duration)


def match_template_in_region(
    img,
    region,
    template,
    min_similarity=0.8,
    green_mask=False,
    scale_template=True,
):
    if img is None or not isinstance(img, np.ndarray):
        return False, 0.0, 0, 0

    if template is None or not isinstance(template, np.ndarray):
        return False, 0.0, 0, 0

    screen.update_frame_size_from_image(img)
    if scale_template:
        template = screen.resize_template_to_frame(template, green_mask=green_mask)

    img_h, img_w = img.shape[:2]
    x1, y1, roi_w, roi_h = screen.clamp_rect(region, img_w, img_h)
    x2, y2 = x1 + roi_w, y1 + roi_h

    if x2 <= x1 or y2 <= y1:
        return False, 0.0, 0, 0

    roi = img[y1:y2, x1:x2]

    if len(roi.shape) == 3 and roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

    if len(template.shape) == 3 and template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
        return False, 0.0, 0, 0

    if green_mask:
        lower_green = np.array([0, 255, 0], dtype=np.uint8)
        upper_green = np.array([0, 255, 0], dtype=np.uint8)
        mask = cv2.bitwise_not(cv2.inRange(template, lower_green, upper_green))
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_val):
        return False, 0.0, 0, 0

    if max_val >= min_similarity:
        return True, max_val, x1 + max_loc[0], y1 + max_loc[1]
    return False, max_val, 0, 0


def match_template_in_region_720(
    img,
    region,
    template,
    min_similarity=0.8,
    green_mask=False,
):
    if img is None or not isinstance(img, np.ndarray):
        return False, 0.0, 0, 0

    screen.update_frame_size_from_image(img)
    best = (False, 0.0, 0, 0)

    for candidate in screen.map_rect_to_frame_candidates(region):
        scaled_template = screen.resize_template(
            template,
            candidate.scale_x,
            candidate.scale_y,
            green_mask=green_mask,
        )
        hit, score, x, y = match_template_in_region(
            img,
            candidate.rect,
            scaled_template,
            min_similarity,
            green_mask,
            scale_template=False,
        )
        if score > best[1]:
            best = (score >= min_similarity, score, x, y)

    return best
