"""钓鱼控条的高频视觉识别。"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

CONTROL_ROI = (399, 43, 486, 14)
GREEN_LOWER = np.array((78, 141, 170), dtype=np.uint8)
GREEN_UPPER = np.array((86, 209, 241), dtype=np.uint8)
CURSOR_LOWER = np.array((24, 64, 253), dtype=np.uint8)
CURSOR_UPPER = np.array((30, 154, 255), dtype=np.uint8)


def _largest_component_box(
    mask: np.ndarray, offset_x: int, offset_y: int, min_count: int
) -> Optional[Box]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return None

    candidates = stats[1:]
    index = int(np.argmax(candidates[:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = candidates[index]
    if int(area) < min_count:
        return None
    return int(offset_x + x), int(offset_y + y), int(width), int(height)


def _all_points_box(
    mask: np.ndarray, offset_x: int, offset_y: int, min_count: int
) -> Optional[Box]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered_mask = np.zeros_like(mask)
    for index in range(1, component_count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_count:
            filtered_mask[labels == index] = 255
    if int(cv2.countNonZero(filtered_mask)) < min_count:
        return None
    points = cv2.findNonZero(filtered_mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return int(offset_x + x), int(offset_y + y), int(width), int(height)


def detect_control_boxes(image: np.ndarray) -> tuple[Optional[Box], Optional[Box]]:
    """在一次 HSV 转换中识别绿条和光标。"""
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None, None

    roi_x, roi_y, roi_width, roi_height = CONTROL_ROI
    if image.shape[1] < roi_x + roi_width or image.shape[0] < roi_y + roi_height:
        return None, None

    roi = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width, :3]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    cursor_mask = cv2.inRange(hsv, CURSOR_LOWER, CURSOR_UPPER)

    # 光标会覆盖绿条中间的像素，绿条必须合并所有匹配点。
    green_box = _all_points_box(green_mask, roi_x, roi_y, min_count=4)
    cursor_box = _largest_component_box(cursor_mask, roi_x, roi_y, min_count=20)
    return green_box, cursor_box
