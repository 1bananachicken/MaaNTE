import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(
    0, str(Path(__file__).parents[1] / "agent" / "custom" / "action" / "AutoFish")
)

from fish_vision import CONTROL_ROI, detect_control_boxes


class FishVisionTests(unittest.TestCase):
    def test_detects_current_green_boundary_and_cursor(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_x, roi_y, _, _ = CONTROL_ROI
        green_bgr = cv2.cvtColor(np.uint8([[[82, 180, 210]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cursor_bgr = cv2.cvtColor(np.uint8([[[27, 100, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        image[roi_y + 2 : roi_y + 10, roi_x + 120 : roi_x + 215] = green_bgr
        image[roi_y : roi_y + 12, roi_x + 195 : roi_x + 198] = cursor_bgr

        green_box, cursor_box = detect_control_boxes(image)

        self.assertEqual(green_box, (roi_x + 120, roi_y + 2, 95, 8))
        self.assertEqual(cursor_box, (roi_x + 195, roi_y, 3, 12))

    def test_rejects_cursor_component_below_minimum_count(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_x, roi_y, _, _ = CONTROL_ROI
        cursor_bgr = cv2.cvtColor(np.uint8([[[27, 100, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        image[roi_y : roi_y + 3, roi_x : roi_x + 3] = cursor_bgr

        _, cursor_box = detect_control_boxes(image)

        self.assertIsNone(cursor_box)


if __name__ == "__main__":
    unittest.main()
