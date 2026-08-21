import cv2
import numpy as np
import time

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context

from utils.logger import logger
from .utils import get_image


_background_gray: np.ndarray = None
_background_region_name: str = ""
_background_captured: bool = False


@AgentServer.custom_recognition("BackgroundDiffPro")
class BackgroundDiffPro(CustomRecognition):
    """Detect foreground objects (e.g. customers) via background subtraction.

    Captures a clean background image on first call, then performs
    frame differencing on subsequent calls to detect changes.

    Pipeline usage:
    ```jsonc
    {
        "DetectCustomers": {
            "recognition": {
                "type": "Custom",
                "param": {
                    "custom_recognition": "BackgroundDiffPro",
                    "custom_recognition_param": {
                        "region_name": "gameplay_area",
                        "diff_threshold": 28,
                        "min_area": 1800,
                        "max_area": 80000
                    }
                }
            },
            "next": ["ServeCustomer"]
        }
    }
    ```
    """

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> CustomRecognition.AnalyzeResult:
        global _background_gray, _background_region_name, _background_captured

        controller = context.tasker.controller
        img = get_image(controller)

        # Parse parameters
        region_name = "gameplay_area"
        diff_threshold = 28
        min_area = 1800
        max_area = 80000
        morph_kernel = 5
        capture_delay = 1.2

        if argv.custom_recognition_param:
            params = argv.custom_recognition_param
            region_name = params.get("region_name", region_name)
            diff_threshold = params.get("diff_threshold", diff_threshold)
            min_area = params.get("min_area", min_area)
            max_area = params.get("max_area", max_area)
            morph_kernel = params.get("morph_kernel", morph_kernel)
            capture_delay = params.get("capture_delay", capture_delay)

        # Parse region from param or use full image
        roi = argv.custom_recognition_param.get("roi", [0, 0, img.shape[1], img.shape[0]]) if argv.custom_recognition_param else [0, 0, img.shape[1], img.shape[0]]
        x, y, w, h = roi
        frame_roi = img[y : y + h, x : x + w]
        frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # Capture clean background on first call
        if not _background_captured or _background_region_name != region_name:
            logger.info("BackgroundDiffPro: capturing clean background for '%s'", region_name)
            time.sleep(capture_delay)
            img = get_image(controller)
            frame_roi = img[y : y + h, x : x + w]
            _background_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            _background_region_name = region_name
            _background_captured = True
            logger.info("BackgroundDiffPro: background captured (shape=%s)", _background_gray.shape)

        if _background_gray.shape != frame_gray.shape:
            logger.warning(
                "BackgroundDiffPro: background shape %s != frame shape %s, recapturing",
                _background_gray.shape, frame_gray.shape,
            )
            _background_gray = frame_gray.copy()

        # Compute absolute difference
        diff = cv2.absdiff(_background_gray, frame_gray)
        _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 20 or bh < 20:
                continue
            # Convert back to full-image coordinates
            boxes.append([x + bx, y + by, bw, bh])

        if boxes:
            logger.debug("BackgroundDiffPro: detected %d foreground boxes", len(boxes))
            return CustomRecognition.AnalyzeResult(
                box=(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]),
                detail={"boxes": boxes, "count": len(boxes)},
            )

        return None


@AgentServer.custom_recognition("BackgroundDiffProReset")
class BackgroundDiffProReset(CustomRecognition):
    """Reset the captured background so it gets re-captured on next call.

    Pipeline usage: call this before entering a new level/scene
    to ensure the background reflects the new environment.
    """

    def analyze(
        self, context: Context, argv: CustomRecognition.AnalyzeArg
    ) -> CustomRecognition.AnalyzeResult:
        global _background_captured, _background_gray, _background_region_name
        _background_captured = False
        _background_gray = None
        _background_region_name = ""
        logger.info("BackgroundDiffProReset: background reset")
        return CustomRecognition.AnalyzeResult(box=(0, 0, 0, 0))
