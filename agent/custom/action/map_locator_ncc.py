import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


logger = get_logger(__name__)


@dataclass
class MapLocationNccResult:
    found: bool
    point: tuple[int, int] | None
    raw_point: tuple[int, int] | None
    score: float
    mode: str
    polygon: np.ndarray | None = None


class MapLocatorNcc:
    MAP_SIZE = (16896, 18176)
    MINI_MAP_ROI = (28, 20, 150, 150)
    MAP_CROP_SIZE = 751
    MINI_GRAY_RANGE = (24, 58)
    BIG_GRAY_RANGE = (18, 77)
    SEARCH_RADIUS = 520
    GLOBAL_MIN_SCORE = 0.52
    LOCAL_MIN_SCORE = 0.46
    SMOOTHING_ALPHA = 0.3
    MIN_FILTER_PIXELS = 120
    CIRCLE_PADDING = 15
    CENTER_RADIUS = 11
    DEBUG_MAP_WIDTH = 900

    def __init__(self, big_map_path: Path | None = None, debug: bool = False):
        root = Path(__file__).parents[3]
        resource_root = root / "assets" if (root / "assets").exists() else root
        self.big_map_path = Path(big_map_path) if big_map_path else (
            resource_root / "resource/base/image/map/map.jpg"
        )
        self.debug = debug
        self.last_center: tuple[int, int] | None = None
        self.smoothed_center: tuple[float, float] | None = None

        big_gray = cv2.imread(str(self.big_map_path), cv2.IMREAD_GRAYSCALE)
        if big_gray is None:
            raise ValueError(f"Failed to read big map: {self.big_map_path}")

        self.origin_h, self.origin_w = big_gray.shape
        if (self.origin_w, self.origin_h) != self.MAP_SIZE:
            logger.warning(
                f"Unexpected big map size: {self.origin_w}x{self.origin_h}, "
                f"expected {self.MAP_SIZE[0]}x{self.MAP_SIZE[1]}"
            )

        self.scale = self.MINI_MAP_ROI[2] / self.MAP_CROP_SIZE
        match_size = (
            int(round(self.origin_w * self.scale)),
            int(round(self.origin_h * self.scale)),
        )
        resized_gray = cv2.resize(big_gray, match_size, interpolation=cv2.INTER_AREA)
        self.big_match = cv2.inRange(resized_gray, *self.BIG_GRAY_RANGE)

        if self.debug:
            self.debug_scale = min(1.0, self.DEBUG_MAP_WIDTH / self.origin_w)
            debug_size = (
                int(round(self.origin_w * self.debug_scale)),
                int(round(self.origin_h * self.debug_scale)),
            )
            debug_gray = cv2.resize(big_gray, debug_size, interpolation=cv2.INTER_AREA)
            self.debug_map = cv2.cvtColor(debug_gray, cv2.COLOR_GRAY2BGR)

        logger.info(
            f"NCC big map loaded: origin={self.origin_w}x{self.origin_h}, "
            f"match={self.big_match.shape[1]}x{self.big_match.shape[0]}"
        )

    def locate(self, frame: np.ndarray) -> MapLocationNccResult:
        x, y, w, h = self.MINI_MAP_ROI
        if frame is None or y + h > frame.shape[0] or x + w > frame.shape[1]:
            raise ValueError(f"Frame does not contain mini map ROI: {self.MINI_MAP_ROI}")

        minimap = frame[y : y + h, x : x + w]
        if minimap.ndim == 2:
            mini_gray = minimap
        elif minimap.shape[2] == 4:
            mini_gray = cv2.cvtColor(minimap, cv2.COLOR_BGRA2GRAY)
        else:
            mini_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        template = cv2.inRange(mini_gray, *self.MINI_GRAY_RANGE)
        template_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        cv2.circle(template_mask, center, min(w, h) // 2 - self.CIRCLE_PADDING, 255, -1)
        cv2.circle(template_mask, center, self.CENTER_RADIUS, 0, -1)
        template = cv2.bitwise_and(template, template_mask)

        raw_point = None
        polygon = None
        score = 0.0
        mode = "rejected"

        if np.count_nonzero(template) >= self.MIN_FILTER_PIXELS:
            offset_x = 0
            offset_y = 0
            search_image = self.big_match
            min_score = self.GLOBAL_MIN_SCORE
            mode = "global"

            if self.last_center is not None:
                center_x = self.last_center[0] * self.scale
                center_y = self.last_center[1] * self.scale
                radius = self.SEARCH_RADIUS * self.scale
                offset_x = max(0, int(math.floor(center_x - radius - w * 0.5)))
                offset_y = max(0, int(math.floor(center_y - radius - h * 0.5)))
                end_x = min(self.big_match.shape[1], int(math.ceil(center_x + radius + w * 0.5)))
                end_y = min(self.big_match.shape[0], int(math.ceil(center_y + radius + h * 0.5)))
                search_image = self.big_match[offset_y:end_y, offset_x:end_x]
                min_score = self.LOCAL_MIN_SCORE
                mode = "local"

            response = cv2.matchTemplate(
                search_image,
                template,
                cv2.TM_CCORR_NORMED,
                mask=template_mask,
            )
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            response[(response < -1e-6) | (response > 1.0 + 1e-6)] = -1.0
            np.clip(response, 0.0, 1.0, out=response)
            _, score, _, max_loc = cv2.minMaxLoc(response)

            if score >= min_score:
                match_x = offset_x + max_loc[0]
                match_y = offset_y + max_loc[1]
                raw_point = (
                    int(round((match_x + w * 0.5) / self.scale)),
                    int(round((match_y + h * 0.5) / self.scale)),
                )
                polygon = np.float32(
                    [
                        [match_x, match_y],
                        [match_x + w, match_y],
                        [match_x + w, match_y + h],
                        [match_x, match_y + h],
                    ]
                ).reshape(-1, 1, 2) / self.scale

        if raw_point is None:
            self.last_center = None
            self.smoothed_center = None
            point = None
        elif self.smoothed_center is None:
            self.smoothed_center = tuple(float(value) for value in raw_point)
            point = raw_point
        else:
            alpha = self.SMOOTHING_ALPHA
            self.smoothed_center = (
                self.smoothed_center[0] * (1.0 - alpha) + raw_point[0] * alpha,
                self.smoothed_center[1] * (1.0 - alpha) + raw_point[1] * alpha,
            )
            point = tuple(int(round(value)) for value in self.smoothed_center)
        self.last_center = raw_point

        result = MapLocationNccResult(
            found=point is not None,
            point=point,
            raw_point=raw_point,
            score=float(score),
            mode=mode if raw_point is not None else "rejected",
            polygon=polygon,
        )
        if self.debug:
            self.show_debug(template, result)
        return result

    def show_debug(self, template: np.ndarray, result: MapLocationNccResult) -> None:
        map_view = self.debug_map.copy()
        if result.polygon is not None:
            cv2.polylines(map_view, [np.int32(result.polygon * self.debug_scale)], True, (0, 255, 0), 2)
        if result.point is not None:
            point = tuple(int(value * self.debug_scale) for value in result.point)
            cv2.circle(map_view, point, 5, (0, 0, 255), -1)

        cv2.putText(
            map_view,
            f"ncc={result.score:.3f} mode={result.mode} point={result.point} raw={result.raw_point}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        mini_view = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        mini_view = cv2.resize(mini_view, (280, 280), interpolation=cv2.INTER_NEAREST)
        if mini_view.shape[0] < map_view.shape[0]:
            mini_view = cv2.copyMakeBorder(
                mini_view,
                0,
                map_view.shape[0] - mini_view.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
            )
        cv2.imshow("Map Locator NCC", np.concatenate([mini_view, map_view], axis=1))


@AgentServer.custom_action("map_locator_ncc")
class MapLocatorNccTestAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = argv.custom_action_param or {}
        if not isinstance(params, dict):
            try:
                params = json.loads(params)
            except Exception as exc:
                logger.warning(f"Parse custom_action_param failed, use defaults: {exc}")
                params = {}
        if not isinstance(params, dict):
            params = {}

        debug = bool(params.get("debug", False))
        try:
            locator = MapLocatorNcc(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                debug=debug,
            )
        except Exception as exc:
            logger.error(f"Map locator NCC init failed: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(f"Map locator NCC started: map={locator.big_map_path}, debug={debug}")
        controller = context.tasker.controller
        last_result = MapLocationNccResult(False, None, None, 0.0, "init")

        try:
            while not context.tasker.stopping:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    continue

                last_result = locator.locate(frame)
                logger.info(
                    f"Map NCC location result: point={last_result.point}, "
                    f"raw={last_result.raw_point}, "
                    f"score={last_result.score:.3f}, mode={last_result.mode}"
                )
                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                sleep_time = 0.1 - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Map locator NCC failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            if debug:
                cv2.destroyAllWindows()

        return CustomAction.RunResult(success=last_result.found)
