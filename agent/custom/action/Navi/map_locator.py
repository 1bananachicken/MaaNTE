import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..Common.logger import get_logger
from .resources import resource_base_path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

logger = get_logger(__name__)


@dataclass
class MapLocationResult:
    found: bool
    point: tuple[int, int] | None
    raw_point: tuple[int, int] | None
    score: float
    mode: str
    polygon: np.ndarray | None = None


class MapLocator:
    MAP_SIZE = (11264, 11264)  # (16896, 18176)
    MINI_MAP_ROI = (28, 15, 150, 150)
    MAP_CROP_SIZES = (268, 530, 660)
    MAP_CROP_SIZE = MAP_CROP_SIZES[0]
    SCALE_PROBE_INTERVAL = 10
    MINI_GRAY_RANGE = (24, 58)
    BIG_GRAY_RANGE = (18, 77)
    SEARCH_RADIUS = 256
    GLOBAL_MIN_SCORE = 0.75
    LOCAL_MIN_SCORE = 0.65
    SMOOTHING_ALPHA = 0.3
    MIN_FILTER_PIXELS = 120
    CIRCLE_PADDING = 11
    CENTER_RADIUS = 11
    DEBUG_MAP_WIDTH = 900
    DEBUG_ZOOM_MAP_WIDTH = (
        4000  # resolution of the separate high-res map used for zoom crops
    )
    DEBUG_ZOOM_SIZE = 300  # pixel side length of the zoomed map inset
    DEBUG_ZOOM_RADIUS_DEFAULT = 400  # default half-side of the original-coord zoom area
    DEBUG_ZOOM_RADIUS_MAX = 2000
    DEBUG_WIN = "Map Locator NCC"
    # Each entry: (y_start, y_end, x_start, x_end) in original-image coordinates.
    # They are scaled for the candidate match map before indexing its cache.
    GLOBAL_SEARCH_REGIONS: list[tuple[int, int, int, int]] = [
        (8976, 9700, 1506, 2644),
        (2561, 8703, 2312, 7719),
    ]

    def __init__(self, big_map_path: Path | None = None, debug: bool = False):
        self.big_map_path = (
            Path(big_map_path)
            if big_map_path
            else (resource_base_path() / "image/map/bigworldmapSecond.png")
        )
        self.debug = debug
        self.last_center: tuple[int, int] | None = None
        self.smoothed_center: tuple[float, float] | None = None
        self._active_map_crop_index = 0
        self._frame_count = 0

        big_gray = cv2.imread(str(self.big_map_path), cv2.IMREAD_GRAYSCALE)
        if big_gray is None:
            raise ValueError(f"Failed to read big map: {self.big_map_path}")

        self.origin_h, self.origin_w = big_gray.shape
        if (self.origin_w, self.origin_h) != self.MAP_SIZE:
            logger.warning(
                f"Unexpected big map size: {self.origin_w}x{self.origin_h}, "
                f"expected {self.MAP_SIZE[0]}x{self.MAP_SIZE[1]}"
            )

        self._match_maps: list[tuple[float, np.ndarray]] = []
        for map_crop_size in self.MAP_CROP_SIZES:
            scale = self.MINI_MAP_ROI[2] / map_crop_size
            match_size = (
                int(round(self.origin_w * scale)),
                int(round(self.origin_h * scale)),
            )
            resized_gray = cv2.resize(
                big_gray, match_size, interpolation=cv2.INTER_AREA
            )
            self._match_maps.append(
                (scale, cv2.inRange(resized_gray, *self.BIG_GRAY_RANGE))
            )
        self._activate_map_crop_size(0)

        if self.debug:
            self.debug_scale = min(1.0, self.DEBUG_MAP_WIDTH / self.origin_w)
            debug_size = (
                int(round(self.origin_w * self.debug_scale)),
                int(round(self.origin_h * self.debug_scale)),
            )
            debug_gray = cv2.resize(big_gray, debug_size, interpolation=cv2.INTER_AREA)
            self.debug_map = cv2.cvtColor(debug_gray, cv2.COLOR_GRAY2BGR)
            # high-res map kept solely for sharp zoom crops
            self.zoom_map_scale = min(1.0, self.DEBUG_ZOOM_MAP_WIDTH / self.origin_w)
            zoom_map_size = (
                int(round(self.origin_w * self.zoom_map_scale)),
                int(round(self.origin_h * self.zoom_map_scale)),
            )
            zoom_map_gray = cv2.resize(
                big_gray, zoom_map_size, interpolation=cv2.INTER_AREA
            )
            self.zoom_map = cv2.cvtColor(zoom_map_gray, cv2.COLOR_GRAY2BGR)
            # zoom / pan state
            self._zoom_radius: int = self.DEBUG_ZOOM_RADIUS_DEFAULT
            self._zoom_pan: list[float] = [0.0, 0.0]  # pan offset in original coords
            self._drag_state: dict = {
                "active": False,
                "sx": 0,
                "sy": 0,
                "pan0": [0.0, 0.0],
            }
            self._debug_win_ready = False

        logger.info(
            f"NCC big map loaded: origin={self.origin_w}x{self.origin_h}, "
            f"crop_sizes={self.MAP_CROP_SIZES}"
        )

    def _activate_map_crop_size(self, index: int) -> None:
        previous_size = getattr(self, "map_crop_size", None)
        self._active_map_crop_index = index
        self.map_crop_size = self.MAP_CROP_SIZES[index]
        self.scale, self.big_match = self._match_maps[index]
        if previous_size is not None and previous_size != self.map_crop_size:
            logger.info(
                f"NCC map crop size switched: {previous_size} -> {self.map_crop_size}"
            )

    def locate(self, frame: np.ndarray) -> MapLocationResult:
        x, y, w, h = self.MINI_MAP_ROI
        if frame is None or y + h > frame.shape[0] or x + w > frame.shape[1]:
            raise ValueError(
                f"Frame does not contain mini map ROI: {self.MINI_MAP_ROI}"
            )

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

        if self._frame_count % self.SCALE_PROBE_INTERVAL == 0:
            matches = [
                self._match_template(template, template_mask, w, h, index)
                for index in range(len(self.MAP_CROP_SIZES))
            ]
            valid_matches = [
                (index, match) for index, match in enumerate(matches) if match.found
            ]
            if valid_matches:
                selected_index, result = max(
                    valid_matches,
                    key=lambda item: (
                        item[1].score,
                        item[0] == self._active_map_crop_index,
                    ),
                )
                self._activate_map_crop_size(selected_index)
            else:
                result = matches[self._active_map_crop_index]
        else:
            result = self._match_template(
                template, template_mask, w, h, self._active_map_crop_index
            )
        self._frame_count += 1

        if self._frame_count == 10000:
            self._frame_count = 0  # prevent overflow in long runs

        raw_point = result.raw_point
        polygon = result.polygon
        score = result.score
        mode = result.mode

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

        result = MapLocationResult(
            found=point is not None,
            point=point,
            raw_point=raw_point,
            score=float(score),
            mode=mode,
            polygon=polygon,
        )
        if self.debug:
            self.show_debug(template, result)

        return result

    def _match_template(
        self,
        template: np.ndarray,
        template_mask: np.ndarray,
        w: int,
        h: int,
        map_crop_index: int,
    ) -> MapLocationResult:
        scale, big_match = self._match_maps[map_crop_index]
        raw_point = None
        polygon = None
        score = 0.0
        mode = "rejected"

        if np.count_nonzero(template) >= self.MIN_FILTER_PIXELS:
            min_score = self.GLOBAL_MIN_SCORE
            mode = "global"

            if self.last_center is not None:
                # ---- local search ----
                center_x = self.last_center[0] * scale
                center_y = self.last_center[1] * scale
                radius = self.SEARCH_RADIUS * scale
                offset_x = max(0, int(math.floor(center_x - radius - w * 0.5)))
                offset_y = max(0, int(math.floor(center_y - radius - h * 0.5)))
                end_x = min(
                    big_match.shape[1], int(math.ceil(center_x + radius + w * 0.5))
                )
                end_y = min(
                    big_match.shape[0], int(math.ceil(center_y + radius + h * 0.5))
                )
                search_image = big_match[offset_y:end_y, offset_x:end_x]
                min_score = self.LOCAL_MIN_SCORE
                mode = "local"

                response = cv2.matchTemplate(
                    search_image, template, cv2.TM_CCORR_NORMED, mask=template_mask
                )
                response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
                response[(response < -1e-6) | (response > 1.0 + 1e-6)] = -1.0
                np.clip(response, 0.0, 1.0, out=response)
                _, score, _, max_loc = cv2.minMaxLoc(response)
                best_offset_x, best_offset_y = offset_x, offset_y
                best_loc = max_loc
            else:
                # ---- global search across all valid regions ----
                best_score_g = -1.0
                best_loc = None
                best_offset_x, best_offset_y = 0, 0
                for ry0, ry1, rx0, rx1 in self.GLOBAL_SEARCH_REGIONS:
                    # regions are defined in original-image coords; scale to match-image coords
                    mry0 = int(round(ry0 * scale))
                    mry1 = int(round(ry1 * scale))
                    mrx0 = int(round(rx0 * scale))
                    mrx1 = int(round(rx1 * scale))
                    region = big_match[mry0:mry1, mrx0:mrx1]
                    if region.shape[0] < h or region.shape[1] < w:
                        logger.debug(
                            f"Global search region ({rx0},{ry0})-({rx1},{ry1}) skipped: "
                            f"region too small ({region.shape[1]}x{region.shape[0]}) for template ({w}x{h})"
                        )
                        continue
                    resp = cv2.matchTemplate(
                        region, template, cv2.TM_CCORR_NORMED, mask=template_mask
                    )
                    resp = np.nan_to_num(resp, nan=-1.0, posinf=-1.0, neginf=-1.0)
                    resp[(resp < -1e-6) | (resp > 1.0 + 1e-6)] = -1.0
                    np.clip(resp, 0.0, 1.0, out=resp)
                    _, s, _, loc = cv2.minMaxLoc(resp)

                    if s > best_score_g:
                        best_score_g = s
                        best_loc = loc
                        best_offset_x, best_offset_y = mrx0, mry0
                score = best_score_g

            if best_loc is not None and score >= min_score:
                match_x = best_offset_x + best_loc[0]
                match_y = best_offset_y + best_loc[1]
                raw_point = (
                    int(round((match_x + w * 0.5) / scale)),
                    int(round((match_y + h * 0.5) / scale)),
                )
                polygon = (
                    np.float32(
                        [
                            [match_x, match_y],
                            [match_x + w, match_y],
                            [match_x + w, match_y + h],
                            [match_x, match_y + h],
                        ]
                    ).reshape(-1, 1, 2)
                    / scale
                )

        return MapLocationResult(
            found=raw_point is not None,
            point=raw_point,
            raw_point=raw_point,
            score=float(score),
            mode=mode if raw_point is not None else "rejected",
            polygon=polygon,
        )

    def _setup_debug_window(self) -> None:
        """Create the OpenCV window with trackbar and mouse callback (once)."""
        cv2.namedWindow(self.DEBUG_WIN, cv2.WINDOW_AUTOSIZE)

        def _on_zoom(val: int) -> None:
            self._zoom_radius = max(10, val)

        cv2.createTrackbar(
            "Zoom",
            self.DEBUG_WIN,
            self._zoom_radius,
            self.DEBUG_ZOOM_RADIUS_MAX,
            _on_zoom,
        )

        mini_w = 280
        map_w = self.debug_map.shape[1]
        zoom_x_start = mini_w + map_w  # pixel x where the zoom panel begins

        def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            ds = self._drag_state
            if event == cv2.EVENT_LBUTTONDOWN and x >= zoom_x_start:
                ds["active"] = True
                ds["sx"], ds["sy"] = x, y
                ds["pan0"] = list(self._zoom_pan)
            elif event == cv2.EVENT_MOUSEMOVE and ds["active"]:
                # 1 screen pixel = (2*r / zoom_side) original pixels
                r = self._zoom_radius
                ratio = (2.0 * r) / self.DEBUG_ZOOM_SIZE
                dx = (x - ds["sx"]) * ratio
                dy = (y - ds["sy"]) * ratio
                self._zoom_pan = [ds["pan0"][0] - dx, ds["pan0"][1] - dy]
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONDBLCLK):
                ds["active"] = False

        cv2.setMouseCallback(self.DEBUG_WIN, _on_mouse)
        self._debug_win_ready = True

    def show_debug(self, template: np.ndarray, result: MapLocationResult) -> None:
        if not self._debug_win_ready:
            self._setup_debug_window()

        # ---- full-map overview panel ----
        map_view = self.debug_map.copy()
        if result.polygon is not None:
            cv2.polylines(
                map_view,
                [np.int32(result.polygon * self.debug_scale)],
                True,
                (0, 255, 0),
                1,
            )
        if result.point is not None:
            point = tuple(int(value * self.debug_scale) for value in result.point)
            cv2.circle(map_view, point, 1, (0, 0, 255), -1)
        cv2.putText(
            map_view,
            f"ncc={result.score:.3f} crop={self.map_crop_size} mode={result.mode} "
            f"point={result.point} raw={result.raw_point}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ---- mini template panel ----
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

        # ---- zoomed inset panel (pannable + zoomable) ----
        zoom_side = self.DEBUG_ZOOM_SIZE
        r = self._zoom_radius
        if result.point is not None:
            # zoom center = match point + user pan offset
            cx = result.point[0] + self._zoom_pan[0]
            cy = result.point[1] + self._zoom_pan[1]
        else:
            cx = self.origin_w / 2 + self._zoom_pan[0]
            cy = self.origin_h / 2 + self._zoom_pan[1]
        cx = float(np.clip(cx, r, self.origin_w - r))
        cy = float(np.clip(cy, r, self.origin_h - r))

        z_x0 = cx - r
        z_y0 = cy - r
        z_x1 = cx + r
        z_y1 = cy + r
        # crop from the high-res zoom_map for sharpness
        zs = self.zoom_map_scale
        cx0 = int(max(0, z_x0 * zs))
        cy0 = int(max(0, z_y0 * zs))
        cx1 = int(min(self.zoom_map.shape[1], z_x1 * zs))
        cy1 = int(min(self.zoom_map.shape[0], z_y1 * zs))

        if cx1 > cx0 and cy1 > cy0:
            zoom_crop = self.zoom_map[cy0:cy1, cx0:cx1].copy()
            # draw polygon
            if result.polygon is not None:
                poly_shifted = (result.polygon - np.float32([[[z_x0, z_y0]]])) * zs
                cv2.polylines(zoom_crop, [np.int32(poly_shifted)], True, (0, 255, 0), 2)
            # draw match point
            if result.point is not None:
                dot_x = int((result.point[0] - z_x0) * zs)
                dot_y = int((result.point[1] - z_y0) * zs)
                cv2.circle(zoom_crop, (dot_x, dot_y), 4, (0, 0, 255), -1)
            zoom_view = cv2.resize(
                zoom_crop, (zoom_side, zoom_side), interpolation=cv2.INTER_AREA
            )
        else:
            zoom_view = np.zeros((zoom_side, zoom_side, 3), dtype=np.uint8)

        if result.point is None:
            cv2.putText(
                zoom_view,
                "no match",
                (10, zoom_side // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )

        # ---- assemble panels ----
        target_h = map_view.shape[0]
        panels = [mini_view, map_view]
        if zoom_view.shape[0] < target_h:
            zoom_view = cv2.copyMakeBorder(
                zoom_view, 0, target_h - zoom_view.shape[0], 0, 0, cv2.BORDER_CONSTANT
            )
        elif zoom_view.shape[0] > target_h:
            zoom_view = zoom_view[:target_h]
        panels.append(zoom_view)
        cv2.imshow(self.DEBUG_WIN, np.concatenate(panels, axis=1))


@AgentServer.custom_action("map_locator")
class MapLocatorTestAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
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
            locator = MapLocator(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                debug=debug,
            )
        except Exception as exc:
            logger.error(f"Map locator NCC init failed: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(
            f"Map locator NCC started: map={locator.big_map_path}, debug={debug}"
        )
        controller = context.tasker.controller
        last_result = MapLocationResult(False, None, None, 0.0, "init")

        try:
            while not context.tasker.stopping:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    continue

                last_result = locator.locate(frame)

                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                if debug:
                    logger.debug(f"time={time.perf_counter() - started:.3f}s")
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
