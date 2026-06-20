import math
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..Common.logger import get_logger
from ..Common.utils import match_template_in_region
from .debug_windows import pump_debug_windows
from .resource_paths import resource_base_path

logger = get_logger(__name__)


@dataclass
class MapLocationResult:
    found: bool
    point: tuple[int, int] | None
    raw_point: tuple[int, int] | None
    score: float
    mode: str
    polygon: np.ndarray | None = None
    raw_coordinate: tuple[float, float] | tuple[float, float, float] | None = None


class MapLocator:
    MAP_SIZE = (11264, 11264)  # 大地图模板尺寸
    MINI_MAP_ROI = (28, 15, 150, 150)  # 小地图ROI
    BUTTON_ROI = (16, 656, 31, 35)
    MAP_CROP_SIZES = (
        268,
        530,
        660,
    )  # 大地图匹配模板尺寸（对应不同缩放级别下的小地图尺寸）
    SEARCH_RADIUS = 256  # 邻域搜索半径
    GLOBAL_MIN_SCORE = 0.85  # 全局搜索的最低置信度
    LOCAL_MIN_SCORE = 0.75  # 邻域搜索的最低置信度
    TELEPORT_DISTANCE = 320
    SMOOTHING_ALPHA = 0.7  # 越小越平滑，但是响应越慢
    MIN_FILTER_PIXELS = (
        120  # 模板中至少要有这么多有效像素才进行匹配，否则直接放弃，避免误匹配
    )
    CIRCLE_PADDING = 11  # 小地图圆形掩码的内边距，单位像素

    # 由于地图上存在大量纯黑区域，因此限定搜索范围，降低计算量
    # 这些坐标在原图（11264x11264）上定义，代码中会根据当前缩放自动调整
    # 坐标格式 [y_start, y_end, x_start, x_end]
    GLOBAL_SEARCH_REGIONS: list[tuple[int, int, int, int]] = [
        (8976, 9700, 1506, 2644),
        (2561, 8703, 2312, 7719),
    ]

    # debug visualization parameters
    DEBUG_MAP_WIDTH = 900
    DEBUG_ZOOM_MAP_WIDTH = 4000
    DEBUG_ZOOM_SIZE = 300
    DEBUG_ZOOM_RADIUS_DEFAULT = 400
    DEBUG_ZOOM_RADIUS_MAX = 2000
    DEBUG_WIN = "Map Locator"
    _shared_assets: dict[str, Any] | None = None
    _shared_assets_lock = threading.Lock()

    def __init__(self, debug: bool = False):
        self._closed = False
        assets = self.shared_assets()
        self.big_map_path = assets["big_map_path"]
        self.chat_button_path = assets["chat_button_path"]
        self.chat_template = assets["chat_template"]
        self.debug = debug
        self.last_center: tuple[int, int] | None = None
        self.smoothed_center: tuple[float, float] | None = None
        self._active_map_crop_index = 0
        self.origin_h = assets["origin_h"]
        self.origin_w = assets["origin_w"]

        # 由于模板匹配对缩放敏感，需要预生成多尺度匹配模板，适配不同速度下的小地图尺寸
        # Fucking Neverness to Everness
        self._match_maps = assets["match_maps"]
        self.activate_map_crop_size(0)

        if self.debug:
            big_gray = cv2.imread(str(self.big_map_path), cv2.IMREAD_GRAYSCALE)
            if big_gray is None:
                raise ValueError(f"Failed to read big map: {self.big_map_path}")
            self.debug_scale = min(1.0, self.DEBUG_MAP_WIDTH / self.origin_w)
            debug_size = (
                int(round(self.origin_w * self.debug_scale)),
                int(round(self.origin_h * self.debug_scale)),
            )
            debug_gray = cv2.resize(big_gray, debug_size, interpolation=cv2.INTER_AREA)
            self.debug_map = cv2.cvtColor(debug_gray, cv2.COLOR_GRAY2BGR)

            self.zoom_map_scale = min(1.0, self.DEBUG_ZOOM_MAP_WIDTH / self.origin_w)
            zoom_map_size = (
                int(round(self.origin_w * self.zoom_map_scale)),
                int(round(self.origin_h * self.zoom_map_scale)),
            )
            zoom_map_gray = cv2.resize(
                big_gray, zoom_map_size, interpolation=cv2.INTER_AREA
            )
            self.zoom_map = cv2.cvtColor(zoom_map_gray, cv2.COLOR_GRAY2BGR)

            self._zoom_radius: int = self.DEBUG_ZOOM_RADIUS_DEFAULT
            self._zoom_pan: list[float] = [0.0, 0.0]
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

    @classmethod
    def shared_assets(cls) -> dict[str, Any]:
        with cls._shared_assets_lock:
            if cls._shared_assets is None:
                cls._shared_assets = cls.load_shared_assets()
            return cls._shared_assets

    @classmethod
    def load_shared_assets(cls) -> dict[str, Any]:
        big_map_path = resource_base_path() / "image/map/bigworldmapSecond.png"
        chat_button_path = resource_base_path() / "image/Common/Button/InWorld/Chat.png"
        chat_template = cv2.imread(str(chat_button_path), cv2.IMREAD_COLOR)
        big_gray = cv2.imread(str(big_map_path), cv2.IMREAD_GRAYSCALE)
        if big_gray is None:
            raise ValueError(f"Failed to read big map: {big_map_path}")

        origin_h, origin_w = big_gray.shape
        if (origin_w, origin_h) != cls.MAP_SIZE:
            logger.warning(
                f"Unexpected big map size: {origin_w}x{origin_h}, "
                f"expected {cls.MAP_SIZE[0]}x{cls.MAP_SIZE[1]}"
            )

        match_maps: list[tuple[float, np.ndarray]] = []
        for map_crop_size in cls.MAP_CROP_SIZES:
            scale = cls.MINI_MAP_ROI[2] / map_crop_size
            match_size = (
                int(round(origin_w * scale)),
                int(round(origin_h * scale)),
            )
            resized_gray = cv2.resize(
                big_gray,
                match_size,
                interpolation=cv2.INTER_AREA,
            )
            match_maps.append((scale, resized_gray))

        logger.info(
            f"NCC shared map assets loaded: origin={origin_w}x{origin_h}, "
            f"crop_sizes={cls.MAP_CROP_SIZES}"
        )
        return {
            "big_map_path": big_map_path,
            "chat_button_path": chat_button_path,
            "chat_template": chat_template,
            "origin_h": origin_h,
            "origin_w": origin_w,
            "match_maps": match_maps,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self.debug and getattr(self, "_debug_win_ready", False):
            try:
                cv2.destroyWindow(self.DEBUG_WIN)
            except cv2.error as exc:
                logger.debug("Map locator debug window close failed: %s", exc)
            self._debug_win_ready = False

        self.chat_template = None
        self.last_center = None
        self.smoothed_center = None
        self._match_maps = []
        self.big_match = None

        if self.debug:
            self.debug_map = None
            self.zoom_map = None

    def activate_map_crop_size(self, index: int) -> None:
        previous_size = getattr(self, "map_crop_size", None)
        self._active_map_crop_index = index
        self.map_crop_size = self.MAP_CROP_SIZES[index]
        self.scale, self.big_match = self._match_maps[index]
        if (
            previous_size is not None
            and previous_size != self.map_crop_size
            and self.debug
        ):
            logger.info(
                f"NCC map crop size switched: {previous_size} -> {self.map_crop_size}"
            )

    def last_location_result(self, mode: str, score: float = 0.0) -> MapLocationResult:
        point = None
        if self.smoothed_center is not None:
            point = tuple(int(round(value)) for value in self.smoothed_center)
        elif self.last_center is not None:
            point = self.last_center

        return MapLocationResult(
            found=point is not None,
            point=point,
            raw_point=self.last_center,
            score=score,
            mode=mode,
            polygon=None,
        )

    def locate(self, frame: np.ndarray) -> MapLocationResult:
        if self._closed:
            raise RuntimeError("map locator is closed")

        x, y, w, h = self.MINI_MAP_ROI
        if frame is None or y + h > frame.shape[0] or x + w > frame.shape[1]:
            raise ValueError(
                f"Frame does not contain mini map ROI: {self.MINI_MAP_ROI}"
            )

        minimap = frame[y : y + h, x : x + w]
        # 小地图预处理
        # 1. 初始化 Mask
        template_mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        cv2.circle(template_mask, center, min(w, h) // 2 - self.CIRCLE_PADDING, 255, -1)

        # 2. HSV颜色过滤，去除小图标干扰
        hsv_img = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([179, 66, 80]))

        # 3. 结合颜色掩码和圆形掩码
        combined_mask = cv2.bitwise_and(color_mask, template_mask)

        # 4. 直接取 V 通道作为灰度，并应用掩码
        v_channel = hsv_img[:, :, 2]
        mini_gray = cv2.bitwise_and(v_channel, combined_mask)

        # 5. 大地图灰度 = 小地图灰度 - 3，在这里做严格对齐
        template = cv2.subtract(mini_gray, 3)

        # 检查是否在大世界界面
        button_found, prob, _, _ = match_template_in_region(
            frame,
            self.BUTTON_ROI,
            self.chat_template,
            min_similarity=0.2,
            green_mask=True,
        )
        if not button_found:

            result = self.last_location_result(
                "not_in_world",
                1.0 if self.last_center is not None else 0.0,
            )
            if self.debug:
                self.show_debug(template, result)
            return result

        # 多尺度匹配
        selected_index, result = self.match_template_all_scales(
            template,
            template_mask,
            w,
            h,
        )
        if result.found:
            self.activate_map_crop_size(selected_index)

        result = self.recover_from_teleport(
            template,
            template_mask,
            w,
            h,
            result,
        )

        raw_point = result.raw_point  # 原始坐标
        polygon = result.polygon  # 匹配区域的四边形顶点坐标
        score = result.score  # 置信度
        mode = result.mode  # 匹配模式（local/global/rejected）

        if raw_point is None:
            self.last_center = None
            self.smoothed_center = None
            point = None
        elif self.smoothed_center is None or mode == "global_teleport":
            self.smoothed_center = tuple(float(value) for value in raw_point)
            point = raw_point
        else:
            # EMA平滑坐标，减少抖动
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

    def match_template_all_scales(
        self,
        template: np.ndarray,
        template_mask: np.ndarray,
        w: int,
        h: int,
        *,
        force_global: bool = False,
    ) -> tuple[int, MapLocationResult]:
        matches = [
            self.match_template(
                template,
                template_mask,
                w,
                h,
                index,
                force_global=force_global,
            )
            for index in range(len(self.MAP_CROP_SIZES))
        ]
        valid_matches = [
            (index, match) for index, match in enumerate(matches) if match.found
        ]
        if valid_matches:
            return max(
                valid_matches,
                key=lambda item: (
                    item[1].score,
                    item[0] == self._active_map_crop_index,
                ),
            )
        return self._active_map_crop_index, matches[self._active_map_crop_index]

    def recover_from_teleport(
        self,
        template: np.ndarray,
        template_mask: np.ndarray,
        w: int,
        h: int,
        result: MapLocationResult,
    ) -> MapLocationResult:
        if self.last_center is None:
            return result

        should_recover = not result.found or result.raw_point is None
        if result.mode == "local" and result.raw_point is not None:
            distance = math.hypot(
                result.raw_point[0] - self.last_center[0],
                result.raw_point[1] - self.last_center[1],
            )
            should_recover = should_recover or distance >= self.TELEPORT_DISTANCE

        if not should_recover:
            return result

        global_index, global_result = self.match_template_all_scales(
            template,
            template_mask,
            w,
            h,
            force_global=True,
        )
        if not global_result.found or global_result.raw_point is None:
            return result

        self.activate_map_crop_size(global_index)
        return MapLocationResult(
            found=global_result.found,
            point=global_result.point,
            raw_point=global_result.raw_point,
            score=global_result.score,
            mode="global_teleport",
            polygon=global_result.polygon,
        )

    def match_template(
        self,
        template: np.ndarray,
        template_mask: np.ndarray,
        w: int,
        h: int,
        map_crop_index: int,
        *,
        force_global: bool = False,
    ) -> MapLocationResult:
        scale, big_match = self._match_maps[map_crop_index]
        raw_point = None
        polygon = None
        score = 0.0
        mode = "rejected"

        if np.count_nonzero(template) >= self.MIN_FILTER_PIXELS:
            min_score = self.GLOBAL_MIN_SCORE
            mode = "global"

            if self.last_center is not None and not force_global:
                # 邻域搜索
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
                # 全局搜索
                best_score_g = -1.0
                best_loc = None
                best_offset_x, best_offset_y = 0, 0
                for ry0, ry1, rx0, rx1 in self.GLOBAL_SEARCH_REGIONS:
                    # 将全局搜索区域坐标根据当前缩放调整到匹配图尺寸
                    mry0 = int(round(ry0 * scale))
                    mry1 = int(round(ry1 * scale))
                    mrx0 = int(round(rx0 * scale))
                    mrx1 = int(round(rx1 * scale))
                    region = big_match[mry0:mry1, mrx0:mrx1]
                    if region.shape[0] < h or region.shape[1] < w:
                        logger.warning(
                            f"Global search region ({rx0},{ry0})-({rx1},{ry1}) skipped: "
                            f"region too small ({region.shape[1]}x{region.shape[0]}) for template ({w}x{h})"
                        )
                        continue
                    # 核心匹配计算
                    response = cv2.matchTemplate(
                        region, template, cv2.TM_CCORR_NORMED, mask=template_mask
                    )
                    response = np.nan_to_num(
                        response, nan=-1.0, posinf=-1.0, neginf=-1.0
                    )
                    response[(response < -1e-6) | (response > 1.0 + 1e-6)] = -1.0
                    np.clip(response, 0.0, 1.0, out=response)
                    _, s, _, loc = cv2.minMaxLoc(response)

                    if s > best_score_g:
                        best_score_g = s
                        best_loc = loc
                        best_offset_x, best_offset_y = mrx0, mry0
                score = best_score_g

            if best_loc is not None and score >= min_score:
                # 坐标映射
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

    def setup_debug_window(self) -> None:
        """创建带滑块和鼠标回调的 OpenCV 调试窗口。"""
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
            self.setup_debug_window()

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

        zoom_side = self.DEBUG_ZOOM_SIZE
        r = self._zoom_radius
        if result.point is not None:
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

        zs = self.zoom_map_scale
        cx0 = int(max(0, z_x0 * zs))
        cy0 = int(max(0, z_y0 * zs))
        cx1 = int(min(self.zoom_map.shape[1], z_x1 * zs))
        cy1 = int(min(self.zoom_map.shape[0], z_y1 * zs))

        if cx1 > cx0 and cy1 > cy0:
            zoom_crop = self.zoom_map[cy0:cy1, cx0:cx1].copy()
            if result.polygon is not None:
                poly_shifted = (result.polygon - np.float32([[[z_x0, z_y0]]])) * zs
                cv2.polylines(zoom_crop, [np.int32(poly_shifted)], True, (0, 255, 0), 2)

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
        pump_debug_windows()
