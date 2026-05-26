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
    def __init__(
        self,
        big_map_path: Path | None = None,
        mini_map_roi: list[int] | None = None,
        debug: bool = False,
        min_score: float = 0.52,
        local_min_score: float = 0.46,
        local_trust_score: float = 0.90,
        global_check_interval: int = 4,
        relocate_score: float = 0.62,
        relocate_distance: int = 350,
        relocate_confirm_frames: int = 2,
        template_scales: list[float] | None = None,
        global_search_long_side: int = 1600,
        global_refine_radius: int = 520,
        search_radius: int = 520,
        min_mask_pixels: int = 650,
        circle_padding: int = 15,
        center_radius: int = 11,
        debug_map_width: int = 900,
        max_processing_long_side: int = 6144,
        lost_reset_frames: int = 3,
    ):
        self.big_map_path = Path(big_map_path) if big_map_path else self.default_big_map_path()
        self.mini_map_roi = mini_map_roi or [24, 14, 159, 157]
        self.debug = debug
        self.min_score = min_score
        self.local_min_score = local_min_score
        self.local_trust_score = local_trust_score
        self.global_check_interval = max(1, global_check_interval)
        self.relocate_score = relocate_score
        self.relocate_distance = relocate_distance
        self.relocate_confirm_frames = max(1, relocate_confirm_frames)
        self.template_scales = template_scales or [0.96, 1.0, 1.04]
        self.global_search_long_side = global_search_long_side
        self.global_refine_radius = global_refine_radius
        self.search_radius = search_radius
        self.min_mask_pixels = min_mask_pixels
        self.circle_padding = circle_padding
        self.center_radius = center_radius
        self.debug_map_width = debug_map_width
        self.max_processing_long_side = max_processing_long_side
        self.lost_reset_frames = lost_reset_frames

        self.last_center: tuple[int, int] | None = None
        self.pending_center: tuple[int, int] | None = None
        self.pending_count = 0
        self.lost_frames = 0
        self.frame_index = 0

        self._load_big_map()

    @staticmethod
    def default_big_map_path() -> Path:
        abs_path = Path(__file__).parents[3]
        if (abs_path / "assets").exists():
            return abs_path / "assets/resource/base/image/map/map.jpg"
        return abs_path / "resource/base/image/map/map.jpg"

    def locate(self, frame: np.ndarray) -> MapLocationNccResult:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        masked, template_mask = self._extract_masked_minimap(frame)
        mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        mini_match = self._build_match_image(mini_gray, template_mask)

        polygon = None
        player_point = None
        raw_player_point = None
        mode = "lost"
        score = 0.0

        self.frame_index += 1
        if int(np.count_nonzero(template_mask)) >= self.min_mask_pixels:
            raw_player_point, polygon, score, mode = self._match_minimap(mini_gray, mini_match, template_mask)
            player_point = raw_player_point

        player_point, polygon = self._filter_point(player_point, polygon, score, mode)
        if raw_player_point is None:
            self.lost_frames += 1
            if self.lost_frames > self.lost_reset_frames:
                self.last_center = None
        else:
            self.lost_frames = 0

        if player_point is None:
            player_point = self.last_center

        result = MapLocationNccResult(
            found=player_point is not None,
            point=player_point,
            raw_point=raw_player_point,
            score=score,
            mode=mode,
            polygon=polygon,
        )

        if self.debug:
            self.show_debug(masked, result)

        return result

    def show_debug(self, masked_minimap: np.ndarray, result: MapLocationNccResult) -> None:
        map_view = self.big_map.copy()
        if result.polygon is not None:
            cv2.polylines(map_view, [np.int32(result.polygon)], True, (0, 255, 0), 3)
        if result.point is not None:
            draw_player_point = (
                int(result.point[0] * self.big_map_scale),
                int(result.point[1] * self.big_map_scale),
            )
            cv2.circle(map_view, draw_player_point, 16, (0, 0, 255), -1)

        cv2.putText(
            map_view,
            f"ncc={result.score:.3f} mode={result.mode}",
            (12, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if result.point is not None:
            cv2.putText(
                map_view,
                f"coordinate=({result.point[0]}, {result.point[1]})",
                (12, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if map_view.shape[1] > self.debug_map_width:
            scale = self.debug_map_width / map_view.shape[1]
            map_view = cv2.resize(
                map_view,
                (self.debug_map_width, int(map_view.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        mini_view = cv2.resize(masked_minimap, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            mini_view,
            "mini map",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        canvas_height = max(mini_view.shape[0], map_view.shape[0])
        if mini_view.shape[0] < canvas_height:
            mini_view = np.concatenate(
                [mini_view, np.zeros((canvas_height - mini_view.shape[0], mini_view.shape[1], 3), dtype=np.uint8)],
                axis=0,
            )
        if map_view.shape[0] < canvas_height:
            map_view = np.concatenate(
                [map_view, np.zeros((canvas_height - map_view.shape[0], map_view.shape[1], 3), dtype=np.uint8)],
                axis=0,
            )

        cv2.imshow("Map Locator NCC", np.concatenate([mini_view, map_view], axis=1))

    def close_debug(self) -> None:
        if self.debug:
            cv2.destroyWindow("Map Locator NCC")

    def _load_big_map(self) -> None:
        if not self.big_map_path.exists():
            raise FileNotFoundError(f"Big map not found: {self.big_map_path}")

        big_map = cv2.imread(str(self.big_map_path), cv2.IMREAD_COLOR)
        if big_map is None:
            raise ValueError(f"Failed to read big map: {self.big_map_path}")

        self.origin_h, self.origin_w = big_map.shape[:2]
        self.big_map_scale = min(
            1.0,
            self.max_processing_long_side / max(self.origin_h, self.origin_w),
        )
        if self.big_map_scale < 1.0:
            big_map = cv2.resize(
                big_map,
                (int(self.origin_w * self.big_map_scale), int(self.origin_h * self.big_map_scale)),
                interpolation=cv2.INTER_AREA,
            )

        self.big_map = cv2.convertScaleAbs(big_map, alpha=2.5, beta=-20)
        big_match_base = cv2.convertScaleAbs(big_map, alpha=3.8, beta=-40)
        self.big_gray = cv2.cvtColor(big_match_base, cv2.COLOR_BGR2GRAY)
        self.big_match = self._build_match_image(self.big_gray)
        self.global_search_scale = min(
            1.0,
            self.global_search_long_side / max(self.big_gray.shape[:2]),
        )
        if self.global_search_scale < 1.0:
            global_size = (
                max(1, int(self.big_gray.shape[1] * self.global_search_scale)),
                max(1, int(self.big_gray.shape[0] * self.global_search_scale)),
            )
            self.global_gray = cv2.resize(self.big_gray, global_size, interpolation=cv2.INTER_AREA)
            self.global_match = cv2.resize(self.big_match, global_size, interpolation=cv2.INTER_AREA)
        else:
            self.global_gray = self.big_gray
            self.global_match = self.big_match
        logger.debug(f"NCC big map loaded: {self.big_map.shape[1]}x{self.big_map.shape[0]}")

    def _extract_masked_minimap(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, w, h = self.mini_map_roi
        minimap = frame[y : y + h, x : x + w]
        masked = minimap.copy()
        mh, mw = masked.shape[:2]

        center = (mw // 2, mh // 2)
        radius = max(1, min(mw, mh) // 2 - self.circle_padding)
        circle_mask = np.zeros((mh, mw), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)

        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        upper_hsv = np.array([179, 66, 80], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        final_mask = cv2.bitwise_and(circle_mask, hsv_mask)
        masked = cv2.bitwise_and(masked, masked, mask=final_mask)
        masked = cv2.convertScaleAbs(masked, alpha=3.8, beta=-40)

        masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        detail_mask = cv2.inRange(masked_gray, 25, 255)
        template_mask = cv2.bitwise_and(final_mask, detail_mask)
        cv2.circle(template_mask, center, self.center_radius + 5, 0, -1)
        template_mask = cv2.erode(template_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        cv2.circle(masked, center, self.center_radius, (0, 0, 0), -1)
        return masked, template_mask

    def _build_match_image(self, gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        work = gray
        if mask is not None:
            work = cv2.bitwise_and(work, work, mask=mask)

        work = cv2.GaussianBlur(work, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(work)
        edges = cv2.Canny(equalized, 35, 110)
        edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
        feature = cv2.addWeighted(equalized, 0.35, edges, 0.65, 0)

        if mask is not None:
            feature = cv2.bitwise_and(feature, feature, mask=mask)
        return feature

    def _match_minimap(
        self,
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None, float, str]:
        local_result = None
        if self.last_center is not None:
            local_result = self._match_in_local_region(mini_gray, mini_match, template_mask)
            point, polygon, score = local_result
            should_probe_global = (
                score < self.local_trust_score
                or self.frame_index % self.global_check_interval == 0
            )
            if point is not None and score >= self.local_min_score and not should_probe_global:
                return point, polygon, score, "local"

        global_result = self._match_global_region(mini_gray, mini_match, template_mask)
        global_point, global_polygon, global_score = global_result

        if self.last_center is None:
            if global_point is not None and global_score >= self.min_score:
                return global_point, global_polygon, global_score, "global"
            return None, None, global_score, "rejected"

        if global_point is not None and global_score >= self.relocate_score:
            jump = math.hypot(global_point[0] - self.last_center[0], global_point[1] - self.last_center[1])
            if jump >= self.relocate_distance:
                return global_point, global_polygon, global_score, "global_relocate"

        if local_result is not None:
            local_point, local_polygon, local_score = local_result
            if local_point is not None and local_score >= self.local_min_score:
                return local_point, local_polygon, local_score, "local"

        if global_point is not None and global_score >= self.min_score:
            return global_point, global_polygon, global_score, "global"
        return None, None, max(global_score, local_result[2] if local_result is not None else 0.0), "rejected"

    def _match_in_local_region(
        self,
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None, float]:
        th, tw = mini_match.shape[:2]
        map_h, map_w = self.big_match.shape[:2]
        center_x = self.last_center[0] * self.big_map_scale
        center_y = self.last_center[1] * self.big_map_scale
        radius = self.search_radius * self.big_map_scale

        x0 = max(0, int(math.floor(center_x - radius - tw * 0.5)))
        y0 = max(0, int(math.floor(center_y - radius - th * 0.5)))
        x1 = min(map_w, int(math.ceil(center_x + radius + tw * 0.5)))
        y1 = min(map_h, int(math.ceil(center_y + radius + th * 0.5)))

        if x1 - x0 < tw or y1 - y0 < th:
            return None, None, 0.0

        search_gray = self.big_gray[y0:y1, x0:x1]
        search_match = self.big_match[y0:y1, x0:x1]
        return self._match_in_region(
            mini_gray,
            mini_match,
            template_mask,
            x0,
            y0,
            search_gray,
            search_match,
            self.big_map_scale,
        )

    def _match_global_region(
        self,
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None, float]:
        coarse_result = self._match_in_region(
            mini_gray,
            mini_match,
            template_mask,
            0,
            0,
            self.global_gray,
            self.global_match,
            self.big_map_scale * self.global_search_scale,
            template_base_scale=self.global_search_scale,
            polygon_display_scale=1.0 / self.global_search_scale,
        )
        coarse_point, _, coarse_score = coarse_result
        if coarse_point is None:
            return coarse_result

        refined_result = self._match_around_point(
            coarse_point,
            mini_gray,
            mini_match,
            template_mask,
            self.global_refine_radius,
        )
        refined_point, _, refined_score = refined_result
        if refined_point is not None:
            return refined_result
        return coarse_result

    def _match_around_point(
        self,
        center_point: tuple[int, int],
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
        radius: int,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None, float]:
        th, tw = mini_match.shape[:2]
        map_h, map_w = self.big_match.shape[:2]
        center_x = center_point[0] * self.big_map_scale
        center_y = center_point[1] * self.big_map_scale
        scaled_radius = radius * self.big_map_scale

        x0 = max(0, int(math.floor(center_x - scaled_radius - tw * 0.5)))
        y0 = max(0, int(math.floor(center_y - scaled_radius - th * 0.5)))
        x1 = min(map_w, int(math.ceil(center_x + scaled_radius + tw * 0.5)))
        y1 = min(map_h, int(math.ceil(center_y + scaled_radius + th * 0.5)))

        if x1 - x0 < tw or y1 - y0 < th:
            return None, None, 0.0

        return self._match_in_region(
            mini_gray,
            mini_match,
            template_mask,
            x0,
            y0,
            self.big_gray[y0:y1, x0:x1],
            self.big_match[y0:y1, x0:x1],
            self.big_map_scale,
        )

    def _match_in_region(
        self,
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        search_gray: np.ndarray,
        search_match: np.ndarray,
        image_to_origin_scale: float,
        template_base_scale: float = 1.0,
        polygon_display_scale: float = 1.0,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None, float]:
        best_point = None
        best_polygon = None
        best_score = -1.0

        for scale in self.template_scales:
            effective_scale = scale * template_base_scale
            scaled_gray, scaled_match, scaled_mask = self._scale_template(
                mini_gray,
                mini_match,
                template_mask,
                effective_scale,
            )
            th, tw = scaled_match.shape[:2]
            if search_match.shape[0] < th or search_match.shape[1] < tw:
                continue
            if int(np.count_nonzero(scaled_mask)) < self.min_mask_pixels:
                continue

            raw_response = cv2.matchTemplate(
                search_gray,
                scaled_gray,
                cv2.TM_CCOEFF_NORMED,
                mask=scaled_mask,
            )
            feature_response = cv2.matchTemplate(
                search_match,
                scaled_match,
                cv2.TM_CCORR_NORMED,
                mask=scaled_mask,
            )
            raw_response = np.nan_to_num(raw_response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            feature_response = np.nan_to_num(feature_response, nan=-1.0, posinf=-1.0, neginf=-1.0)
            response = raw_response * 0.58 + feature_response * 0.42
            _, max_val, _, max_loc = cv2.minMaxLoc(response)

            if float(max_val) <= best_score:
                continue

            x = offset_x + max_loc[0]
            y = offset_y + max_loc[1]
            best_point = (
                int((x + tw * 0.5) / image_to_origin_scale),
                int((y + th * 0.5) / image_to_origin_scale),
            )
            best_polygon = np.float32(
                [
                    [x * polygon_display_scale, y * polygon_display_scale],
                    [(x + tw - 1) * polygon_display_scale, y * polygon_display_scale],
                    [(x + tw - 1) * polygon_display_scale, (y + th - 1) * polygon_display_scale],
                    [x * polygon_display_scale, (y + th - 1) * polygon_display_scale],
                ]
            ).reshape(-1, 1, 2)
            best_score = float(max_val)

        if best_point is None:
            return None, None, 0.0
        return best_point, best_polygon, best_score

    @staticmethod
    def _scale_template(
        mini_gray: np.ndarray,
        mini_match: np.ndarray,
        template_mask: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if abs(scale - 1.0) < 1e-3:
            return mini_gray, mini_match, template_mask

        h, w = mini_gray.shape[:2]
        scaled_size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
        scaled_gray = cv2.resize(mini_gray, scaled_size, interpolation=cv2.INTER_AREA)
        scaled_match = cv2.resize(mini_match, scaled_size, interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(template_mask, scaled_size, interpolation=cv2.INTER_NEAREST)
        return scaled_gray, scaled_match, scaled_mask

    def _filter_point(
        self,
        player_point: tuple[int, int] | None,
        polygon: np.ndarray | None,
        score: float,
        mode: str,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None]:
        if player_point is None:
            return None, polygon

        accept_point = True
        px, py = player_point
        if px < 0 or py < 0 or px >= self.origin_w or py >= self.origin_h:
            accept_point = False

        if accept_point and self.last_center is not None:
            jump = math.hypot(player_point[0] - self.last_center[0], player_point[1] - self.last_center[1])
            max_jump = 100 if score >= self.min_score + 0.08 else 65
            is_relocate = mode == "global_relocate"

            if jump > max_jump:
                if self.pending_center is not None:
                    pending_jump = math.hypot(
                        player_point[0] - self.pending_center[0],
                        player_point[1] - self.pending_center[1],
                    )
                    if pending_jump <= 30:
                        self.pending_count += 1
                    else:
                        self.pending_center = player_point
                        self.pending_count = 1
                else:
                    self.pending_center = player_point
                    self.pending_count = 1

                required_count = self.relocate_confirm_frames if is_relocate else 2
                if is_relocate and score >= self.relocate_score + 0.08:
                    required_count = 1

                if self.pending_count < required_count:
                    logger.debug(
                        f"Reject NCC jump raw={player_point} last={self.last_center} jump={jump:.1f} score={score:.3f}"
                    )
                    accept_point = False
                else:
                    logger.debug(f"Accept delayed NCC jump raw={player_point} last={self.last_center} jump={jump:.1f}")
                    self.pending_center = None
                    self.pending_count = 0
            else:
                self.pending_center = None
                self.pending_count = 0

        if not accept_point:
            return self.last_center, None

        if mode == "global_relocate":
            self.last_center = player_point
            return player_point, polygon

        if self.last_center is not None:
            player_point = (
                int(self.last_center[0] * 0.7 + player_point[0] * 0.3),
                int(self.last_center[1] * 0.7 + player_point[1] * 0.3),
            )
        self.last_center = player_point
        return player_point, polygon


@AgentServer.custom_action("map_locator_ncc")
class MapLocatorNccTestAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        params = _load_params(argv.custom_action_param)
        debug = bool(params.get("debug", False))
        frame_interval = float(params.get("frame_interval", 0.1))

        try:
            locator = MapLocatorNcc(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                mini_map_roi=params.get("mini_map_roi"),
                debug=debug,
                min_score=float(params.get("min_score", 0.52)),
                local_min_score=float(params.get("local_min_score", 0.46)),
                local_trust_score=float(params.get("local_trust_score", 0.90)),
                global_check_interval=int(params.get("global_check_interval", 4)),
                relocate_score=float(params.get("relocate_score", 0.62)),
                relocate_distance=int(params.get("relocate_distance", 350)),
                relocate_confirm_frames=int(params.get("relocate_confirm_frames", 2)),
                template_scales=_parse_float_list(params.get("template_scales"), [0.96, 1.0, 1.04]),
                global_search_long_side=int(params.get("global_search_long_side", 1600)),
                global_refine_radius=int(params.get("global_refine_radius", 520)),
                search_radius=int(params.get("search_radius", 520)),
                min_mask_pixels=int(params.get("min_mask_pixels", 650)),
                circle_padding=int(params.get("circle_padding", 15)),
                center_radius=int(params.get("center_radius", 11)),
                debug_map_width=int(params.get("debug_map_width", 900)),
                max_processing_long_side=int(params.get("max_processing_long_side", 6144)),
                lost_reset_frames=int(params.get("lost_reset_frames", 3)),
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
                if last_result.point is not None:
                    logger.info(
                        f"Map NCC location result: point={last_result.point}, "
                        f"score={last_result.score:.3f}, mode={last_result.mode}"
                    )
                else:
                    logger.info(
                        f"Map NCC location result: not found, "
                        f"score={last_result.score:.3f}, mode={last_result.mode}"
                    )

                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                sleep_time = frame_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Map locator NCC failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            if debug:
                cv2.destroyAllWindows()

        return CustomAction.RunResult(success=last_result.found)


def _load_params(custom_action_param) -> dict:
    if not custom_action_param:
        return {}
    if isinstance(custom_action_param, dict):
        return custom_action_param
    try:
        params = json.loads(custom_action_param)
        return params if isinstance(params, dict) else {}
    except Exception as exc:
        logger.warning(f"Parse custom_action_param failed, use defaults: {exc}")
        return {}


def _parse_float_list(value, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        parsed = [float(item) for item in value if float(item) > 0]
        return parsed or default
    if isinstance(value, str):
        try:
            raw_items = json.loads(value)
            if isinstance(raw_items, (list, tuple)):
                parsed = [float(item) for item in raw_items if float(item) > 0]
                return parsed or default
        except Exception:
            pass

        parsed = [float(item.strip()) for item in value.split(",") if item.strip() and float(item.strip()) > 0]
        return parsed or default
    return default
