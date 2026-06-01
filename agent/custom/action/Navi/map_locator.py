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
    keypoints: int
    matches: int
    inliers: int
    polygon: np.ndarray | None = None


class MapLocator:
    def __init__(
        self,
        big_map_path: Path | None = None,
        mini_map_roi: list[int] | None = None,
        debug: bool = False,
        nfeatures: int = 0,
        ratio_thresh: float = 0.8,
        min_matches: int = 8,
        min_inliers: int = 4,
        ransac_thresh: float = 12.0,
        circle_padding: int = 15,
        center_radius: int = 11,
        debug_map_width: int = 900,
        max_processing_long_side: int = 6144,
    ):
        self.big_map_path = (
            Path(big_map_path) if big_map_path else self.default_big_map_path()
        )
        self.mini_map_roi = mini_map_roi or [28, 20, 150, 150]
        self.debug = debug
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.circle_padding = circle_padding
        self.center_radius = center_radius
        self.debug_map_width = debug_map_width
        self.max_processing_long_side = max_processing_long_side

        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.last_center: tuple[int, int] | None = None
        self.pending_center: tuple[int, int] | None = None
        self.pending_count = 0

        self._load_big_map()

    @staticmethod
    def default_big_map_path() -> Path:
        return resource_base_path() / "image/map/map.jpg"

    def locate(self, frame: np.ndarray) -> MapLocationResult:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        masked = self._extract_masked_minimap(frame)
        mh, mw = masked.shape[:2]
        mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        kp_mini, des_mini = self.sift.detectAndCompute(mini_gray, None)

        polygon = None
        player_point = None
        raw_player_point = None
        good_matches = []
        inliers = 0
        keypoints = 0 if kp_mini is None else len(kp_mini)

        if des_mini is not None and keypoints >= self.min_matches:
            knn_matches = self.matcher.knnMatch(des_mini, self.des_big, k=2)
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                src_pts = np.float32(
                    [kp_mini[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [self.big_points[m.trainIdx] for m in good_matches]
                ).reshape(-1, 1, 2)

                transform, mask = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_thresh,
                )
                if transform is not None and mask is not None:
                    inliers = int(mask.sum())
                    if inliers >= self.min_inliers:
                        corners = np.float32(
                            [[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]]
                        ).reshape(-1, 1, 2)
                        polygon = cv2.transform(corners, transform)

                        player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(
                            -1, 1, 2
                        )
                        player_dst = cv2.transform(player_src, transform)[0, 0]
                        player_point = (
                            int(player_dst[0] / self.big_map_scale),
                            int(player_dst[1] / self.big_map_scale),
                        )
                        raw_player_point = player_point

        player_point, polygon = self._filter_point(
            player_point, polygon, inliers, len(good_matches)
        )
        if player_point is None:
            player_point = self.last_center

        result = MapLocationResult(
            found=player_point is not None,
            point=player_point,
            raw_point=raw_player_point,
            keypoints=keypoints,
            matches=len(good_matches),
            inliers=inliers,
            polygon=polygon,
        )

        if self.debug:
            self.show_debug(masked, result)

        return result

    def show_debug(self, masked_minimap: np.ndarray, result: MapLocationResult) -> None:
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
            f"kp={result.keypoints} matches={result.matches} inliers={result.inliers}",
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

        mini_view = cv2.resize(
            masked_minimap, (280, 280), interpolation=cv2.INTER_NEAREST
        )
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
                [
                    mini_view,
                    np.zeros(
                        (canvas_height - mini_view.shape[0], mini_view.shape[1], 3),
                        dtype=np.uint8,
                    ),
                ],
                axis=0,
            )
        if map_view.shape[0] < canvas_height:
            map_view = np.concatenate(
                [
                    map_view,
                    np.zeros(
                        (canvas_height - map_view.shape[0], map_view.shape[1], 3),
                        dtype=np.uint8,
                    ),
                ],
                axis=0,
            )

        cv2.imshow("Map Locator", np.concatenate([mini_view, map_view], axis=1))

    def close_debug(self) -> None:
        if self.debug:
            cv2.destroyWindow("Map Locator")

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
                (
                    int(self.origin_w * self.big_map_scale),
                    int(self.origin_h * self.big_map_scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        self.big_map = cv2.convertScaleAbs(big_map, alpha=2.5, beta=-20)
        big_gray = cv2.cvtColor(self.big_map, cv2.COLOR_BGR2GRAY)
        cache_path = self.big_map_path.with_name(
            f"{self.big_map_path.stem}.sift_cache.npz"
        )
        cache_meta = {
            "map_size": int(self.big_map_path.stat().st_size),
            "map_mtime_ns": self.big_map_path.stat().st_mtime_ns,
            "origin_w": self.origin_w,
            "origin_h": self.origin_h,
            "proc_w": self.big_map.shape[1],
            "proc_h": self.big_map.shape[0],
            "scale": self.big_map_scale,
        }

        self.big_points = None
        self.des_big = None
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    cache_meta_raw = cache["meta"]
                    saved_meta = json.loads(
                        cache_meta_raw.item()
                        if hasattr(cache_meta_raw, "item")
                        else str(cache_meta_raw)
                    )
                    if saved_meta == cache_meta:
                        self.big_points = cache["keypoints"].astype(
                            np.float32, copy=False
                        )
                        self.des_big = cache["descriptors"]
            except Exception as exc:
                logger.error(f"Failed to read feature cache: {exc}")

        if self.big_points is None or self.des_big is None:
            kp_big, self.des_big = self.sift.detectAndCompute(big_gray, None)
            if self.des_big is not None:
                self.big_points = np.float32([kp.pt for kp in kp_big])
                try:
                    np.savez_compressed(
                        cache_path,
                        meta=json.dumps(cache_meta, ensure_ascii=False),
                        keypoints=self.big_points,
                        descriptors=self.des_big,
                    )
                except Exception as exc:
                    logger.error(f"Failed to save feature cache: {exc}")

        if (
            self.des_big is None
            or self.big_points is None
            or len(self.big_points) < self.min_matches
        ):
            raise ValueError("Big map does not have enough feature points")

        logger.debug(f"Big map keypoints: {len(self.big_points)}")

    def _extract_masked_minimap(self, frame: np.ndarray) -> np.ndarray:
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
        cv2.circle(masked, center, self.center_radius, (0, 0, 0), -1)
        return masked

    def _filter_point(
        self,
        player_point: tuple[int, int] | None,
        polygon: np.ndarray | None,
        inliers: int,
        matches: int,
    ) -> tuple[tuple[int, int] | None, np.ndarray | None]:
        if player_point is None:
            return None, polygon

        accept_point = True
        px, py = player_point
        if px < 0 or py < 0 or px >= self.origin_w or py >= self.origin_h:
            accept_point = False

        if accept_point and self.last_center is not None:
            jump = math.hypot(
                player_point[0] - self.last_center[0],
                player_point[1] - self.last_center[1],
            )
            max_jump = 90 if inliers >= 8 or matches >= 14 else 60

            if jump > max_jump:
                if self.pending_center is not None:
                    pending_jump = math.hypot(
                        player_point[0] - self.pending_center[0],
                        player_point[1] - self.pending_center[1],
                    )
                    if pending_jump <= 25:
                        self.pending_count += 1
                    else:
                        self.pending_center = player_point
                        self.pending_count = 1
                else:
                    self.pending_center = player_point
                    self.pending_count = 1

                if self.pending_count < 2:
                    logger.debug(
                        f"Reject jump raw={player_point} last={self.last_center} jump={jump:.1f} inliers={inliers}"
                    )
                    accept_point = False
                else:
                    logger.debug(
                        f"Accept delayed jump raw={player_point} last={self.last_center} jump={jump:.1f}"
                    )
                    self.pending_center = None
                    self.pending_count = 0
            else:
                self.pending_center = None
                self.pending_count = 0

        if not accept_point:
            return self.last_center, None

        if self.last_center is not None:
            player_point = (
                int(self.last_center[0] * 0.7 + player_point[0] * 0.3),
                int(self.last_center[1] * 0.7 + player_point[1] * 0.3),
            )
        self.last_center = player_point
        return player_point, polygon


@AgentServer.custom_action("map_locator")
class MapLocatorTestAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = _load_params(argv.custom_action_param)
        debug = bool(params.get("debug", False))
        frame_interval = float(params.get("frame_interval", 0.1))

        try:
            locator = MapLocator(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                mini_map_roi=params.get("mini_map_roi"),
                debug=debug,
                nfeatures=int(params.get("nfeatures", 0)),
                ratio_thresh=float(params.get("ratio_thresh", 0.8)),
                min_matches=int(params.get("min_matches", 8)),
                min_inliers=int(params.get("min_inliers", 4)),
                ransac_thresh=float(params.get("ransac_thresh", 12.0)),
                circle_padding=int(params.get("circle_padding", 15)),
                center_radius=int(params.get("center_radius", 11)),
                debug_map_width=int(params.get("debug_map_width", 900)),
                max_processing_long_side=int(
                    params.get("max_processing_long_side", 6144)
                ),
            )
        except Exception as exc:
            logger.error(f"Map locator init failed: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(f"Map locator started: map={locator.big_map_path}, debug={debug}")
        controller = context.tasker.controller
        last_result = MapLocationResult(False, None, None, 0, 0, 0)

        try:
            while not context.tasker.stopping:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    continue

                last_result = locator.locate(frame)
                if last_result.point is not None:
                    logger.info(
                        f"Map location result: point={last_result.point}, "
                        f"matches={last_result.matches}, inliers={last_result.inliers}"
                    )
                else:
                    logger.info(
                        f"Map location result: not found, "
                        f"keypoints={last_result.keypoints}, matches={last_result.matches}, inliers={last_result.inliers}"
                    )

                if debug and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                sleep_time = frame_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            logger.error(f"Map locator failed: {exc}")
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
