from __future__ import annotations

import cv2
import json
import math
import time
import numpy as np

from pathlib import Path
from .Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = get_logger(__name__)


class _LightGlueBackend:
    def __init__(self, max_num_keypoints: int = 1024, device: str = "auto") -> None:
        self.max_num_keypoints = max_num_keypoints
        self.device_request = device
        self.ready = False
        self.device = None
        self.torch = None
        self.extractor = None
        self.matcher = None
        self._error_message = ""

    def ensure_ready(self) -> bool:
        if self.ready:
            return True

        try:
            import torch
        except Exception as exc:
            self._error_message = (
                "缺少 PyTorch，无法启用 LightGlue。请先安装 torch。"
                f" 原始错误: {exc}"
            )
            logger.error(self._error_message)
            return False

        try:
            from lightglue import LightGlue, SuperPoint
        except Exception as exc:
            self._error_message = (
                "缺少 LightGlue 依赖，无法启用新定位器。"
                " 请先安装 lightglue。"
                f" 原始错误: {exc}"
            )
            logger.error(self._error_message)
            return False

        requested_device = (self.device_request or "auto").strip().lower()
        if requested_device == "auto":
            if torch.cuda.is_available():
                requested_device = "cuda"
            else:
                requested_device = "cpu"

        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("请求使用 CUDA，但当前不可用，已回退到 CPU")
            requested_device = "cpu"

        self.torch = torch
        self.device = torch.device(requested_device)

        try:
            self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        except Exception as exc:
            self._error_message = f"初始化 LightGlue 失败: {exc}"
            logger.error(self._error_message)
            return False

        self.ready = True
        logger.info(f"LightGlue 后端已就绪: device={self.device.type}, max_num_keypoints={self.max_num_keypoints}")
        return True

    @property
    def error_message(self) -> str:
        return self._error_message

    def extract(self, image_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        if not self.ensure_ready():
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.torch.from_numpy(np.ascontiguousarray(image_rgb)).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        with self.torch.inference_mode():
            features = self.extractor.extract(image_tensor)

        return self._to_numpy_dict(features)

    def match(self, feats0: dict[str, np.ndarray], feats1: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not self.ensure_ready():
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

        with self.torch.inference_mode():
            result = self.matcher(
                {
                    "image0": self._to_torch_dict(feats0),
                    "image1": self._to_torch_dict(feats1),
                }
            )

        return self._decode_matches(result)

    def _to_numpy_dict(self, features: dict) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key in ("keypoints", "descriptors", "keypoint_scores", "image_size"):
            value = features.get(key)
            if value is None:
                continue

            if hasattr(value, "detach"):
                array = value.detach().cpu().numpy()
            else:
                array = np.asarray(value)

            if array.ndim >= 1 and array.shape[0] == 1:
                array = array[0]

            if key == "image_size":
                result[key] = np.ascontiguousarray(array.astype(np.int32, copy=False))
            else:
                result[key] = np.ascontiguousarray(array.astype(np.float32, copy=False))
        return result

    def _to_torch_dict(self, features: dict[str, np.ndarray]) -> dict:
        result = {}
        for key, value in features.items():
            tensor = self.torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            if tensor.ndim >= 1:
                tensor = tensor.unsqueeze(0)
            result[key] = tensor
        return result

    def _decode_matches(self, output: dict) -> tuple[np.ndarray, np.ndarray]:
        if "matches" in output:
            matches = output["matches"]
            scores = output.get("scores")

            if hasattr(matches, "detach"):
                matches = matches.detach().cpu().numpy()
            else:
                matches = np.asarray(matches)

            if scores is None:
                scores = np.ones((matches.shape[1] if matches.ndim == 3 else len(matches),), dtype=np.float32)
            elif hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            else:
                scores = np.asarray(scores)

            if matches.ndim == 3:
                matches = matches[0]
            if scores.ndim >= 2:
                scores = scores[0]

            return matches.astype(np.int32, copy=False), scores.astype(np.float32, copy=False)

        if "matches0" in output:
            matches0 = output["matches0"]
            scores0 = output.get("matching_scores0")

            if hasattr(matches0, "detach"):
                matches0 = matches0.detach().cpu().numpy()
            else:
                matches0 = np.asarray(matches0)
            if matches0.ndim >= 2:
                matches0 = matches0[0]

            valid = matches0 > -1
            idx0 = np.nonzero(valid)[0].astype(np.int32, copy=False)
            idx1 = matches0[valid].astype(np.int32, copy=False)
            matches = np.stack([idx0, idx1], axis=1) if len(idx0) > 0 else np.empty((0, 2), dtype=np.int32)

            if scores0 is None:
                scores = np.ones((len(matches),), dtype=np.float32)
            else:
                if hasattr(scores0, "detach"):
                    scores0 = scores0.detach().cpu().numpy()
                else:
                    scores0 = np.asarray(scores0)
                if scores0.ndim >= 2:
                    scores0 = scores0[0]
                scores = scores0[valid].astype(np.float32, copy=False)

            return matches, scores

        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)


@AgentServer.custom_action("map_locator_lightglue")
class MapLocatorLightGlue(CustomAction):
    abs_path = Path(__file__).parents[3]
    map_name = "map.jpg"
    if Path.exists(abs_path / "assets"):
        default_big_map = abs_path / f"assets/resource/base/image/map/{map_name}"
    else:
        default_big_map = abs_path / f"resource/base/image/map/{map_name}"

    def __init__(self) -> None:
        super().__init__()
        self._backend_cache: dict[tuple[int, str], _LightGlueBackend] = {}

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger.info("=== Map Locator LightGlue Started ===")
        controller = context.tasker.controller

        params = self._parse_params(argv.custom_action_param)
        big_map_path = self.default_big_map
        mini_map_roi = params.get("mini_map_roi", [24, 14, 159, 157])
        frame_interval = float(params.get("frame_interval", 0.1))
        min_matches = int(params.get("min_matches", 10))
        min_inliers = int(params.get("min_inliers", 5))
        ransac_thresh = float(params.get("ransac_thresh", 12.0))
        circle_padding = int(params.get("circle_padding", 15))
        center_radius = int(params.get("center_radius", 11))
        debug_map_width = int(params.get("debug_map_width", 900))
        max_processing_long_side = int(params.get("max_processing_long_side", 6144))
        chunk_size = int(params.get("chunk_size", 2800))
        chunk_overlap = int(params.get("chunk_overlap", 200))
        max_num_keypoints = int(params.get("max_num_keypoints", 1024))
        device = str(params.get("device", "auto"))

        backend = self._get_backend(max_num_keypoints=max_num_keypoints, device=device)
        if not backend.ensure_ready():
            logger.error("LightGlue 后端初始化失败，定位流程结束")
            return CustomAction.RunResult(success=False)

        if not big_map_path.exists():
            logger.error(f"大地图不存在: {big_map_path}")
            return CustomAction.RunResult(success=False)

        logger.info(f"正在加载原始大图: {big_map_path}")
        original_map = cv2.imread(str(self.default_big_map), cv2.IMREAD_COLOR)
        if original_map is None:
            logger.error(f"大地图读取失败: {big_map_path}")
            return CustomAction.RunResult(success=False)

        origin_h, origin_w = original_map.shape[:2]
        logger.info(f"原始分辨率: {origin_w}x{origin_h}")

        global_scale = min(1.0, max_processing_long_side / max(origin_h, origin_w))
        global_map = cv2.resize(
            original_map,
            (int(origin_w * global_scale), int(origin_h * global_scale)),
            interpolation=cv2.INTER_AREA,
        )
        global_map = cv2.convertScaleAbs(global_map, alpha=2.5, beta=-20)

        cache_meta = {
            "backend": "lightglue_superpoint",
            "device": backend.device.type if backend.device is not None else device,
            "map_size": int(big_map_path.stat().st_size),
            "map_mtime_ns": big_map_path.stat().st_mtime_ns,
            "origin_w": origin_w,
            "origin_h": origin_h,
            "scale": global_scale,
            "max_num_keypoints": max_num_keypoints,
        }

        cache_path = big_map_path.with_name(f"{big_map_path.stem}.lightglue_global_cache.npz")
        global_features = self._load_feature_cache(cache_path, cache_meta)
        if global_features is None:
            logger.info("正在计算低分辨率全局 LightGlue 特征...")
            global_features = backend.extract(global_map)
            if not self._has_enough_features(global_features, min_matches):
                logger.error("全局视图特征点不足")
                return CustomAction.RunResult(success=False)
            self._save_feature_cache(cache_path, cache_meta, global_features)

        logger.debug(f"全局视图特征点数: {self._feature_count(global_features)}")

        chunk_cols = math.ceil(origin_w / chunk_size)
        chunk_rows = math.ceil(origin_h / chunk_size)
        chunks_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray] | None, int, int]] = {}

        chunks_cache_dir = big_map_path.with_name(f"{big_map_path.stem}_lightglue_chunks_cache")
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始预加载所有 {chunk_cols}x{chunk_rows} 个分块 LightGlue 特征...")
        for r in range(chunk_rows):
            for c in range(chunk_cols):
                raw_cx = c * chunk_size
                raw_cy = r * chunk_size
                cx = max(0, raw_cx - chunk_overlap)
                cy = max(0, raw_cy - chunk_overlap)
                ex = min(origin_w, raw_cx + chunk_size + chunk_overlap)
                ey = min(origin_h, raw_cy + chunk_size + chunk_overlap)
                cw = ex - cx
                ch = ey - cy

                chunk_meta = {
                    **cache_meta,
                    "chunk_col": c,
                    "chunk_row": r,
                    "chunk_x": cx,
                    "chunk_y": cy,
                    "chunk_w": cw,
                    "chunk_h": ch,
                    "chunk_overlap": chunk_overlap,
                    "chunk_size": chunk_size,
                }
                chunk_cache_path = chunks_cache_dir / f"chunk_{c}_{r}.npz"
                chunk_features = self._load_feature_cache(chunk_cache_path, chunk_meta)

                if chunk_features is None:
                    logger.debug(f"加载并计算分块 {(c, r)} (大小 {cw}x{ch})...")
                    chunk_img = original_map[cy:cy + ch, cx:cx + cw]
                    chunk_img = cv2.convertScaleAbs(chunk_img, alpha=2.5, beta=-20)
                    chunk_features = backend.extract(chunk_img)
                    if chunk_features is not None:
                        self._save_feature_cache(chunk_cache_path, chunk_meta, chunk_features)

                chunks_cache[(c, r)] = (chunk_features, cx, cy)

        logger.info("所有分块 LightGlue 特征预加载完毕")

        last_center = None
        pending_center = None
        pending_count = 0
        lost_frames = 0
        current_chunk_idx = None

        logger.info("press Q to quit")

        while True:
            if context.tasker.stopping:
                cv2.destroyAllWindows()
                break

            loop_start = time.perf_counter()
            img = controller.post_screencap().wait().get()

            x, y, w, h = mini_map_roi
            minimap = img[y:y + h, x:x + w]
            masked = self._preprocess_minimap(minimap, circle_padding, center_radius)
            mh, mw = masked.shape[:2]

            mini_features = backend.extract(masked)

            polygon = None
            player_point = None
            match_count = 0
            inliers = 0
            raw_player_point = None

            if self._has_enough_features(mini_features, min_matches):
                approx_player_point = last_center

                if approx_player_point is None:
                    global_matches, global_scores = backend.match(mini_features, global_features)
                    if len(global_matches) >= min_matches:
                        global_matches, global_scores = self._filter_matches(global_matches, global_scores)
                        if len(global_matches) >= min_matches:
                            src_pts = np.float32([mini_features["keypoints"][m[0]] for m in global_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([global_features["keypoints"][m[1]] for m in global_matches]).reshape(-1, 1, 2)

                            M_global, mask_global = cv2.estimateAffinePartial2D(
                                src_pts,
                                dst_pts,
                                method=cv2.RANSAC,
                                ransacReprojThreshold=ransac_thresh,
                            )
                            if M_global is not None and mask_global is not None and int(mask_global.sum()) >= min_inliers:
                                player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                                player_dst = cv2.transform(player_src, M_global)[0, 0]
                                approx_player_point = (
                                    int(player_dst[0] / global_scale),
                                    int(player_dst[1] / global_scale),
                                )

                if approx_player_point is not None:
                    target_col = max(0, min(chunk_cols - 1, int(approx_player_point[0] // chunk_size)))
                    target_row = max(0, min(chunk_rows - 1, int(approx_player_point[1] // chunk_size)))

                    raw_neighbors = [
                        (target_col, target_row),
                        (target_col - 1, target_row),
                        (target_col + 1, target_row),
                        (target_col, target_row - 1),
                        (target_col, target_row + 1),
                        (target_col - 1, target_row - 1),
                        (target_col + 1, target_row - 1),
                        (target_col - 1, target_row + 1),
                        (target_col + 1, target_row + 1),
                    ]
                    neighbors = [(c, r) for c, r in raw_neighbors if 0 <= c < chunk_cols and 0 <= r < chunk_rows]

                    def dist_to_chunk_center(idx: tuple[int, int]) -> float:
                        c, r = idx
                        center_x = c * chunk_size + chunk_size / 2
                        center_y = r * chunk_size + chunk_size / 2
                        return math.hypot(approx_player_point[0] - center_x, approx_player_point[1] - center_y)

                    sorted_neighbors = sorted(neighbors, key=dist_to_chunk_center)

                    for neighbor_idx in sorted_neighbors:
                        chunk_features, cx, cy = chunks_cache[neighbor_idx]
                        if not self._has_enough_features(chunk_features, min_matches):
                            continue

                        candidate_matches, candidate_scores = backend.match(mini_features, chunk_features)
                        if len(candidate_matches) < min_matches:
                            continue

                        candidate_matches, candidate_scores = self._filter_matches(candidate_matches, candidate_scores)
                        if len(candidate_matches) < min_matches:
                            continue

                        src_pts = np.float32([mini_features["keypoints"][m[0]] for m in candidate_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([chunk_features["keypoints"][m[1]] for m in candidate_matches]).reshape(-1, 1, 2)

                        M, mask = cv2.estimateAffinePartial2D(
                            src_pts,
                            dst_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=ransac_thresh,
                        )
                        if M is None or mask is None:
                            continue

                        current_inliers = int(mask.sum())
                        if current_inliers < min_inliers:
                            continue

                        current_chunk_idx = neighbor_idx
                        match_count = len(candidate_matches)
                        inliers = current_inliers

                        corners = np.float32([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]]).reshape(-1, 1, 2)
                        polygon_local = cv2.transform(corners, M)
                        polygon = polygon_local.copy()
                        for i in range(len(polygon)):
                            polygon[i][0][0] = (polygon[i][0][0] + cx) * global_scale
                            polygon[i][0][1] = (polygon[i][0][1] + cy) * global_scale

                        player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                        player_dst = cv2.transform(player_src, M)[0, 0]
                        player_point = (
                            int(player_dst[0] + cx),
                            int(player_dst[1] + cy),
                        )
                        raw_player_point = player_point
                        break

            if player_point is not None:
                accept_point = True
                px, py = player_point

                if px < 0 or py < 0 or px >= origin_w or py >= origin_h:
                    accept_point = False

                if accept_point and last_center is not None:
                    jump = math.hypot(player_point[0] - last_center[0], player_point[1] - last_center[1])
                    max_jump = 60
                    if inliers >= 8 or match_count >= 14:
                        max_jump = 90

                    if jump > max_jump:
                        if pending_center is not None:
                            pending_jump = math.hypot(player_point[0] - pending_center[0], player_point[1] - pending_center[1])
                            if pending_jump <= 25:
                                pending_count += 1
                            else:
                                pending_center = player_point
                                pending_count = 1
                        else:
                            pending_center = player_point
                            pending_count = 1

                        if pending_count < 2:
                            logger.debug(
                                f"reject jump raw={player_point} last={last_center} jump={jump:.1f} inliers={inliers}"
                            )
                            accept_point = False
                        else:
                            logger.debug(
                                f"accept delayed jump raw={player_point} last={last_center} jump={jump:.1f}"
                            )
                            pending_center = None
                            pending_count = 0
                    else:
                        pending_center = None
                        pending_count = 0

                if accept_point:
                    if last_center is not None:
                        player_point = (
                            int(last_center[0] * 0.7 + player_point[0] * 0.3),
                            int(last_center[1] * 0.7 + player_point[1] * 0.3),
                        )
                    last_center = player_point
                else:
                    player_point = last_center
                    polygon = None

            if raw_player_point is None:
                lost_frames += 1
                if lost_frames > 2:
                    last_center = None
            else:
                lost_frames = 0

            if player_point is None:
                player_point = last_center

            map_view = global_map.copy()
            if polygon is not None:
                cv2.polylines(map_view, [np.int32(polygon)], True, (0, 255, 0), 3)
            if player_point is not None:
                draw_player_point = (
                    int(player_point[0] * global_scale),
                    int(player_point[1] * global_scale),
                )
                cv2.circle(map_view, draw_player_point, 16, (0, 0, 255), -1)

            cv2.putText(
                map_view,
                f"kp={self._feature_count(mini_features)} matches={match_count} inliers={inliers}",
                (12, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if current_chunk_idx is not None:
                cv2.putText(
                    map_view,
                    f"chunk={current_chunk_idx}",
                    (12, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (255, 128, 0),
                    2,
                    cv2.LINE_AA,
                )

            if raw_player_point is not None and player_point is not None and raw_player_point != player_point:
                cv2.putText(
                    map_view,
                    f"coordinate=({player_point[0]}, {player_point[1]})",
                    (12, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if map_view.shape[1] > debug_map_width:
                scale = debug_map_width / map_view.shape[1]
                map_view = cv2.resize(
                    map_view,
                    (debug_map_width, int(map_view.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            mini_view = cv2.resize(masked, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.putText(mini_view, "mini map", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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

            cv2.imshow("map_locator_lightglue", np.concatenate([mini_view, map_view], axis=1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep_time = frame_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)

    def _parse_params(self, custom_action_param: str) -> dict:
        if not custom_action_param:
            return {}

        try:
            params = json.loads(custom_action_param)
            if isinstance(params, dict):
                return params
        except Exception as exc:
            logger.warning(f"解析 custom_action_param 失败，将使用默认参数: {exc}")
        return {}

    def _get_backend(self, max_num_keypoints: int, device: str) -> _LightGlueBackend:
        key = (max_num_keypoints, device)
        if key not in self._backend_cache:
            self._backend_cache[key] = _LightGlueBackend(
                max_num_keypoints=max_num_keypoints,
                device=device,
            )
        return self._backend_cache[key]

    def _preprocess_minimap(self, minimap: np.ndarray, circle_padding: int, center_radius: int) -> np.ndarray:
        masked = minimap.copy()
        mh, mw = masked.shape[:2]

        center = (mw // 2, mh // 2)
        radius = max(1, min(mw, mh) // 2 - circle_padding)
        circle_mask = np.zeros((mh, mw), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)

        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        upper_hsv = np.array([179, 66, 80], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        final_mask = cv2.bitwise_and(circle_mask, hsv_mask)
        masked = cv2.bitwise_and(masked, masked, mask=final_mask)
        masked = cv2.convertScaleAbs(masked, alpha=3.8, beta=-40)
        cv2.circle(masked, center, center_radius, (0, 0, 0), -1)
        return masked

    def _feature_count(self, features: dict[str, np.ndarray] | None) -> int:
        if not features:
            return 0
        keypoints = features.get("keypoints")
        if keypoints is None:
            return 0
        return int(len(keypoints))

    def _has_enough_features(self, features: dict[str, np.ndarray] | None, min_matches: int) -> bool:
        if not features:
            return False
        return self._feature_count(features) >= min_matches

    def _filter_matches(self, matches: np.ndarray, scores: np.ndarray, keep_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        if len(matches) == 0:
            return matches, scores

        if len(scores) != len(matches):
            scores = np.ones((len(matches),), dtype=np.float32)

        keep_count = max(8, int(len(matches) * keep_ratio))
        keep_count = min(len(matches), keep_count)
        if keep_count <= 0:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

        order = np.argsort(scores)[::-1][:keep_count]
        return matches[order], scores[order]

    def _load_feature_cache(self, cache_path: Path, cache_meta: dict) -> dict[str, np.ndarray] | None:
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_meta_raw = cache["meta"]
                saved_meta = json.loads(cache_meta_raw.item() if hasattr(cache_meta_raw, "item") else str(cache_meta_raw))
                if saved_meta != cache_meta:
                    return None

                features: dict[str, np.ndarray] = {}
                for key in ("keypoints", "descriptors", "keypoint_scores", "image_size"):
                    if key not in cache:
                        continue
                    array = cache[key]
                    if array.size == 0:
                        continue
                    features[key] = array

                return features if features else None
        except Exception as exc:
            logger.error(f"特征缓存读取失败 {cache_path.name}: {exc}")
            return None

    def _save_feature_cache(self, cache_path: Path, cache_meta: dict, features: dict[str, np.ndarray] | None) -> None:
        if not features:
            return

        try:
            np.savez_compressed(
                cache_path,
                meta=json.dumps(cache_meta, ensure_ascii=False),
                keypoints=features.get("keypoints", np.empty((0, 2), dtype=np.float32)),
                descriptors=features.get("descriptors", np.empty((0,), dtype=np.float32)),
                keypoint_scores=features.get("keypoint_scores", np.empty((0,), dtype=np.float32)),
                image_size=features.get("image_size", np.empty((0,), dtype=np.int32)),
            )
        except Exception as exc:
            logger.error(f"特征缓存保存失败 {cache_path.name}: {exc}")
