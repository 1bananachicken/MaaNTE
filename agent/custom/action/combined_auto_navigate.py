import heapq
import json
import math
import os
import subprocess
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from .Common.logger import get_logger
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = get_logger(__name__)


@AgentServer.custom_action("combined_auto_navigate")
class CombinedAutoNavigate(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger.info("=== A* Voice Navigate Started ===")
        controller = context.tasker.controller

        params = self._load_params(argv.custom_action_param)
        settings = self._read_settings(params)

        # 离线大地图用于展示和定位，map_data.npz 的 mask 用于 A* 可行走区域。
        base_dir = self._resolve_base_dir()
        map_path = base_dir / "image/map/map.jpg"
        map_data_path = base_dir / "image/map/map_data.npz"

        if not map_path.exists():
            logger.error(f"大地图不存在: {map_path}")
            return CustomAction.RunResult(success=False)
        if not map_data_path.exists():
            logger.error(f"寻路掩码不存在: {map_data_path}")
            return CustomAction.RunResult(success=False)

        big_map = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if big_map is None:
            logger.error(f"大地图读取失败: {map_path}")
            return CustomAction.RunResult(success=False)

        try:
            mask = np.load(map_data_path, allow_pickle=False)["mask"]
        except Exception as exc:
            logger.exception(f"寻路掩码读取失败: {exc}")
            return CustomAction.RunResult(success=False)

        if mask.shape[:2] != big_map.shape[:2]:
            logger.error(f"寻路掩码尺寸不匹配: mask={mask.shape[:2]}, map={big_map.shape[:2]}")
            return CustomAction.RunResult(success=False)

        # 当前位置仍然使用小地图到离线大地图的 SIFT 匹配，不依赖游戏内蓝线。
        locator = self._prepare_locator(map_path, big_map, settings)
        if locator is None:
            return CustomAction.RunResult(success=False)

        start_point = self._wait_player_position(context, controller, locator, settings)
        if start_point is None:
            logger.error("无法定位当前位置，寻路中止")
            return CustomAction.RunResult(success=False)

        target_point = self._select_target_point(big_map, mask, start_point, settings)
        if target_point is None:
            logger.warning("未选择目标点，导航取消")
            return CustomAction.RunResult(success=False)

        # A* 只负责规划路线；后续只播报和可视化，不做任何按键/鼠标自动操作。
        route_polyline = self._plan_route(mask, start_point, target_point, settings)
        if not route_polyline:
            logger.error("A* 寻路失败，无法生成导航路线")
            return CustomAction.RunResult(success=False)

        route_distances = self._polyline_distances(route_polyline)
        route_total_m = route_distances[-1] * settings["meters_per_pixel"]
        # 转弯点从最终平滑后的折线里提取，避免对 A* 的格子抖动过度播报。
        turn_events = self._build_turn_events(route_polyline, route_distances, settings)
        logger.info(
            f"A* 路线生成成功，节点数={len(route_polyline)}，全程约 {route_total_m:.1f} 米，"
            f"转弯提示点={len(turn_events)}"
        )

        debug_view = self._draw_astar_view(big_map, mask, route_polyline, start_point, target_point, settings)
        self._save_debug_view(debug_view)
        nav_base_view = None
        nav_scale = None
        if settings["visualize_astar"]:
            self._show_astar_view(debug_view, "A* Navigation Path", settings["visualization_width"])
            nav_base_view, nav_scale = self._make_map_preview(big_map, None, settings["visualization_width"])

        # 语音管理器保证同一时间只有一句 TTS 在播，普通提示会合并等待。
        speech = _SpeechManager(settings["voice_enabled"], settings["speech_min_interval"])
        speech.say(f"路线规划完成，开始导航，全程约 {max(1, int(round(route_total_m)))} 米", priority=True)

        last_center = start_point
        last_progress_along = 0.0
        current_segment_idx = 0
        last_offroute_speech = 0.0
        offroute_active = False
        next_turn_idx = 0
        spoken_turn_far = set()
        spoken_turn_now = set()
        arrival_px = settings["arrival_distance_m"] / max(0.001, settings["meters_per_pixel"])
        offroute_px = settings["offroute_distance_m"] / max(0.001, settings["meters_per_pixel"])
        turn_pass_grace_px = settings["turn_pass_grace_m"] / max(0.001, settings["meters_per_pixel"])

        try:
            while True:
                if context.tasker.stopping:
                    speech.stop()
                    cv2.destroyAllWindows()
                    return CustomAction.RunResult(success=False)

                loop_start = time.perf_counter()
                speech.tick()
                frame = controller.post_screencap().wait().get()
                if frame is None:
                    time.sleep(0.05)
                    continue

                player_point, locate_stats = self._locate_player(frame, locator, settings)
                if player_point is not None:
                    player_point = self._smooth_position(player_point, last_center, locate_stats, settings)
                    last_center = player_point
                else:
                    player_point = last_center

                if player_point is not None:
                    progress = self._project_progress(
                        player_point,
                        route_polyline,
                        route_distances,
                        current_segment_idx,
                        last_progress_along,
                        offroute_px,
                    )
                    if progress is not None:
                        current_segment_idx, segment_distance, along = progress
                        if along > last_progress_along:
                            last_progress_along = along

                        distance_to_end = max(0.0, route_distances[-1] - last_progress_along)
                        direct_distance_to_target = math.hypot(
                            player_point[0] - route_polyline[-1][0],
                            player_point[1] - route_polyline[-1][1],
                        )

                        if distance_to_end <= arrival_px or direct_distance_to_target <= arrival_px:
                            message = "已到达目标点附近，缺德导航结束"
                            speech.say(message, priority=True)
                            logger.info(message)
                            cv2.destroyAllWindows()
                            return CustomAction.RunResult(success=True)

                        if segment_distance > offroute_px:
                            now = time.perf_counter()
                            if now - last_offroute_speech >= 8.0:
                                speech.say("您已偏离路线")
                                last_offroute_speech = now
                            offroute_active = True
                        else:
                            if offroute_active:
                                speech.say("已回到路线")
                            offroute_active = False

                        while (
                            next_turn_idx < len(turn_events)
                            and turn_events[next_turn_idx]["along"] < last_progress_along - turn_pass_grace_px
                        ):
                            # 已经过掉的转弯点不再补播任何提前提示。
                            for announce_distance_m in settings["turn_announce_distances_m"]:
                                spoken_turn_far.add((next_turn_idx, announce_distance_m))
                            spoken_turn_now.add(next_turn_idx)
                            next_turn_idx += 1

                        if next_turn_idx < len(turn_events):
                            turn_event = turn_events[next_turn_idx]
                            remain_m = (turn_event["along"] - last_progress_along) * settings["meters_per_pixel"]
                            if 0 < remain_m <= settings["turn_now_distance_m"] and next_turn_idx not in spoken_turn_now:
                                speech.say(f"现在{turn_event['direction']}转", priority=True)
                                for announce_distance_m in settings["turn_announce_distances_m"]:
                                    spoken_turn_far.add((next_turn_idx, announce_distance_m))
                                spoken_turn_now.add(next_turn_idx)
                            elif next_turn_idx not in spoken_turn_now:
                                pending_thresholds = [
                                    announce_distance_m
                                    for announce_distance_m in settings["turn_announce_distances_m"]
                                    if 0 < remain_m <= announce_distance_m
                                    and (next_turn_idx, announce_distance_m) not in spoken_turn_far
                                ]
                                if pending_thresholds:
                                    # 如果一开始就进入 100 米内，只播最近阈值，并跳过更远阈值。
                                    announce_distance_m = min(pending_thresholds)
                                    speech.say(
                                        f"前方 {int(round(announce_distance_m))} 米{turn_event['direction']}转",
                                    )
                                    for skipped_distance_m in pending_thresholds:
                                        spoken_turn_far.add((next_turn_idx, skipped_distance_m))

                if settings["visualize_astar"]:
                    nav_view = self._draw_navigation_view(
                        nav_base_view,
                        nav_scale,
                        route_polyline,
                        player_point,
                        target_point,
                        last_progress_along,
                        route_distances,
                        settings,
                    )
                    self._show_astar_view(nav_view, "A* Navigation Path", settings["visualization_width"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        speech.stop()
                        cv2.destroyAllWindows()
                        return CustomAction.RunResult(success=False)

                sleep_time = settings["frame_interval"] - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            speech.stop()
            cv2.destroyAllWindows()
            logger.exception(f"A* 导航运行异常: {exc}")
            return CustomAction.RunResult(success=False)

    def _load_params(self, raw_param: str | None) -> dict:
        if not raw_param:
            return {}
        try:
            loaded = json.loads(raw_param)
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            logger.warning(f"解析 custom_action_param 失败，将使用默认参数: {exc}")
        return {}

    def _read_settings(self, params: dict) -> dict:
        # JSON 里只需要放常用调参项；没写出来的高级参数在这里保留默认值。
        def as_bool(name: str, default: bool) -> bool:
            value = params.get(name, default)
            if isinstance(value, str):
                return value.strip().lower() not in ("0", "false", "no", "off")
            return bool(value)

        def as_float(name: str, default: float, minimum: float | None = None) -> float:
            try:
                value = float(params.get(name, default))
            except Exception:
                value = default
            if minimum is not None:
                value = max(minimum, value)
            return value

        def as_int(name: str, default: int, minimum: int | None = None) -> int:
            try:
                value = int(params.get(name, default))
            except Exception:
                value = default
            if minimum is not None:
                value = max(minimum, value)
            return value

        def as_float_list(name: str, default: list[float]) -> list[float]:
            raw_value = params.get(name, None)
            if raw_value is None and name == "turn_announce_distances_m":
                raw_value = params.get("turn_announce_distance_m", default)
            if isinstance(raw_value, str):
                values = [part.strip() for part in raw_value.split(",") if part.strip()]
            elif isinstance(raw_value, (list, tuple)):
                values = raw_value
            else:
                values = [raw_value]

            parsed = []
            for value in values:
                try:
                    parsed.append(max(1.0, float(value)))
                except Exception:
                    continue
            if not parsed:
                parsed = default
            return sorted(set(parsed), reverse=True)

        mini_map_roi = params.get("mini_map_roi", [24, 14, 159, 157])
        if not isinstance(mini_map_roi, list) or len(mini_map_roi) != 4:
            mini_map_roi = [24, 14, 159, 157]

        return {
            "mini_map_roi": [int(v) for v in mini_map_roi],
            "voice_enabled": as_bool("voice_enabled", True),
            "visualize_astar": as_bool("visualize_astar", True),
            "meters_per_pixel": as_float("meters_per_pixel", 0.2144, 0.001),
            "grid_size": as_int("grid_size", 16, 2),
            "walkable_threshold": as_int("walkable_threshold", 1, 1),
            "snap_radius_cells": as_int("snap_radius_cells", 220, 1),
            "path_smoothing": as_bool("path_smoothing", True),
            "path_smoothing_max_skip_cells": as_int("path_smoothing_max_skip_cells", 120, 2),
            "arrival_distance_m": as_float("arrival_distance_m", 12.0, 1.0),
            "offroute_distance_m": as_float("offroute_distance_m", 35.0, 1.0),
            "route_simplify_m": as_float("route_simplify_m", 4.0, 0.0),
            "frame_interval": as_float("frame_interval", 0.12, 0.03),
            "speech_min_interval": as_float("speech_min_interval", 1.2, 0.0),
            "position_alpha": min(1.0, max(0.05, as_float("position_alpha", 0.65))),
            "turn_angle_threshold": as_float("turn_angle_threshold", 35.0, 1.0),
            "turn_merge_distance_m": as_float("turn_merge_distance_m", 12.0, 0.0),
            "turn_announce_distances_m": as_float_list("turn_announce_distances_m", [500.0, 200.0]),
            "turn_now_distance_m": as_float("turn_now_distance_m", 40.0, 1.0),
            "turn_pass_grace_m": as_float("turn_pass_grace_m", 8.0, 0.0),
            "visualization_width": as_int("visualization_width", 1100, 320),
            "selector_window_width": as_int("selector_window_width", 1600, 320),
            "selector_window_height": as_int("selector_window_height", 960, 240),
        }

    def _resolve_base_dir(self) -> Path:
        root = Path(__file__).parents[3]
        if (root / "assets").exists():
            return root / "assets/resource/base"
        return root / "resource/base"

    def _prepare_locator(self, map_path: Path, big_map: np.ndarray, settings: dict) -> dict | None:
        # 对大地图做一次 SIFT 特征提取并缓存；运行中每帧只匹配小地图。
        origin_h, origin_w = big_map.shape[:2]
        max_processing_long_side = 6144
        scale = min(1.0, max_processing_long_side / max(origin_h, origin_w))
        work_map = big_map
        if scale < 1.0:
            work_map = cv2.resize(
                big_map,
                (int(origin_w * scale), int(origin_h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        work_map = cv2.convertScaleAbs(work_map, alpha=2.5, beta=-20)
        work_gray = cv2.cvtColor(work_map, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=0)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        cache_path = map_path.with_name(f"{map_path.stem}.sift_cache.npz")
        cache_meta = {
            "map_size": int(map_path.stat().st_size),
            "map_mtime_ns": map_path.stat().st_mtime_ns,
            "origin_w": origin_w,
            "origin_h": origin_h,
            "proc_w": work_map.shape[1],
            "proc_h": work_map.shape[0],
            "scale": scale,
        }

        big_points = None
        des_big = None
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    cache_meta_raw = cache["meta"]
                    saved_meta = json.loads(cache_meta_raw.item() if hasattr(cache_meta_raw, "item") else str(cache_meta_raw))
                    if saved_meta == cache_meta:
                        big_points = cache["keypoints"].astype(np.float32, copy=False)
                        des_big = cache["descriptors"]
            except Exception as exc:
                logger.warning(f"读取大地图特征缓存失败，将重新提取: {exc}")

        if big_points is None or des_big is None:
            kp_big, des_big = sift.detectAndCompute(work_gray, None)
            if des_big is not None:
                big_points = np.float32([kp.pt for kp in kp_big])
                try:
                    np.savez_compressed(
                        cache_path,
                        meta=json.dumps(cache_meta, ensure_ascii=False),
                        keypoints=big_points,
                        descriptors=des_big,
                    )
                except Exception as exc:
                    logger.warning(f"保存大地图特征缓存失败: {exc}")

        if des_big is None or big_points is None or len(big_points) < 8:
            logger.error("大地图特征点不足，无法定位")
            return None

        return {
            "sift": sift,
            "matcher": matcher,
            "big_points": big_points,
            "des_big": des_big,
            "scale": scale,
            "origin_w": origin_w,
            "origin_h": origin_h,
        }

    def _wait_player_position(self, context: Context, controller, locator: dict, settings: dict):
        logger.info("正在定位当前位置...")
        last_point = None
        for _ in range(40):
            if context.tasker.stopping:
                return None
            frame = controller.post_screencap().wait().get()
            if frame is None:
                time.sleep(0.05)
                continue
            point, stats = self._locate_player(frame, locator, settings)
            if point is not None:
                point = self._smooth_position(point, last_point, stats, settings)
                logger.info(f"当前位置定位成功: {point}")
                return point
            time.sleep(0.08)
        return None

    def _locate_player(self, frame: np.ndarray, locator: dict, settings: dict):
        # 小地图只有圆形区域可靠，先裁圆、压暗 UI 干扰，再做 SIFT 匹配。
        x, y, w, h = settings["mini_map_roi"]
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        minimap = frame[y:y + h, x:x + w].copy()
        mh, mw = minimap.shape[:2]
        center = (mw // 2, mh // 2)
        radius = max(1, min(mw, mh) // 2 - 15)
        circle_mask = np.zeros((mh, mw), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([179, 66, 80], dtype=np.uint8),
        )
        final_mask = cv2.bitwise_and(circle_mask, hsv_mask)
        masked = cv2.bitwise_and(minimap, minimap, mask=final_mask)
        masked = cv2.convertScaleAbs(masked, alpha=3.8, beta=-40)
        cv2.circle(masked, center, 11, (0, 0, 0), -1)

        mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        kp_mini, des_mini = locator["sift"].detectAndCompute(mini_gray, None)
        if des_mini is None or len(kp_mini) < 8:
            return None, {"inliers": 0, "matches": 0}

        good_matches = []
        for pair in locator["matcher"].knnMatch(des_mini, locator["des_big"], k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return None, {"inliers": 0, "matches": len(good_matches)}

        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([locator["big_points"][m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=12.0,
        )
        if affine is None or inlier_mask is None:
            return None, {"inliers": 0, "matches": len(good_matches)}

        inliers = int(inlier_mask.sum())
        if inliers < 4:
            return None, {"inliers": inliers, "matches": len(good_matches)}

        player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
        player_dst = cv2.transform(player_src, affine)[0, 0]
        player_point = (
            int(player_dst[0] / locator["scale"]),
            int(player_dst[1] / locator["scale"]),
        )
        if (
            player_point[0] < 0
            or player_point[1] < 0
            or player_point[0] >= locator["origin_w"]
            or player_point[1] >= locator["origin_h"]
        ):
            return None, {"inliers": inliers, "matches": len(good_matches)}

        return player_point, {"inliers": inliers, "matches": len(good_matches)}

    def _smooth_position(self, point, last_point, stats: dict, settings: dict):
        if last_point is None:
            return point

        jump = math.hypot(point[0] - last_point[0], point[1] - last_point[1])
        max_jump = 90 if stats.get("inliers", 0) >= 8 or stats.get("matches", 0) >= 14 else 60
        if jump > max_jump:
            return last_point

        alpha = settings["position_alpha"]
        return (
            int(last_point[0] * (1.0 - alpha) + point[0] * alpha),
            int(last_point[1] * (1.0 - alpha) + point[1] * alpha),
        )

    def _select_target_point(self, big_map: np.ndarray, mask: np.ndarray, start_point, settings: dict):
        selector = _MapPointSelector(
            big_map,
            mask,
            start_point,
            settings["selector_window_width"],
            settings["selector_window_height"],
            "Select A* Target",
        )
        return selector.select()

    def _plan_route(self, mask: np.ndarray, start_point, target_point, settings: dict):
        # 在降采样网格上跑 A*，速度更稳；最终再映射回原图坐标。
        grid_size = settings["grid_size"]
        grid_w = math.ceil(mask.shape[1] / grid_size)
        grid_h = math.ceil(mask.shape[0] / grid_size)
        resized = cv2.resize(mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        walkable = resized >= settings["walkable_threshold"]

        start_cell = self._to_grid_cell(start_point, grid_size, walkable.shape)
        target_cell = self._to_grid_cell(target_point, grid_size, walkable.shape)
        # 点选位置可能不在路面中心，先吸附到最近可行走格。
        start_cell = self._nearest_walkable_cell(walkable, start_cell, settings["snap_radius_cells"])
        target_cell = self._nearest_walkable_cell(walkable, target_cell, settings["snap_radius_cells"])
        if start_cell is None or target_cell is None:
            logger.error(f"起点或终点附近没有可行走区域: start={start_cell}, target={target_cell}")
            return None

        component_count, labels = cv2.connectedComponents(walkable.astype(np.uint8), 8)
        start_label = int(labels[start_cell[1], start_cell[0]])
        target_label = int(labels[target_cell[1], target_cell[0]])
        if start_label == 0:
            logger.error("起点不在可行走区域")
            return None
        if target_label != start_label:
            # 如果终点吸附到了隔离区域，尝试拉回到起点所在连通块。
            logger.warning("目标点与当前位置不在同一连通区域，尝试吸附到最近的同连通区域")
            snapped = self._nearest_labeled_cell(labels, target_cell, start_label, settings["snap_radius_cells"] * 3)
            if snapped is None:
                logger.error(f"目标区域不可达，连通块数量={component_count - 1}")
                return None
            target_cell = snapped

        cell_path = self._astar(walkable, start_cell, target_cell)
        if not cell_path:
            return None
        raw_cell_path_len = len(cell_path)
        if settings["path_smoothing"]:
            # A* 原始路径会贴着网格走；视线平滑会尽量用可通行直线替换连续折线。
            cell_path = self._smooth_cell_path(walkable, cell_path, settings["path_smoothing_max_skip_cells"])
            logger.info(f"A* 路线平滑: grid nodes {raw_cell_path_len} -> {len(cell_path)}")

        route = [self._from_grid_cell(cell, grid_size, mask.shape) for cell in cell_path]
        route[0] = start_point

        epsilon = settings["route_simplify_m"] / max(0.001, settings["meters_per_pixel"])
        route = self._simplify_polyline(route, epsilon)
        if len(route) < 2:
            return None
        return route

    def _to_grid_cell(self, point, grid_size: int, grid_shape):
        grid_h, grid_w = grid_shape
        return (
            max(0, min(grid_w - 1, int(point[0] // grid_size))),
            max(0, min(grid_h - 1, int(point[1] // grid_size))),
        )

    def _from_grid_cell(self, cell, grid_size: int, map_shape):
        map_h, map_w = map_shape[:2]
        return (
            max(0, min(map_w - 1, int((cell[0] + 0.5) * grid_size))),
            max(0, min(map_h - 1, int((cell[1] + 0.5) * grid_size))),
        )

    def _nearest_walkable_cell(self, walkable: np.ndarray, cell, max_radius: int):
        x, y = cell
        h, w = walkable.shape
        if walkable[y, x]:
            return cell

        for radius in range(1, max_radius + 1):
            x1 = max(0, x - radius)
            x2 = min(w - 1, x + radius)
            y1 = max(0, y - radius)
            y2 = min(h - 1, y + radius)
            candidates = []
            if y1 >= 0:
                xs = np.where(walkable[y1, x1:x2 + 1])[0]
                candidates.extend((x1 + int(cx), y1) for cx in xs)
            if y2 != y1:
                xs = np.where(walkable[y2, x1:x2 + 1])[0]
                candidates.extend((x1 + int(cx), y2) for cx in xs)
            if y2 - y1 > 1:
                ys = np.where(walkable[y1 + 1:y2, x1])[0]
                candidates.extend((x1, y1 + 1 + int(cy)) for cy in ys)
                if x2 != x1:
                    ys = np.where(walkable[y1 + 1:y2, x2])[0]
                    candidates.extend((x2, y1 + 1 + int(cy)) for cy in ys)

            if candidates:
                return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        return None

    def _nearest_labeled_cell(self, labels: np.ndarray, cell, target_label: int, max_radius: int):
        x, y = cell
        h, w = labels.shape
        for radius in range(1, max_radius + 1):
            x1 = max(0, x - radius)
            x2 = min(w - 1, x + radius)
            y1 = max(0, y - radius)
            y2 = min(h - 1, y + radius)
            candidates = []
            top = np.where(labels[y1, x1:x2 + 1] == target_label)[0]
            candidates.extend((x1 + int(cx), y1) for cx in top)
            if y2 != y1:
                bottom = np.where(labels[y2, x1:x2 + 1] == target_label)[0]
                candidates.extend((x1 + int(cx), y2) for cx in bottom)
            if y2 - y1 > 1:
                left = np.where(labels[y1 + 1:y2, x1] == target_label)[0]
                candidates.extend((x1, y1 + 1 + int(cy)) for cy in left)
                if x2 != x1:
                    right = np.where(labels[y1 + 1:y2, x2] == target_label)[0]
                    candidates.extend((x2, y1 + 1 + int(cy)) for cy in right)
            if candidates:
                return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        return None

    def _astar(self, walkable: np.ndarray, start, goal):
        # 8 邻接 A*：对角移动代价为 sqrt(2)，路径会比 4 邻接自然一些。
        h, w = walkable.shape
        start_idx = start[1] * w + start[0]
        goal_idx = goal[1] * w + goal[0]
        g_score = np.full(h * w, np.inf, dtype=np.float32)
        parent = np.full(h * w, -1, dtype=np.int32)
        closed = np.zeros(h * w, dtype=np.bool_)

        def heuristic(x: int, y: int) -> float:
            return math.hypot(goal[0] - x, goal[1] - y)

        g_score[start_idx] = 0.0
        open_heap = [(heuristic(start[0], start[1]), 0.0, start[0], start[1])]
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )

        visited = 0
        while open_heap:
            _, current_g, x, y = heapq.heappop(open_heap)
            idx = y * w + x
            if closed[idx]:
                continue
            closed[idx] = True
            visited += 1

            if idx == goal_idx:
                logger.info(f"A* 搜索完成，访问节点={visited}")
                return self._reconstruct_cell_path(parent, start_idx, goal_idx, w)

            for dx, dy, step_cost in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h or not walkable[ny, nx]:
                    continue

                next_idx = ny * w + nx
                if closed[next_idx]:
                    continue

                tentative_g = current_g + step_cost
                if tentative_g < g_score[next_idx]:
                    g_score[next_idx] = tentative_g
                    parent[next_idx] = idx
                    heapq.heappush(open_heap, (tentative_g + heuristic(nx, ny), tentative_g, nx, ny))

        logger.error(f"A* 搜索失败，访问节点={visited}")
        return None

    def _smooth_cell_path(self, walkable: np.ndarray, path: list[tuple[int, int]], max_skip_cells: int):
        # 从当前锚点尽量连到更远的点；只要中间全可走，就删除被跨过的格点。
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        anchor_idx = 0
        last_idx = len(path) - 1
        while anchor_idx < last_idx:
            farthest_idx = min(last_idx, anchor_idx + max_skip_cells)
            next_idx = anchor_idx + 1
            for candidate_idx in range(farthest_idx, anchor_idx, -1):
                if self._has_line_of_sight(walkable, path[anchor_idx], path[candidate_idx]):
                    next_idx = candidate_idx
                    break
            smoothed.append(path[next_idx])
            anchor_idx = next_idx

        return smoothed

    def _has_line_of_sight(self, walkable: np.ndarray, start, end):
        # Bresenham 直线检测，确保平滑后的线段不会穿过不可走区域。
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        h, w = walkable.shape

        while True:
            if x0 < 0 or y0 < 0 or x0 >= w or y0 >= h or not walkable[y0, x0]:
                return False
            if x0 == x1 and y0 == y1:
                return True

            err2 = 2 * err
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy

    def _reconstruct_cell_path(self, parent: np.ndarray, start_idx: int, goal_idx: int, width: int):
        path = []
        current = goal_idx
        while current != -1:
            path.append((int(current % width), int(current // width)))
            if current == start_idx:
                break
            current = int(parent[current])

        if not path or path[-1] != (int(start_idx % width), int(start_idx // width)):
            return None
        path.reverse()
        return path

    def _simplify_polyline(self, route: list[tuple[int, int]], epsilon: float):
        # 最后用 Douglas-Peucker 轻量简化，减少细碎转角导致的多余提示。
        if len(route) <= 2 or epsilon <= 0:
            return route
        points = np.array(route, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(points, epsilon=epsilon, closed=False)
        simplified = [(int(point[0][0]), int(point[0][1])) for point in approx]
        if simplified[0] != route[0]:
            simplified.insert(0, route[0])
        if simplified[-1] != route[-1]:
            simplified.append(route[-1])
        return simplified

    def _polyline_distances(self, route: list[tuple[int, int]]):
        distances = [0.0]
        for idx in range(1, len(route)):
            distances.append(
                distances[-1] + math.hypot(route[idx][0] - route[idx - 1][0], route[idx][1] - route[idx - 1][1])
            )
        return distances

    def _build_turn_events(self, route: list[tuple[int, int]], distances: list[float], settings: dict):
        # 只把角度足够大、距离不太近的折点当成路口提示点。
        if len(route) < 3:
            return []

        events = []
        merge_distance_px = settings["turn_merge_distance_m"] / max(0.001, settings["meters_per_pixel"])
        for idx in range(1, len(route) - 1):
            prev_pt = route[idx - 1]
            curr_pt = route[idx]
            next_pt = route[idx + 1]
            prev_len = math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
            next_len = math.hypot(next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
            if prev_len < 1.0 or next_len < 1.0:
                continue

            heading_a = self._bearing(prev_pt, curr_pt)
            heading_b = self._bearing(curr_pt, next_pt)
            turn_delta = self._angle_delta(heading_b, heading_a)
            turn_angle = abs(turn_delta)
            if turn_angle < settings["turn_angle_threshold"]:
                continue

            event = {
                "idx": idx,
                "along": distances[idx],
                "direction": "右" if turn_delta > 0 else "左",
                "angle": turn_angle,
            }
            if events and event["along"] - events[-1]["along"] <= merge_distance_px:
                if event["angle"] > events[-1]["angle"]:
                    events[-1] = event
            else:
                events.append(event)

        return events

    def _project_progress(self, point, route, distances, current_segment_idx: int, last_progress: float, offroute_px: float):
        best_score = float("inf")
        best_distance = float("inf")
        best_along = last_progress
        best_segment_idx = current_segment_idx
        search_start = max(0, current_segment_idx - 5)

        for idx in range(search_start, len(route) - 1):
            x1, y1 = route[idx]
            x2, y2 = route[idx + 1]
            seg_dx = x2 - x1
            seg_dy = y2 - y1
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq <= 1e-6:
                continue
            t = ((point[0] - x1) * seg_dx + (point[1] - y1) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = x1 + seg_dx * t
            proj_y = y1 + seg_dy * t
            distance_to_segment = math.hypot(point[0] - proj_x, point[1] - proj_y)
            segment_along = distances[idx] + math.sqrt(seg_len_sq) * t
            if segment_along + offroute_px < last_progress:
                continue

            backtrack_penalty = max(0.0, last_progress - segment_along) * 0.35
            forward_bonus = min(max(0.0, segment_along - last_progress), offroute_px * 2.0) * 0.08
            score = distance_to_segment + backtrack_penalty - forward_bonus
            if score < best_score:
                best_score = score
                best_distance = distance_to_segment
                best_along = segment_along
                best_segment_idx = idx

        if best_score == float("inf"):
            return None
        return best_segment_idx, best_distance, best_along

    def _find_lookahead_point(self, route, distances, target_along: float):
        if not route:
            return None
        if target_along >= distances[-1]:
            return route[-1]
        idx = max(1, bisect_left(distances, target_along))
        prev_distance = distances[idx - 1]
        next_distance = distances[idx]
        span = max(1e-6, next_distance - prev_distance)
        t = (target_along - prev_distance) / span
        x = route[idx - 1][0] + (route[idx][0] - route[idx - 1][0]) * t
        y = route[idx - 1][1] + (route[idx][1] - route[idx - 1][1]) * t
        return int(x), int(y)

    def _bearing(self, src, dst):
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        return math.degrees(math.atan2(dx, -dy)) % 360.0

    def _angle_delta(self, target_angle: float, current_angle: float):
        return (target_angle - current_angle + 180.0) % 360.0 - 180.0

    def _draw_astar_view(self, big_map: np.ndarray, mask: np.ndarray, route, start_point, target_point, settings: dict):
        width = max(900, min(2200, max(settings["visualization_width"], 1800)))
        view, scale = self._make_map_preview(big_map, mask, width)
        self._draw_route(view, route, scale, (0, 180, 255), max(2, int(8 * scale)))
        cv2.circle(view, self._scale_point(start_point, scale), max(4, int(42 * scale)), (0, 255, 0), -1)
        cv2.circle(view, self._scale_point(target_point, scale), max(4, int(42 * scale)), (0, 0, 255), -1)
        cv2.putText(
            view,
            "A* path",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        return view

    def _draw_navigation_view(
        self,
        base_view: np.ndarray,
        scale: float,
        route,
        player_point,
        target_point,
        progress_along: float,
        route_distances,
        settings: dict,
    ):
        view = base_view.copy()
        self._draw_route(view, route, scale, (0, 180, 255), max(2, int(8 * scale)))

        progress_point = self._find_lookahead_point(route, route_distances, progress_along)
        if progress_point is not None:
            cv2.circle(view, self._scale_point(progress_point, scale), max(3, int(24 * scale)), (255, 255, 0), -1)

        if player_point is not None:
            draw_player = self._scale_point(player_point, scale)
            cv2.circle(view, draw_player, max(4, int(38 * scale)), (0, 0, 255), -1)

        cv2.circle(view, self._scale_point(target_point, scale), max(4, int(42 * scale)), (255, 0, 255), -1)
        remaining_m = max(0.0, (route_distances[-1] - progress_along) * settings["meters_per_pixel"])
        cv2.putText(
            view,
            f"remain {remaining_m:.1f}m",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        return view

    def _make_map_preview(self, big_map: np.ndarray, mask: np.ndarray | None, width: int):
        width = min(width, big_map.shape[1])
        scale = width / big_map.shape[1]
        height = max(1, int(big_map.shape[0] * scale))
        view = cv2.resize(big_map, (width, height), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask_preview = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) > 0
            view[mask_preview] = (view[mask_preview] * 0.55 + np.array([40, 120, 40]) * 0.45).astype(np.uint8)
        return view, scale

    def _scale_point(self, point, scale: float):
        return int(point[0] * scale), int(point[1] * scale)

    def _draw_route(self, view: np.ndarray, route, scale: float, color, thickness: int):
        if len(route) < 2:
            return
        points = np.array([(int(x * scale), int(y * scale)) for x, y in route], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [points], False, color, thickness, cv2.LINE_AA)

    def _save_debug_view(self, view: np.ndarray):
        try:
            debug_dir = Path("./debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / "astar_path.png"
            cv2.imwrite(str(out_path), view)
            logger.info(f"A* 寻路可视化已保存: {out_path}")
        except Exception as exc:
            logger.warning(f"A* 可视化保存失败: {exc}")

    def _show_astar_view(self, view: np.ndarray, title: str, width: int):
        preview = self._resize_to_width(view, width)
        cv2.imshow(title, preview)

    def _resize_to_width(self, image: np.ndarray, width: int):
        if image.shape[1] <= width:
            return image
        scale = width / image.shape[1]
        return cv2.resize(image, (width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)

class _SpeechManager:
    # System.Speech 的 Speak 是同步阻塞的，所以每句放到独立 powershell 进程里播。
    # 这里集中管理进程，避免多个播报同时发声。
    def __init__(self, enabled: bool, min_interval: float):
        self.enabled = enabled
        self.min_interval = min_interval
        self.process = None
        self.pending_text = None
        self.last_started_at = 0.0

    def say(self, text: str, priority: bool = False):
        if not self.enabled or os.name != "nt":
            logger.info(f"语音播报: {text}")
            return

        self._reap()
        if self.process is not None:
            if priority:
                # “现在转弯”和“到达”优先级最高，直接打断普通提示。
                self.stop()
            else:
                # 普通提示只保留最新一句，避免积压一串过期播报。
                self.pending_text = text
                return

        now = time.perf_counter()
        if not priority and now - self.last_started_at < self.min_interval:
            self.pending_text = text
            return

        self._start(text)

    def tick(self):
        if not self.pending_text:
            self._reap()
            return

        self._reap()
        if self.process is not None:
            return

        if time.perf_counter() - self.last_started_at < self.min_interval:
            return

        text = self.pending_text
        self.pending_text = None
        self._start(text)

    def stop(self):
        self.pending_text = None
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.process = None

    def _reap(self):
        if self.process is not None and self.process.poll() is not None:
            self.process = None

    def _start(self, text: str):
        logger.info(f"语音播报: {text}")
        safe_text = text.replace("'", "''")
        self.process = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-Command",
                f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 0; $s.Volume = 100; $s.Speak('{safe_text}')",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.last_started_at = time.perf_counter()


class _MapPointSelector:
    # OpenCV 的窗口大小和画布大小不是一回事，所以这里同时控制渲染尺寸和 resizeWindow。
    def __init__(
        self,
        big_map: np.ndarray,
        mask: np.ndarray,
        start_point,
        window_width: int,
        window_height: int,
        window_name: str,
    ):
        self.big_map = big_map
        self.mask = mask
        self.start_point = start_point
        self.window_width = window_width
        self.window_height = window_height
        self.window_name = window_name
        self.selected_point = None
        self.dragging = False
        self.last_mouse = None
        self.dirty = True
        self.zoom = min(1.0, window_width / big_map.shape[1])
        self.min_zoom = self.zoom
        self.max_zoom = 4.0
        initial_w = max(320, int(min(big_map.shape[1] * self.zoom, window_width)))
        initial_h = max(240, int(min(big_map.shape[0] * self.zoom, window_height)))
        self.offset_x = start_point[0] - initial_w / (2.0 * self.zoom)
        self.offset_y = start_point[1] - initial_h / (2.0 * self.zoom)
        self._clamp_offset(initial_w, initial_h)

    def select(self):
        logger.info("请在弹出的离线大地图上左键选择目标点，Enter 确认，Esc 取消；滚轮缩放，右键拖动平移")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        canvas = self._render()
        cv2.resizeWindow(self.window_name, canvas.shape[1], canvas.shape[0])
        self.dirty = False

        while True:
            if self.dirty:
                canvas = self._render()
                self.dirty = False
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(40) & 0xFF
            if key in (13, 10) and self.selected_point is not None:
                cv2.destroyWindow(self.window_name)
                logger.info(f"目标点已选择: {self.selected_point}")
                return self.selected_point
            if key == 27:
                cv2.destroyWindow(self.window_name)
                return None

    def _render(self):
        view_w = max(320, int(min(self.big_map.shape[1] * self.zoom, self.window_width)))
        view_h = max(240, int(min(self.big_map.shape[0] * self.zoom, self.window_height)))
        self._clamp_offset(view_w, view_h)

        src_x1 = int(self.offset_x)
        src_y1 = int(self.offset_y)
        src_x2 = int(min(self.big_map.shape[1], self.offset_x + view_w / self.zoom))
        src_y2 = int(min(self.big_map.shape[0], self.offset_y + view_h / self.zoom))
        crop = self.big_map[src_y1:src_y2, src_x1:src_x2]
        canvas = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(self.mask[src_y1:src_y2, src_x1:src_x2], (view_w, view_h), interpolation=cv2.INTER_AREA) > 0
        canvas[mask_small] = (canvas[mask_small] * 0.65 + np.array([45, 120, 45]) * 0.35).astype(np.uint8)
        self._draw_point(canvas, self.start_point, src_x1, src_y1, (0, 255, 0), "START")
        if self.selected_point is not None:
            self._draw_point(canvas, self.selected_point, src_x1, src_y1, (0, 0, 255), "TARGET")

        cv2.putText(
            canvas,
            "Left click target | Enter OK | Esc cancel | Wheel zoom | Right drag pan",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_point(self, canvas, point, src_x: int, src_y: int, color, label: str):
        px = int((point[0] - src_x) * self.zoom)
        py = int((point[1] - src_y) * self.zoom)
        if px < -50 or py < -50 or px > canvas.shape[1] + 50 or py > canvas.shape[0] + 50:
            return
        cv2.circle(canvas, (px, py), 9, color, -1)
        cv2.putText(canvas, label, (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = self._screen_to_map(x, y)
            self.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
            self.last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.last_mouse is not None:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            self.offset_x -= dx / self.zoom
            self.offset_y -= dy / self.zoom
            self.last_mouse = (x, y)
            self.dirty = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            before = self._screen_to_map(x, y)
            if flags > 0:
                self.zoom = min(self.max_zoom, self.zoom * 1.25)
            else:
                self.zoom = max(self.min_zoom, self.zoom / 1.25)
            self.offset_x = before[0] - x / self.zoom
            self.offset_y = before[1] - y / self.zoom
            self.dirty = True

    def _screen_to_map(self, x: int, y: int):
        map_x = int(self.offset_x + x / self.zoom)
        map_y = int(self.offset_y + y / self.zoom)
        return (
            max(0, min(self.big_map.shape[1] - 1, map_x)),
            max(0, min(self.big_map.shape[0] - 1, map_y)),
        )

    def _clamp_offset(self, view_w: int, view_h: int):
        max_x = max(0.0, self.big_map.shape[1] - view_w / self.zoom)
        max_y = max(0.0, self.big_map.shape[0] - view_h / self.zoom)
        self.offset_x = max(0.0, min(max_x, self.offset_x))
        self.offset_y = max(0.0, min(max_y, self.offset_y))
