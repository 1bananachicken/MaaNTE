import cv2
import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime

from .Common.logger import get_logger
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = get_logger(__name__)


@AgentServer.custom_action("combined_auto_navigate")
class CombinedAutoNavigate(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger.info("=== Blue Route Voice Navigation Started ===")
        controller = context.tasker.controller

        params = {}
        if argv.custom_action_param:
            try:
                loaded_params = json.loads(argv.custom_action_param)
                if isinstance(loaded_params, dict):
                    params = loaded_params
            except Exception as exc:
                logger.warning(f"解析 custom_action_param 失败，将使用默认参数: {exc}")

        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            base_dir = abs_path / "assets/resource/base"
        else:
            base_dir = abs_path / "resource/base"

        map_path = base_dir / "image/map/map.jpg"
        model_path = base_dir / "model/navi/pointer_model.onnx"

        mini_map_roi = params.get("mini_map_roi", [24, 14, 159, 157])
        pointer_roi = params.get("pointer_roi", [73, 60, 64, 64])
        screen_map_roi = params.get("screen_map_roi")
        close_map_after_capture = bool(params.get("close_map_after_capture", True))
        debug_window = bool(params.get("debug_window", True))
        route_debug_window = bool(params.get("route_debug_window", False))
        route_debug_pause = bool(params.get("route_debug_pause", route_debug_window))
        route_debug_save = bool(params.get("route_debug_save", True))
        voice_enabled = bool(params.get("voice_enabled", True))

        try:
            meters_per_pixel = float(params.get("meters_per_pixel", 1.0))
            route_sample_spacing = max(8.0, float(params.get("route_sample_spacing", 18.0)))
            turn_angle_threshold = max(18.0, float(params.get("turn_angle_threshold", 35.0)))
            turn_merge_distance = max(20.0, float(params.get("turn_merge_distance", 70.0)))
            turn_announce_distance_m = max(10.0, float(params.get("turn_announce_distance_m", 90.0)))
            turn_announce_distances_raw = params.get("turn_announce_distances_m", [500, 300, 200])
            if isinstance(turn_announce_distances_raw, str):
                turn_announce_distances_m = [
                    max(1.0, float(part.strip()))
                    for part in turn_announce_distances_raw.split(",")
                    if part.strip()
                ]
            elif isinstance(turn_announce_distances_raw, (list, tuple)):
                turn_announce_distances_m = [max(1.0, float(value)) for value in turn_announce_distances_raw]
            else:
                turn_announce_distances_m = [turn_announce_distance_m]
            turn_announce_distances_m = sorted(set(turn_announce_distances_m))
            turn_announce_late_tolerance_m = max(1.0, float(params.get("turn_announce_late_tolerance_m", 80.0)))
            turn_now_distance_m = max(3.0, float(params.get("turn_now_distance_m", 18.0)))
            offroute_distance = max(20.0, float(params.get("offroute_distance", 80.0)))
            arrival_distance_m = max(3.0, float(params.get("arrival_distance_m", 12.0)))
            speech_cooldown = max(0.2, float(params.get("speech_cooldown", 1.0)))
            frame_interval = max(0.03, float(params.get("frame_interval", 0.12)))
            position_alpha = min(1.0, max(0.05, float(params.get("position_alpha", 0.65))))
            turn_pass_grace_m = max(1.0, float(params.get("turn_pass_grace_m", 8.0)))
            turn_event_lead_m = max(0.0, float(params.get("turn_event_lead_m", 25.0)))
            progress_forward_bonus = max(0.0, float(params.get("progress_forward_bonus", 0.08)))
            progress_backtrack_penalty = max(0.0, float(params.get("progress_backtrack_penalty", 0.35)))
            route_h_low = int(params.get("route_h_low", 82))
            route_h_high = int(params.get("route_h_high", 108))
            route_s_low = int(params.get("route_s_low", 70))
            route_v_low = int(params.get("route_v_low", 110))
            route_close_kernel = max(3, int(params.get("route_close_kernel", 5)) | 1)
            route_close_iterations = max(0, int(params.get("route_close_iterations", 2)))
            route_open_kernel = max(1, int(params.get("route_open_kernel", 3)) | 1)
            route_open_iterations = max(0, int(params.get("route_open_iterations", 1)))
            skeleton_method = str(params.get("skeleton_method", "zhang_suen")).strip().lower()
            skeleton_max_iterations = max(1, int(params.get("skeleton_max_iterations", 160)))
            skeleton_bridge_enabled = bool(params.get("skeleton_bridge_enabled", True))
            skeleton_bridge_max_gap = max(4.0, float(params.get("skeleton_bridge_max_gap", 80.0)))
            skeleton_bridge_max_rounds = max(0, int(params.get("skeleton_bridge_max_rounds", 16)))
            skeleton_bridge_max_components = max(2, int(params.get("skeleton_bridge_max_components", 48)))
        except Exception as exc:
            logger.warning(f"解析导航参数失败，将使用保守默认值: {exc}")
            meters_per_pixel = 1.0
            route_sample_spacing = 18.0
            turn_angle_threshold = 35.0
            turn_merge_distance = 70.0
            turn_announce_distance_m = 90.0
            turn_announce_distances_m = [200.0, 300.0, 500.0]
            turn_announce_late_tolerance_m = 80.0
            turn_now_distance_m = 18.0
            offroute_distance = 80.0
            arrival_distance_m = 12.0
            speech_cooldown = 1.0
            frame_interval = 0.12
            position_alpha = 0.65
            turn_pass_grace_m = 8.0
            turn_event_lead_m = 25.0
            progress_forward_bonus = 0.08
            progress_backtrack_penalty = 0.35
            route_h_low = 0
            route_h_high = 179
            route_s_low = 0
            route_v_low = 62
            route_close_kernel = 5
            route_close_iterations = 2
            route_open_kernel = 32
            route_open_iterations = 0
            skeleton_method = "zhang_suen"
            skeleton_max_iterations = 160
            skeleton_bridge_enabled = True
            skeleton_bridge_max_gap = 240.0
            skeleton_bridge_max_rounds = 16
            skeleton_bridge_max_components = 48

        if not map_path.exists():
            logger.error(f"大地图不存在: {map_path}")
            return CustomAction.RunResult(success=False)
        if not model_path.exists():
            logger.error(f"朝向模型不存在: {model_path}")
            return CustomAction.RunResult(success=False)

        big_map = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if big_map is None:
            logger.error(f"大地图读取失败: {map_path}")
            return CustomAction.RunResult(success=False)

        origin_h, origin_w = big_map.shape[:2]
        max_processing_long_side = 6144
        big_map_scale = min(1.0, max_processing_long_side / max(origin_h, origin_w))
        if big_map_scale < 1.0:
            big_map = cv2.resize(
                big_map,
                (int(origin_w * big_map_scale), int(origin_h * big_map_scale)),
                interpolation=cv2.INTER_AREA,
            )
        big_map = cv2.convertScaleAbs(big_map, alpha=2.5, beta=-20)
        big_gray = cv2.cvtColor(big_map, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=0)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        cache_path = map_path.with_name(f"{map_path.stem}.sift_cache.npz")
        cache_meta = {
            "map_size": int(map_path.stat().st_size),
            "map_mtime_ns": map_path.stat().st_mtime_ns,
            "origin_w": origin_w,
            "origin_h": origin_h,
            "proc_w": big_map.shape[1],
            "proc_h": big_map.shape[0],
            "scale": big_map_scale,
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
            kp_big, des_big = sift.detectAndCompute(big_gray, None)
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
            return CustomAction.RunResult(success=False)

        backend = str(params.get("backend") or os.environ.get("MAA_ONNX_BACKEND", "cpu")).strip().lower()
        provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }
        if backend == "auto":
            backend = "directml" if "DmlExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
        if backend not in provider_name_map:
            logger.warning(f"未知推理后端 {backend}，回退到 CPU")
            backend = "cpu"

        provider_name = provider_name_map[backend]
        available_providers = onnxruntime.get_available_providers()
        if provider_name not in available_providers:
            logger.warning(f"请求的后端 {provider_name} 不可用，当前可用 Providers: {available_providers}，回退到 CPU")
            provider_name = "CPUExecutionProvider"

        session_options = onnxruntime.SessionOptions()
        if provider_name == "DmlExecutionProvider":
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=[provider_name],
                provider_options=[{"device_id": 0}],
            )
        else:
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=[provider_name],
            )
        input_name = session.get_inputs()[0].name
        logger.info(f"朝向推理后端: {provider_name}")

        logger.info("请先在大地图上选择目标追踪，确认蓝色路线可见；现在开始捕获蓝线路线")
        first_frame = controller.post_screencap().wait().get()
        if first_frame is None:
            logger.error("截图失败，无法捕获蓝线路线")
            return CustomAction.RunResult(success=False)

        screen_h, screen_w = first_frame.shape[:2]
        if not isinstance(screen_map_roi, list) or len(screen_map_roi) != 4:
            screen_map_roi = [0, 0, screen_w, screen_h]
        sx, sy, sw, sh = [int(v) for v in screen_map_roi]
        sx = max(0, min(sx, screen_w - 1))
        sy = max(0, min(sy, screen_h - 1))
        sw = max(1, min(sw, screen_w - sx))
        sh = max(1, min(sh, screen_h - sy))
        map_screen = first_frame[sy:sy + sh, sx:sx + sw].copy()

        map_screen_for_match = map_screen.copy()
        map_screen_hsv_for_match = cv2.cvtColor(map_screen_for_match, cv2.COLOR_BGR2HSV)
        colored_overlay_mask = cv2.inRange(
            map_screen_hsv_for_match,
            np.array([0, 45, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        map_screen_for_match[colored_overlay_mask > 0] = (0, 0, 0)
        map_screen_for_match = cv2.convertScaleAbs(map_screen_for_match, alpha=2.2, beta=-15)
        map_screen_gray = cv2.cvtColor(map_screen_for_match, cv2.COLOR_BGR2GRAY)
        kp_screen, des_screen = sift.detectAndCompute(map_screen_gray, None)
        if des_screen is None or len(kp_screen) < 20:
            logger.error("大地图截图特征点不足，请确认当前停留在大地图界面")
            return CustomAction.RunResult(success=False)

        good_matches = []
        for pair in matcher.knnMatch(des_screen, des_big, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 20:
            logger.error(f"大地图截图匹配点不足: {len(good_matches)}")
            return CustomAction.RunResult(success=False)

        src_pts = np.float32([kp_screen[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([big_points[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
        map_affine, map_inlier_mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=8.0,
        )
        if map_affine is None or map_inlier_mask is None or int(map_inlier_mask.sum()) < 12:
            logger.error("大地图截图到离线大地图的变换估计失败")
            return CustomAction.RunResult(success=False)
        logger.info(f"大地图截图匹配成功，inliers={int(map_inlier_mask.sum())}/{len(good_matches)}")

        # route_hsv = cv2.cvtColor(map_screen, cv2.COLOR_BGR2HSV)

        # rgb_range = [11, 212, 255]
        rgb_low = [78, 70, 9]
        rgb_high = [255, 212, 51]
        # route_raw_mask = cv2.inRange(
        #     map_screen,
        #     np.array([route_h_low, route_s_low, route_v_low], dtype=np.uint8),
        #     np.array([route_h_high, 255, 255], dtype=np.uint8),
        # )
        route_raw_mask = cv2.inRange(
            map_screen,
            np.array(rgb_low, dtype=np.uint8),
            np.array(rgb_high, dtype=np.uint8),
        )
        route_mask = route_raw_mask.copy()
        route_mask = cv2.morphologyEx(
            route_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (route_close_kernel, route_close_kernel)),
            iterations=route_close_iterations,
        )
        route_mask = cv2.morphologyEx(
            route_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (route_open_kernel, route_open_kernel)),
            iterations=route_open_iterations,
        )
        route_morphed_mask = route_mask.copy()

        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(route_mask, 8)
        route_components_view = np.zeros_like(map_screen)
        for label in range(1, component_count):
            color = (
                int((label * 53) % 205 + 50),
                int((label * 97) % 205 + 50),
                int((label * 151) % 205 + 50),
            )
            route_components_view[component_labels == label] = color

        best_label = 0
        best_score = 0.0
        min_route_area = int(params.get("min_route_area", 120))
        for label in range(1, component_count):
            area = int(component_stats[label, cv2.CC_STAT_AREA])
            width = int(component_stats[label, cv2.CC_STAT_WIDTH])
            height = int(component_stats[label, cv2.CC_STAT_HEIGHT])
            extent = math.hypot(width, height)
            if area < min_route_area or extent < 60:
                continue
            score = float(area) * (1.0 + extent / 1000.0)
            if score > best_score:
                best_label = label
                best_score = score
        if best_label == 0:
            logger.error("没有识别到足够长的蓝色路线，请确认已经在大地图上追踪目标")
            if route_debug_window or route_debug_save:
                panel_w = 420
                panel_h = max(1, int(map_screen.shape[0] * panel_w / max(1, map_screen.shape[1])))
                raw_panel = cv2.resize(map_screen, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                mask_panel = cv2.cvtColor(cv2.resize(route_raw_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                morph_panel = cv2.cvtColor(cv2.resize(route_morphed_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                comp_panel = cv2.resize(route_components_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
                cv2.putText(raw_panel, "capture", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(mask_panel, "raw hsv mask", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(morph_panel, "morphed mask", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(comp_panel, f"components={component_count - 1}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                route_debug_canvas = np.concatenate(
                    [
                        np.concatenate([raw_panel, mask_panel], axis=1),
                        np.concatenate([morph_panel, comp_panel], axis=1),
                    ],
                    axis=0,
                )
                if route_debug_save:
                    debug_dir = Path("./debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(debug_dir / "blue_route_debug.png"), route_debug_canvas)
                    logger.info(f"蓝线识别调试图已保存: {debug_dir / 'blue_route_debug.png'}")
                if route_debug_window:
                    cv2.imshow("Blue Route Capture Debug", route_debug_canvas)
                    cv2.waitKey(0 if route_debug_pause else 1)
            return CustomAction.RunResult(success=False)

        route_mask = np.where(component_labels == best_label, 255, 0).astype(np.uint8)
        skeleton_method_used = skeleton_method
        if skeleton_method in ("ximgproc", "ximgproc_zhang_suen", "opencv_zhang_suen") and hasattr(cv2, "ximgproc"):
            skeleton = cv2.ximgproc.thinning(route_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            skeleton_method_used = "ximgproc_zhang_suen"
        elif skeleton_method in ("ximgproc_guohall", "opencv_guohall") and hasattr(cv2, "ximgproc"):
            skeleton = cv2.ximgproc.thinning(route_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
            skeleton_method_used = "ximgproc_guohall"
        elif skeleton_method in ("morph", "morphology", "legacy"):
            skeleton = np.zeros(route_mask.shape, dtype=np.uint8)
            skeleton_work = route_mask.copy()
            skeleton_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while cv2.countNonZero(skeleton_work) > 0:
                eroded = cv2.erode(skeleton_work, skeleton_kernel)
                opened = cv2.dilate(eroded, skeleton_kernel)
                skeleton_piece = cv2.subtract(skeleton_work, opened)
                skeleton = cv2.bitwise_or(skeleton, skeleton_piece)
                skeleton_work = eroded
            skeleton_method_used = "morphology"
        else:
            thin = (route_mask > 0).astype(np.uint8)
            for _ in range(skeleton_max_iterations):
                changed = False

                padded = np.pad(thin, 1, mode="constant")
                p2 = padded[:-2, 1:-1]
                p3 = padded[:-2, 2:]
                p4 = padded[1:-1, 2:]
                p5 = padded[2:, 2:]
                p6 = padded[2:, 1:-1]
                p7 = padded[2:, :-2]
                p8 = padded[1:-1, :-2]
                p9 = padded[:-2, :-2]
                neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = (
                    ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                    + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                    + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                    + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                    + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                    + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                    + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                    + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                )
                delete_mask = (
                    (thin == 1)
                    & (neighbors >= 2)
                    & (neighbors <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
                if np.any(delete_mask):
                    thin[delete_mask] = 0
                    changed = True

                padded = np.pad(thin, 1, mode="constant")
                p2 = padded[:-2, 1:-1]
                p3 = padded[:-2, 2:]
                p4 = padded[1:-1, 2:]
                p5 = padded[2:, 2:]
                p6 = padded[2:, 1:-1]
                p7 = padded[2:, :-2]
                p8 = padded[1:-1, :-2]
                p9 = padded[:-2, :-2]
                neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = (
                    ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                    + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                    + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                    + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                    + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                    + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                    + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                    + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                )
                delete_mask = (
                    (thin == 1)
                    & (neighbors >= 2)
                    & (neighbors <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )
                if np.any(delete_mask):
                    thin[delete_mask] = 0
                    changed = True

                if not changed:
                    break

            skeleton = (thin * 255).astype(np.uint8)
            skeleton_method_used = "zhang_suen"

        skel_y, skel_x = np.where(skeleton > 0)
        if len(skel_x) < 20:
            logger.error("蓝线路线太短，无法生成导航折线")
            if route_debug_window or route_debug_save:
                panel_w = 420
                panel_h = max(1, int(map_screen.shape[0] * panel_w / max(1, map_screen.shape[1])))
                raw_panel = cv2.resize(map_screen, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                mask_panel = cv2.cvtColor(cv2.resize(route_raw_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                selected_panel = cv2.cvtColor(cv2.resize(route_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                skeleton_panel = cv2.cvtColor(cv2.resize(skeleton, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                cv2.putText(raw_panel, "capture", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(mask_panel, "raw hsv mask", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(selected_panel, "selected component", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(skeleton_panel, f"{skeleton_method_used} pixels={len(skel_x)}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
                route_debug_canvas = np.concatenate(
                    [
                        np.concatenate([raw_panel, mask_panel], axis=1),
                        np.concatenate([selected_panel, skeleton_panel], axis=1),
                    ],
                    axis=0,
                )
                if route_debug_save:
                    debug_dir = Path("./debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(debug_dir / "blue_route_debug.png"), route_debug_canvas)
                    logger.info(f"蓝线识别调试图已保存: {debug_dir / 'blue_route_debug.png'}")
                if route_debug_window:
                    cv2.imshow("Blue Route Capture Debug", route_debug_canvas)
                    cv2.waitKey(0 if route_debug_pause else 1)
            return CustomAction.RunResult(success=False)

        skeleton_component_count, skeleton_labels = cv2.connectedComponents(skeleton, 8)
        skeleton_components_view = np.zeros_like(map_screen)
        skeleton_component_areas = []
        for label in range(1, skeleton_component_count):
            area = int(np.count_nonzero(skeleton_labels == label))
            skeleton_component_areas.append(area)
            color = (
                int((label * 71) % 205 + 50),
                int((label * 113) % 205 + 50),
                int((label * 191) % 205 + 50),
            )
            skeleton_components_view[skeleton_labels == label] = color

        skeleton_bridge_lines = []
        if skeleton_bridge_enabled and skeleton_component_count > 2:
            bridge_mask = cv2.dilate(
                route_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            bridge_round = 0
            while skeleton_component_count > 2 and bridge_round < skeleton_bridge_max_rounds:
                component_infos = []
                for label in range(1, skeleton_component_count):
                    ys, xs = np.where(skeleton_labels == label)
                    if len(xs) == 0:
                        continue
                    sample_step = max(1, len(xs) // 350)
                    component_infos.append(
                        {
                            "area": len(xs),
                            "xs": xs[::sample_step].astype(np.float32, copy=False),
                            "ys": ys[::sample_step].astype(np.float32, copy=False),
                        }
                    )
                component_infos.sort(key=lambda item: item["area"], reverse=True)
                component_infos = component_infos[:skeleton_bridge_max_components]

                best_bridge = None
                best_dist_sq = skeleton_bridge_max_gap * skeleton_bridge_max_gap
                for left_idx in range(len(component_infos)):
                    left = component_infos[left_idx]
                    left_points = np.stack([left["xs"], left["ys"]], axis=1)
                    for right_idx in range(left_idx + 1, len(component_infos)):
                        right = component_infos[right_idx]
                        right_points = np.stack([right["xs"], right["ys"]], axis=1)
                        diff = left_points[:, None, :] - right_points[None, :, :]
                        dist_sq = np.sum(diff * diff, axis=2)
                        flat_idx = int(np.argmin(dist_sq))
                        pair_dist_sq = float(dist_sq.reshape(-1)[flat_idx])
                        if pair_dist_sq >= best_dist_sq:
                            continue
                        li, ri = np.unravel_index(flat_idx, dist_sq.shape)
                        p1 = (int(round(left_points[li, 0])), int(round(left_points[li, 1])))
                        p2 = (int(round(right_points[ri, 0])), int(round(right_points[ri, 1])))
                        probe = np.zeros_like(route_mask)
                        cv2.line(probe, p1, p2, 255, 1, cv2.LINE_8)
                        probe_pixels = max(1, cv2.countNonZero(probe))
                        inside_pixels = cv2.countNonZero(cv2.bitwise_and(probe, bridge_mask))
                        if inside_pixels / probe_pixels < 0.75:
                            continue
                        best_bridge = (p1, p2, pair_dist_sq)
                        best_dist_sq = pair_dist_sq

                if best_bridge is None:
                    break

                cv2.line(skeleton, best_bridge[0], best_bridge[1], 255, 1, cv2.LINE_8)
                skeleton_bridge_lines.append((best_bridge[0], best_bridge[1], math.sqrt(best_bridge[2])))
                skeleton_component_count, skeleton_labels = cv2.connectedComponents(skeleton, 8)
                bridge_round += 1

            skel_y, skel_x = np.where(skeleton > 0)
            skeleton_components_view = np.zeros_like(map_screen)
            skeleton_component_areas = []
            for label in range(1, skeleton_component_count):
                area = int(np.count_nonzero(skeleton_labels == label))
                skeleton_component_areas.append(area)
                color = (
                    int((label * 71) % 205 + 50),
                    int((label * 113) % 205 + 50),
                    int((label * 191) % 205 + 50),
                )
                skeleton_components_view[skeleton_labels == label] = color

        route_pixel_set = set(zip(skel_x.tolist(), skel_y.tolist()))
        endpoints = []
        for pixel in route_pixel_set:
            px, py = pixel
            degree = 0
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    if (px + ox, py + oy) in route_pixel_set:
                        degree += 1
            if degree <= 1:
                endpoints.append(pixel)

        endpoint_candidates = endpoints if len(endpoints) >= 2 else list(route_pixel_set)
        if len(endpoint_candidates) > 700:
            step = max(1, len(endpoint_candidates) // 700)
            endpoint_candidates = endpoint_candidates[::step]
        path_start = endpoint_candidates[0]
        path_end = endpoint_candidates[-1]
        farthest_pair_score = -1
        for i in range(len(endpoint_candidates)):
            ax, ay = endpoint_candidates[i]
            for j in range(i + 1, len(endpoint_candidates)):
                bx, by = endpoint_candidates[j]
                pair_score = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                if pair_score > farthest_pair_score:
                    farthest_pair_score = pair_score
                    path_start = endpoint_candidates[i]
                    path_end = endpoint_candidates[j]

        search_queue = deque([path_start])
        parent = {path_start: None}
        while search_queue:
            current_pixel = search_queue.popleft()
            if current_pixel == path_end:
                break
            cx, cy = current_pixel
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    neighbor = (cx + ox, cy + oy)
                    if neighbor in route_pixel_set and neighbor not in parent:
                        parent[neighbor] = current_pixel
                        search_queue.append(neighbor)

        route_debug_status = "connected" if path_end in parent else "disconnected"
        if route_debug_window or route_debug_save:
            route_overlay = map_screen.copy()
            route_overlay[route_mask > 0] = (
                route_overlay[route_mask > 0] * 0.45 + np.array([0, 255, 255]) * 0.55
            ).astype(np.uint8)
            route_overlay[skeleton > 0] = (0, 0, 255)
            for bridge_line in skeleton_bridge_lines:
                cv2.line(route_overlay, bridge_line[0], bridge_line[1], (255, 0, 255), 2, cv2.LINE_AA)

            reachable_overlay = map_screen.copy()
            if parent:
                reachable_mask = np.zeros_like(route_mask)
                for rx, ry in parent.keys():
                    if 0 <= rx < reachable_mask.shape[1] and 0 <= ry < reachable_mask.shape[0]:
                        reachable_mask[ry, rx] = 255
                reachable_overlay[route_mask > 0] = (
                    reachable_overlay[route_mask > 0] * 0.55 + np.array([0, 255, 255]) * 0.45
                ).astype(np.uint8)
                reachable_overlay[reachable_mask > 0] = (255, 255, 0)

            for endpoint in endpoints:
                cv2.circle(route_overlay, endpoint, 4, (0, 255, 255), -1)
            cv2.circle(route_overlay, path_start, 7, (0, 255, 0), -1)
            cv2.circle(route_overlay, path_end, 7, (0, 0, 255), -1)
            cv2.circle(reachable_overlay, path_start, 7, (0, 255, 0), -1)
            cv2.circle(reachable_overlay, path_end, 7, (0, 0, 255), -1)

            panel_w = 420
            panel_h = max(1, int(map_screen.shape[0] * panel_w / max(1, map_screen.shape[1])))
            raw_panel = cv2.resize(map_screen, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            raw_mask_panel = cv2.cvtColor(cv2.resize(route_raw_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            morph_mask_panel = cv2.cvtColor(cv2.resize(route_morphed_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            comp_panel = cv2.resize(route_components_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
            selected_panel = cv2.cvtColor(cv2.resize(route_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            skeleton_panel = cv2.resize(skeleton_components_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
            overlay_panel = cv2.resize(route_overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            reachable_panel = cv2.resize(reachable_overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            cv2.putText(raw_panel, "capture", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(raw_mask_panel, f"raw hsv H={route_h_low}-{route_h_high} S>={route_s_low} V>={route_v_low}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(morph_mask_panel, f"close {route_close_kernel}x{route_close_kernel}*{route_close_iterations}, open {route_open_kernel}x{route_open_kernel}*{route_open_iterations}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(comp_panel, f"mask components={component_count - 1}, best={best_label}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(selected_panel, "selected route component", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(skeleton_panel, f"{skeleton_method_used} comps={skeleton_component_count - 1}, endpoints={len(endpoints)}, bridges={len(skeleton_bridge_lines)}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay_panel, f"yellow=endpoints green=start red=end purple=bridges", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(reachable_panel, f"BFS {route_debug_status}, reached={len(parent)}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

            route_debug_canvas = np.concatenate(
                [
                    np.concatenate([raw_panel, raw_mask_panel], axis=1),
                    np.concatenate([morph_mask_panel, comp_panel], axis=1),
                    np.concatenate([selected_panel, skeleton_panel], axis=1),
                    np.concatenate([overlay_panel, reachable_panel], axis=1),
                ],
                axis=0,
            )
            if route_debug_save:
                debug_dir = Path("./debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "blue_route_debug.png"), route_debug_canvas)
                logger.info(f"蓝线识别调试图已保存: {debug_dir / 'blue_route_debug.png'}")
            if route_debug_window:
                cv2.imshow("Blue Route Capture Debug", route_debug_canvas)
                cv2.waitKey(0 if route_debug_pause else 1)

        logger.info(
            f"蓝线识别: skeleton_method={skeleton_method_used}, mask_components={component_count - 1}, "
            f"skeleton_components={skeleton_component_count - 1}, endpoints={len(endpoints)}, "
            f"bridges={len(skeleton_bridge_lines)}, status={route_debug_status}, "
            f"skeleton_component_areas={skeleton_component_areas[:8]}"
        )

        if path_end not in parent:
            logger.error("蓝线路线骨架不连通，无法排序")
            return CustomAction.RunResult(success=False)

        ordered_screen_points = []
        current_pixel = path_end
        while current_pixel is not None:
            ordered_screen_points.append(current_pixel)
            current_pixel = parent[current_pixel]
        ordered_screen_points.reverse()

        transformed = cv2.transform(np.float32(ordered_screen_points).reshape(-1, 1, 2), map_affine)[:, 0, :]
        route_polyline = []
        for tx, ty in transformed:
            wx = int(round(tx / big_map_scale))
            wy = int(round(ty / big_map_scale))
            if 0 <= wx < origin_w and 0 <= wy < origin_h:
                if not route_polyline or math.hypot(wx - route_polyline[-1][0], wy - route_polyline[-1][1]) >= route_sample_spacing:
                    route_polyline.append((wx, wy))
        if transformed.shape[0] > 0:
            wx = int(round(transformed[-1][0] / big_map_scale))
            wy = int(round(transformed[-1][1] / big_map_scale))
            if 0 <= wx < origin_w and 0 <= wy < origin_h and (not route_polyline or route_polyline[-1] != (wx, wy)):
                route_polyline.append((wx, wy))
        if len(route_polyline) < 4:
            logger.error("蓝线路线有效点过少，无法导航")
            return CustomAction.RunResult(success=False)

        logger.info(f"蓝线路线捕获成功，折线点数={len(route_polyline)}")
        if close_map_after_capture:
            controller.post_click_key(27).wait()
            time.sleep(1.0)

        pending_speech = "路线已捕获，开始导航"
        last_speech_time = 0.0
        if voice_enabled and os.name == "nt":
            safe_text = pending_speech.replace("'", "''")
            subprocess.Popen(
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
            last_speech_time = time.perf_counter()

        circle_padding = 15
        center_radius = 11
        min_matches = 8
        min_inliers = 4
        ratio_thresh = 0.8
        ransac_thresh = 12.0

        last_center = None
        pending_center = None
        pending_count = 0
        angle = None
        route_ready = False
        route_distances = []
        route_total_length = 0.0
        turn_events = []
        next_turn_idx = 0
        spoken_turn_far = set()
        spoken_turn_now = set()
        current_segment_idx = 0
        last_progress_along = 0.0
        last_offroute_speech = 0.0
        offroute_active = False
        arrival_spoken = False

        try:
            while True:
                if context.tasker.stopping:
                    break

                loop_start = time.perf_counter()
                img_original = controller.post_screencap().wait().get()
                if img_original is None:
                    time.sleep(0.05)
                    continue

                x, y, w_roi, h_roi = [int(v) for v in mini_map_roi]
                minimap = img_original[y:y + h_roi, x:x + w_roi].copy()
                mh, mw = minimap.shape[:2]
                center = (mw // 2, mh // 2)
                radius = max(1, min(mw, mh) // 2 - circle_padding)
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
                cv2.circle(masked, center, center_radius, (0, 0, 0), -1)

                mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                kp_mini, des_mini = sift.detectAndCompute(mini_gray, None)

                player_point = None
                raw_player_point = None
                good_mini_matches = []
                inliers = 0
                if des_mini is not None and len(kp_mini) >= min_matches:
                    for pair in matcher.knnMatch(des_mini, des_big, k=2):
                        if len(pair) < 2:
                            continue
                        m, n = pair
                        if m.distance < ratio_thresh * n.distance:
                            good_mini_matches.append(m)

                    if len(good_mini_matches) >= min_matches:
                        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_mini_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([big_points[m.trainIdx] for m in good_mini_matches]).reshape(-1, 1, 2)
                        mini_affine, mini_inlier_mask = cv2.estimateAffinePartial2D(
                            src_pts,
                            dst_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=ransac_thresh,
                        )
                        if mini_affine is not None and mini_inlier_mask is not None:
                            inliers = int(mini_inlier_mask.sum())
                            if inliers >= min_inliers:
                                player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                                player_dst = cv2.transform(player_src, mini_affine)[0, 0]
                                player_point = (
                                    int(player_dst[0] / big_map_scale),
                                    int(player_dst[1] / big_map_scale),
                                )
                                raw_player_point = player_point

                if player_point is not None:
                    accept_point = True
                    px, py = player_point
                    if px < 0 or py < 0 or px >= origin_w or py >= origin_h:
                        accept_point = False

                    if accept_point and last_center is not None:
                        jump = math.hypot(player_point[0] - last_center[0], player_point[1] - last_center[1])
                        max_jump = 60
                        if inliers >= 8 or len(good_mini_matches) >= 14:
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
                                accept_point = False
                            else:
                                pending_center = None
                                pending_count = 0
                        else:
                            pending_center = None
                            pending_count = 0

                    if accept_point:
                        if last_center is not None:
                            player_point = (
                                int(last_center[0] * (1.0 - position_alpha) + player_point[0] * position_alpha),
                                int(last_center[1] * (1.0 - position_alpha) + player_point[1] * position_alpha),
                            )
                        last_center = player_point
                    else:
                        player_point = last_center
                else:
                    player_point = last_center

                px, py, pw, ph = [int(v) for v in pointer_roi]
                pointer_crop_rgb = img_original[py:py + ph, px:px + pw, ::-1].copy()
                pointer_input = pointer_crop_rgb / 255.0
                pointer_input = pointer_input.transpose(2, 0, 1).astype(np.float32)
                pointer_input = np.expand_dims(pointer_input, axis=0)
                output = session.run(None, {input_name: pointer_input})[0][0]
                confidence = output[:, 4]
                best_idx = int(np.argmax(confidence))
                best_pred = output[best_idx]
                max_conf = float(confidence[best_idx])
                if max_conf > 0.5:
                    kpts = best_pred[6:].reshape(3, 3)
                    tip = kpts[0][:2]
                    left = kpts[1][:2]
                    right = kpts[2][:2]
                    tail_center = (left + right) / 2
                    dx = tip[0] - tail_center[0]
                    dy = tip[1] - tail_center[1]
                    raw_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
                    if angle is None:
                        angle = raw_angle
                    else:
                        angle = (angle + 0.25 * ((raw_angle - angle + 180.0) % 360.0 - 180.0)) % 360.0

                pending_speech = None
                pending_turn_far_key = None
                pending_turn_now_idx = None
                if player_point is not None and not route_ready:
                    first_dist = math.hypot(player_point[0] - route_polyline[0][0], player_point[1] - route_polyline[0][1])
                    last_dist = math.hypot(player_point[0] - route_polyline[-1][0], player_point[1] - route_polyline[-1][1])
                    if last_dist < first_dist:
                        route_polyline.reverse()

                    route_distances = [0.0]
                    for idx in range(1, len(route_polyline)):
                        route_distances.append(
                            route_distances[-1]
                            + math.hypot(
                                route_polyline[idx][0] - route_polyline[idx - 1][0],
                                route_polyline[idx][1] - route_polyline[idx - 1][1],
                            )
                        )
                    route_total_length = route_distances[-1]

                    last_turn_along = -99999.0
                    turn_event_lead_px = turn_event_lead_m / max(0.001, meters_per_pixel)
                    for idx in range(2, len(route_polyline) - 2):
                        prev_pt = route_polyline[idx - 2]
                        curr_pt = route_polyline[idx]
                        next_pt = route_polyline[idx + 2]
                        heading_a = math.degrees(math.atan2(curr_pt[0] - prev_pt[0], -(curr_pt[1] - prev_pt[1]))) % 360.0
                        heading_b = math.degrees(math.atan2(next_pt[0] - curr_pt[0], -(next_pt[1] - curr_pt[1]))) % 360.0
                        turn_delta = (heading_b - heading_a + 180.0) % 360.0 - 180.0
                        if abs(turn_delta) >= turn_angle_threshold:
                            event_along = max(0.0, route_distances[idx] - turn_event_lead_px)
                            if event_along - last_turn_along >= turn_merge_distance:
                                turn_events.append(
                                    {
                                        "idx": idx,
                                        "along": event_along,
                                        "direction": "右" if turn_delta > 0 else "左",
                                        "angle": abs(turn_delta),
                                    }
                                )
                                last_turn_along = event_along
                            elif turn_events and abs(turn_delta) > turn_events[-1]["angle"]:
                                turn_events[-1] = {
                                    "idx": idx,
                                    "along": event_along,
                                    "direction": "右" if turn_delta > 0 else "左",
                                    "angle": abs(turn_delta),
                                }
                                last_turn_along = event_along

                    route_ready = True
                    pending_speech = f"导航开始，全程约 {max(1, int(round(route_total_length * meters_per_pixel)))} 米"
                    logger.info(f"导航路线初始化完成，总长={route_total_length * meters_per_pixel:.1f}m，转弯点={len(turn_events)}")

                if route_ready and player_point is not None:
                    best_score = float("inf")
                    best_distance = float("inf")
                    best_along = last_progress_along
                    best_segment_idx = current_segment_idx
                    search_start = max(0, current_segment_idx - 5)
                    for idx in range(search_start, len(route_polyline) - 1):
                        x1, y1 = route_polyline[idx]
                        x2, y2 = route_polyline[idx + 1]
                        seg_dx = x2 - x1
                        seg_dy = y2 - y1
                        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
                        if seg_len_sq <= 1e-6:
                            continue
                        t = ((player_point[0] - x1) * seg_dx + (player_point[1] - y1) * seg_dy) / seg_len_sq
                        t = max(0.0, min(1.0, t))
                        proj_x = x1 + seg_dx * t
                        proj_y = y1 + seg_dy * t
                        distance_to_segment = math.hypot(player_point[0] - proj_x, player_point[1] - proj_y)
                        segment_along = route_distances[idx] + math.sqrt(seg_len_sq) * t
                        if segment_along + offroute_distance < last_progress_along:
                            continue
                        backtrack_penalty = max(0.0, last_progress_along - segment_along) * progress_backtrack_penalty
                        forward_delta = max(0.0, segment_along - last_progress_along)
                        forward_bonus = min(forward_delta, offroute_distance * 2.0) * progress_forward_bonus
                        score = distance_to_segment + backtrack_penalty - forward_bonus
                        if score < best_score:
                            best_score = score
                            best_distance = distance_to_segment
                            best_along = segment_along
                            best_segment_idx = idx

                    current_segment_idx = best_segment_idx
                    if best_along > last_progress_along:
                        last_progress_along = best_along

                    turn_pass_grace_px = turn_pass_grace_m / max(0.001, meters_per_pixel)
                    while next_turn_idx < len(turn_events) and turn_events[next_turn_idx]["along"] < last_progress_along - turn_pass_grace_px:
                        for announce_distance_m in turn_announce_distances_m:
                            spoken_turn_far.add((next_turn_idx, announce_distance_m))
                        spoken_turn_now.add(next_turn_idx)
                        next_turn_idx += 1

                    distance_to_end_m = max(0.0, route_total_length - last_progress_along) * meters_per_pixel
                    now = time.perf_counter()
                    if best_distance > offroute_distance:
                        if now - last_offroute_speech > 8.0:
                            pending_speech = "您已偏离路线"
                            last_offroute_speech = now
                        offroute_active = True
                    else:
                        if offroute_active:
                            pending_speech = "已回到路线"
                        offroute_active = False

                    if distance_to_end_m <= arrival_distance_m and not arrival_spoken:
                        pending_speech = "已到达目标点附近，缺德导航结束"
                        arrival_spoken = True
                        break

                    elif next_turn_idx < len(turn_events):
                        turn_event = turn_events[next_turn_idx]
                        remain_m = (turn_event["along"] - last_progress_along) * meters_per_pixel
                        if 0 < remain_m <= turn_now_distance_m and next_turn_idx not in spoken_turn_now:
                            pending_speech = f"现在{turn_event['direction']}转"
                            pending_turn_now_idx = next_turn_idx
                        else:
                            pending_threshold_m = None
                            for announce_distance_m in turn_announce_distances_m:
                                announce_key = (next_turn_idx, announce_distance_m)
                                if announce_key in spoken_turn_far:
                                    continue
                                if 0 < remain_m <= announce_distance_m and announce_distance_m - remain_m <= turn_announce_late_tolerance_m:
                                    pending_threshold_m = announce_distance_m
                                    break
                                if remain_m < announce_distance_m - turn_announce_late_tolerance_m:
                                    spoken_turn_far.add(announce_key)
                            if pending_threshold_m is not None:
                                pending_speech = f"前方 {int(round(pending_threshold_m))} 米{turn_event['direction']}转"
                                pending_turn_far_key = (next_turn_idx, pending_threshold_m)

                if pending_speech and voice_enabled:
                    if time.perf_counter() - last_speech_time >= speech_cooldown:
                        logger.info(f"语音播报: {pending_speech}")
                        if os.name == "nt":
                            safe_text = pending_speech.replace("'", "''")
                            subprocess.Popen(
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
                        if pending_turn_far_key is not None:
                            spoken_turn_far.add(pending_turn_far_key)
                        if pending_turn_now_idx is not None:
                            spoken_turn_now.add(pending_turn_now_idx)
                        last_speech_time = time.perf_counter()

                if debug_window:
                    map_view = big_map.copy()
                    for idx in range(len(route_polyline) - 1):
                        pt1 = (
                            int(route_polyline[idx][0] * big_map_scale),
                            int(route_polyline[idx][1] * big_map_scale),
                        )
                        pt2 = (
                            int(route_polyline[idx + 1][0] * big_map_scale),
                            int(route_polyline[idx + 1][1] * big_map_scale),
                        )
                        cv2.line(map_view, pt1, pt2, (255, 0, 0), 2)
                    if route_ready:
                        for event in turn_events:
                            event_pt = route_polyline[event["idx"]]
                            draw_pt = (
                                int(event_pt[0] * big_map_scale),
                                int(event_pt[1] * big_map_scale),
                            )
                            cv2.circle(map_view, draw_pt, 7, (0, 255, 255), -1)
                    if player_point is not None:
                        draw_player = (
                            int(player_point[0] * big_map_scale),
                            int(player_point[1] * big_map_scale),
                        )
                        cv2.circle(map_view, draw_player, 10, (0, 0, 255), -1)
                        if angle is not None:
                            rad = math.radians(angle)
                            draw_head = (
                                int(draw_player[0] + 28 * math.sin(rad)),
                                int(draw_player[1] - 28 * math.cos(rad)),
                            )
                            cv2.line(map_view, draw_player, draw_head, (0, 255, 0), 2)
                    debug_map_width = 900
                    if map_view.shape[1] > debug_map_width:
                        scale = debug_map_width / map_view.shape[1]
                        map_view = cv2.resize(
                            map_view,
                            (debug_map_width, int(map_view.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow("Blue Route Voice Navigation", map_view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                sleep_time = frame_interval - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            logger.exception(f"语音导航运行异常: {exc}")
            cv2.destroyAllWindows()
            return CustomAction.RunResult(success=False)

        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)
