import cv2
import json
import math
import os
import time
import heapq
import numpy as np
import onnxruntime

from pathlib import Path
from .Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


logger = get_logger(__name__)

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def a_star_search(grid, start, goal):
    height, width = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if math.hypot(current[0] - goal[0], current[1] - goal[1]) <= 8:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny, nx] < 128:
                    continue
                
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
                    
    return []


def has_line_of_sight(grid, start, end, walkable_threshold=128):
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))

    if steps == 0:
        if 0 <= x0 < grid.shape[1] and 0 <= y0 < grid.shape[0]:
            return grid[y0, x0] >= walkable_threshold
        return False

    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        if x < 0 or y < 0 or x >= grid.shape[1] or y >= grid.shape[0]:
            return False
        if grid[y, x] < walkable_threshold:
            return False
    return True


def simplify_path_by_visibility(grid, path):
    if len(path) <= 2:
        return list(path)

    simplified = [path[0]]
    anchor_idx = 0
    probe_idx = 1

    while probe_idx < len(path):
        if has_line_of_sight(grid, path[anchor_idx], path[probe_idx]):
            probe_idx += 1
            continue

        simplified.append(path[probe_idx - 1])
        anchor_idx = probe_idx - 1

    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified


def smooth_path_points(path, grid, smoothing=0.25, passes=2):
    if len(path) <= 2 or smoothing <= 0:
        return list(path)

    smoothed = [(float(x), float(y)) for x, y in path]
    smoothing = clamp(float(smoothing), 0.0, 0.45)
    passes = max(1, int(passes))

    for _ in range(passes):
        updated = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            prev_pt = smoothed[i - 1]
            curr_pt = smoothed[i]
            next_pt = smoothed[i + 1]
            candidate = (
                curr_pt[0] * (1.0 - 2.0 * smoothing) + (prev_pt[0] + next_pt[0]) * smoothing,
                curr_pt[1] * (1.0 - 2.0 * smoothing) + (prev_pt[1] + next_pt[1]) * smoothing,
            )

            candidate_int = (int(round(candidate[0])), int(round(candidate[1])))
            prev_int = (int(round(updated[-1][0])), int(round(updated[-1][1])))
            next_int = (int(round(next_pt[0])), int(round(next_pt[1])))

            if (
                0 <= candidate_int[0] < grid.shape[1]
                and 0 <= candidate_int[1] < grid.shape[0]
                and grid[candidate_int[1], candidate_int[0]] >= 128
                and has_line_of_sight(grid, prev_int, candidate_int)
                and has_line_of_sight(grid, candidate_int, next_int)
            ):
                updated.append(candidate)
            else:
                updated.append(curr_pt)

        updated.append(smoothed[-1])
        smoothed = updated

    return [(int(round(x)), int(round(y))) for x, y in smoothed]


def resample_path(path, spacing):
    if len(path) <= 1 or spacing <= 1:
        return list(path)

    spacing = float(spacing)
    resampled = [path[0]]
    carry = 0.0

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]
        dx = x1 - x0
        dy = y1 - y0
        segment_length = math.hypot(dx, dy)

        if segment_length < 1e-6:
            continue

        distance_along = spacing - carry
        while distance_along <= segment_length:
            t = distance_along / segment_length
            point = (int(round(x0 + dx * t)), int(round(y0 + dy * t)))
            if point != resampled[-1]:
                resampled.append(point)
            distance_along += spacing

        carry = max(0.0, segment_length - (distance_along - spacing))

    if resampled[-1] != path[-1]:
        resampled.append(path[-1])
    return resampled


def build_navigation_path(path, grid, spacing=18, smoothing=0.2, smoothing_passes=2):
    if not path:
        return []

    simplified = simplify_path_by_visibility(grid, path)
    smoothed = smooth_path_points(simplified, grid, smoothing=smoothing, passes=smoothing_passes)
    return resample_path(smoothed, spacing)


def point_to_segment_distance(point, seg_start, seg_end):
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_polyline(point, polyline, start_idx=0):
    if len(polyline) < 2:
        return math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1]) if polyline else float("inf")

    start_idx = max(0, min(start_idx, len(polyline) - 2))
    best = float("inf")
    for idx in range(start_idx, len(polyline) - 1):
        best = min(best, point_to_segment_distance(point, polyline[idx], polyline[idx + 1]))
    return best


@AgentServer.custom_action("combined_auto_navigate")
class CombinedAutoNavigate(CustomAction):

    def _resolve_backend(self, custom_action_param: str) -> str:
        backend = os.environ.get("MAA_ONNX_BACKEND", "cpu")
        if custom_action_param:
            try:
                params = json.loads(custom_action_param)
                if isinstance(params, dict) and params.get("backend"):
                    backend = str(params["backend"])
            except:
                pass
        
        backend = backend.strip().lower()
        if backend == "auto":
            available = onnxruntime.get_available_providers()
            if "DmlExecutionProvider" in available:
                return "directml"
            return "cpu"
        
        _provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }

        if backend not in _provider_name_map:
            logger.warning(f"未知推理后端 {backend}，将回退到 CPU")
            return "cpu"
        return backend

    def _get_session(self, model_path: Path, backend: str):
        _provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }
        provider_name = _provider_name_map.get(backend, "CPUExecutionProvider")
        available = onnxruntime.get_available_providers()

        if provider_name not in available:
            provider_name = "CPUExecutionProvider"

        session_options = onnxruntime.SessionOptions()
        providers = [provider_name]
        provider_options = None

        if provider_name == "DmlExecutionProvider":
            provider_options = [{"device_id": 0}]

        if provider_options is None:
            session = onnxruntime.InferenceSession(
                str(model_path), sess_options=session_options, providers=providers
            )
        else:
            session = onnxruntime.InferenceSession(
                str(model_path), sess_options=session_options, providers=providers, provider_options=provider_options
            )
        return session, provider_name

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger.info("=== Combined Auto Navigate Action Started ===")
        controller = context.tasker.controller

        # 键盘按键值定义 (参考 autofish)
        KEY_W = 87
        KEY_A = 65
        KEY_D = 68
        KEY_S = 83

        params = {}
        try:
            if argv.custom_action_param:
                loaded_params = json.loads(argv.custom_action_param)
                if isinstance(loaded_params, dict):
                    params = loaded_params
        except Exception as exc:
            logger.warning(f"解析 custom_action_param 失败，将使用默认参数: {exc}")

        def get_param(name, default, cast=float):
            value = params.get(name, default)
            try:
                return cast(value)
            except Exception:
                return default

        target_x = params.get("target_x")
        target_y = params.get("target_y")
            
        if target_x is None or target_y is None:
            logger.error("未在参数中指定终点坐标! (需要 target_x 和 target_y)")
            return CustomAction.RunResult(success=False)

        nav_cfg = {
            "path_spacing": max(8.0, get_param("path_spacing", 18.0, float)),
            "path_smoothing": clamp(get_param("path_smoothing", 0.18, float), 0.0, 0.4),
            "path_smoothing_passes": max(1, get_param("path_smoothing_passes", 2, int)),
            "lookahead_min": max(12.0, get_param("lookahead_min", 30, float)),
            "lookahead_max": max(20.0, get_param("lookahead_max", 60, float)),
            "arrival_radius": max(6.0, get_param("arrival_radius", 18.0, float)),
            "replan_distance": max(20.0, get_param("replan_distance", 70.0, float)),
            "forward_turn_limit": clamp(get_param("forward_turn_limit", 42.0, float), 15.0, 90.0),
            "turn_deadband": clamp(get_param("turn_deadband", 9, float), 1.0, 30.0),
            "turn_hysteresis": clamp(get_param("turn_hysteresis", 3.0, float), 0.5, 20.0),
            "hard_turn_angle": clamp(get_param("hard_turn_angle", 72.0, float), 25.0, 140.0),
            "kp": clamp(get_param("kp", 0.78, float), 0.0, 5.0),
            "kd": clamp(get_param("kd", 0.08, float), 0.0, 2.0),
            "lateral_gain": clamp(get_param("lateral_gain", 0.55, float), 0.0, 3.0),
            "derivative_alpha": clamp(get_param("derivative_alpha", 0.25, float), 0.0, 0.95),
            "angle_alpha": clamp(get_param("angle_alpha", 0.22, float), 0.0, 1.0),
            "position_alpha": clamp(get_param("position_alpha", 0.35, float), 0.0, 1.0),
        }
        nav_cfg["lookahead_max"] = max(nav_cfg["lookahead_max"], nav_cfg["lookahead_min"] + 4.0)

        logger.info(
            "导航参数: "
            f"spacing={nav_cfg['path_spacing']:.1f}, "
            f"lookahead=({nav_cfg['lookahead_min']:.1f},{nav_cfg['lookahead_max']:.1f}), "
            f"deadband={nav_cfg['turn_deadband']:.1f}, "
            f"kp={nav_cfg['kp']:.2f}, kd={nav_cfg['kd']:.2f}, lateral={nav_cfg['lateral_gain']:.2f}"
        )

        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            base_dir = abs_path / "assets/resource/base"
        else:
            base_dir = abs_path / "resource/base"

        map_path = base_dir / "image/map/map.jpg"
        npz_path = base_dir / "image/map/map_data.npz"
        model_path = base_dir / "model/navi/pointer_model.onnx"

        ########################
        # 1. 准备 SIFT 和 大地图
        ########################
        if not map_path.exists():
            logger.error("地图文件 map.jpg 不存在")
            return CustomAction.RunResult(success=False)
            
        big_map = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if big_map is None:
            logger.error("地图读取失败")
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
        sift = cv2.SIFT_create(nfeatures=0)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        big_gray = cv2.cvtColor(big_map, cv2.COLOR_BGR2GRAY)
        
        cache_path = map_path.with_name(f"{map_path.stem}.sift_cache.npz")
        kp_big, des_big = None, None
        big_points = None
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    big_points = cache["keypoints"].astype(np.float32, copy=False)
                    des_big = cache["descriptors"]
            except Exception as e:
                logger.warning(f"缓存特征读取失败: {e}")
                
        if big_points is None or des_big is None:
            kp_big, des_big = sift.detectAndCompute(big_gray, None)
            if des_big is not None:
                big_points = np.float32([kp.pt for kp in kp_big])
                
        if des_big is None:
            logger.error("大地图特征提取失败")
            return CustomAction.RunResult(success=False)

        ########################
        # 2. 准备 ONNX Session
        ########################
        if not model_path.exists():
            logger.error("ONNX 模型不存在")
            return CustomAction.RunResult(success=False)
        backend = self._resolve_backend(argv.custom_action_param)
        session, provider_name = self._get_session(model_path, backend)
        input_name = session.get_inputs()[0].name

        ########################
        # 3. 准备二值化通行地图
        ########################
        if not npz_path.exists():
            logger.error("二值化通行地图缺失 (map_data.npz)")
            return CustomAction.RunResult(success=False)
        try:
            data = np.load(str(npz_path))
            key = data.files[0]
            grid_map = data[key]
        except Exception as e:
            logger.error("由于异常，读取二值化地图失败")
            return CustomAction.RunResult(success=False)

        ########################
        # 4. 主循环控制
        ########################
        mini_map_roi = [24, 14, 159, 157]
        pointer_roi = [73, 60, 64, 64]
        
        circle_padding = 15
        center_radius = 11

        last_center = None
        angle = None
        
        current_w = False
        current_a = False
        current_d = False

        def update_keys(w, a, d):
            nonlocal current_w, current_a, current_d
            if w != current_w:
                if w: controller.post_key_down(KEY_W)
                else: controller.post_key_up(KEY_W)
                current_w = w
            if a != current_a:
                if a: controller.post_key_down(KEY_A)
                else: controller.post_key_up(KEY_A)
                current_a = a
            if d != current_d:
                if d: controller.post_key_down(KEY_D)
                else: controller.post_key_up(KEY_D)
                current_d = d

        def release_all():
            update_keys(False, False, False)

        waypoints = []
        waypoint_idx = 0
        
        # 定义平滑与控制需要用的状态变量
        last_angle = None
        angle = None
        prev_error = 0.0
        smoothed_d_error = 0.0
        last_time = time.perf_counter()
        current_path = []
        
        # 记录当前转弯按键状态，用于加入阻带滞后（Hysteresis），避免按键高频抖动
        is_turning_right = False
        is_turning_left = False
        
        logger.info(f"开启自动寻路至终点: ({target_x}, {target_y})")

        try:
            while True:
                if context.tasker.stopping:
                    release_all()
                    break

                img_original = controller.post_screencap().wait().get()
                if img_original is None:
                    time.sleep(0.01)
                    continue
                
                # ==== 定位解析 (融合 map_locator.py) ====
                x, y, w_roi, h_roi = mini_map_roi
                minimap = img_original[y:y+h_roi, x:x+w_roi].copy()
                mh, mw = minimap.shape[:2]

                center = (mw // 2, mh // 2)
                radius = max(1, min(mw, mh) // 2 - circle_padding)
                circle_mask = np.zeros((mh, mw), dtype=np.uint8)
                cv2.circle(circle_mask, center, radius, 255, -1)

                hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
                lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
                upper_hsv = np.array([179, 66, 80], dtype=np.uint8)
                hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

                final_mask = cv2.bitwise_and(circle_mask, hsv_mask)
                masked = cv2.bitwise_and(minimap, minimap, mask=final_mask)
                masked = cv2.convertScaleAbs(masked, alpha=3.8, beta=-40)
                cv2.circle(masked, center, center_radius, (0, 0, 0), -1)

                mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                kp_mini, des_mini = sift.detectAndCompute(mini_gray, None)

                player_point = None
                if des_mini is not None and len(kp_mini) >= 8:
                    knn_matches = matcher.knnMatch(des_mini, des_big, k=2)
                    good_matches = []
                    for pair in knn_matches:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < 0.8 * n.distance:
                                good_matches.append(m)

                    if len(good_matches) >= 8:
                        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([big_points[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
                        
                        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=12.0)
                        if M is not None and mask is not None and int(mask.sum()) >= 4:
                            player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                            player_dst = cv2.transform(player_src, M)[0, 0]
                            player_point = (
                                int(player_dst[0] / big_map_scale),
                                int(player_dst[1] / big_map_scale),
                            )

                if player_point is not None:
                    if last_center is not None:
                        # 简单的坐标平滑计算
                        player_point = (
                            int(last_center[0] * (1.0 - nav_cfg["position_alpha"]) + player_point[0] * nav_cfg["position_alpha"]),
                            int(last_center[1] * (1.0 - nav_cfg["position_alpha"]) + player_point[1] * nav_cfg["position_alpha"]),
                        )
                    last_center = player_point
                else:
                    player_point = last_center

                # ==== 方向解析 (融合 predict_angle.py) ====
                px, py, pw, ph = pointer_roi
                img_crop_rgb = img_original[py:py+ph, px:px+pw, ::-1].copy() # 转换为RGB传入模型
                img_input = img_crop_rgb / 255.0    
                img_input = img_input.transpose(2, 0, 1).astype(np.float32) 
                img_input = np.expand_dims(img_input, axis=0)  

                result = session.run(None, {input_name: img_input})
                output = result[0][0]
                confidence = output[:, 4]
                best_idx = np.argmax(confidence)
                best_pred = output[best_idx]
                max_conf = confidence[best_idx]

                if max_conf > 0.5:            
                    kpts = best_pred[6:].reshape(3, 3)
                    tip = kpts[0][:2]    
                    left = kpts[1][:2]   
                    right = kpts[2][:2]  
                    tail_center = (left + right) / 2
                    dx = tip[0] - tail_center[0]
                    dy = tip[1] - tail_center[1]
                    raw_angle = math.degrees(math.atan2(dx, -dy)) % 360
                    
                    if last_angle is None:
                        angle = raw_angle
                        last_angle = raw_angle
                    else:
                        # 处理 360 度循环平滑 (Exponential Moving Average)
                        angle_diff_smooth = (raw_angle - angle + 180) % 360 - 180
                        angle = (angle + nav_cfg["angle_alpha"] * angle_diff_smooth) % 360
                    last_angle = angle
                        
                # ==== 连同 A* 寻路与载具运动控制 ====
                if player_point is not None and angle is not None:
                    logger.debug(player_point)
                    # 获取路线
                    if len(waypoints) == 0:
                        path_pts = a_star_search(grid_map, player_point, (target_x, target_y))
                        if not path_pts:
                            logger.error("无法找到到达终点的无障碍路径")
                            release_all()
                            return CustomAction.RunResult(success=False)

                        current_path = [player_point] + path_pts
                        waypoints = build_navigation_path(
                            current_path,
                            grid_map,
                            spacing=nav_cfg["path_spacing"],
                            smoothing=nav_cfg["path_smoothing"],
                            smoothing_passes=nav_cfg["path_smoothing_passes"],
                        )
                        if len(waypoints) < 2:
                            waypoints = current_path
                        waypoint_idx = 0
                        prev_error = 0.0
                        smoothed_d_error = 0.0
                        is_turning_right = False
                        is_turning_left = False
                        last_time = time.perf_counter()
                        logger.info(
                            f"路径规划成功，A*原始节点 {len(current_path)} 个，平滑后关键节点 {len(waypoints)} 个"
                        )
                    
                    if waypoint_idx < len(waypoints):
                        if len(waypoints) >= 2:
                            path_offset = distance_to_polyline(player_point, waypoints, start_idx=max(0, waypoint_idx - 1))
                            if path_offset > nav_cfg["replan_distance"]:
                                logger.info(
                                    f"偏离路径 {path_offset:.1f}px，触发重规划"
                                )
                                waypoints = []
                                current_path = []
                                release_all()
                                time.sleep(0.02)
                                continue

                        # ==== 前瞻点与平滑寻路 (Pure Pursuit + 横向误差修正) ====
                        next_idx = min(waypoint_idx + 1, len(waypoints) - 1)
                        segment_dx = waypoints[next_idx][0] - waypoints[waypoint_idx][0]
                        segment_dy = waypoints[next_idx][1] - waypoints[waypoint_idx][1]
                        segment_angle = math.degrees(math.atan2(segment_dx, -segment_dy)) % 360
                        heading_error_for_lookahead = abs(normalize_angle_deg(segment_angle - angle))
                        lookahead_distance = clamp(
                            nav_cfg["lookahead_max"] - 0.35 * heading_error_for_lookahead,
                            nav_cfg["lookahead_min"],
                            nav_cfg["lookahead_max"],
                        )

                        while waypoint_idx < len(waypoints) - 1:
                            check_pt = waypoints[waypoint_idx]
                            dist_to_check = math.hypot(check_pt[0] - player_point[0], check_pt[1] - player_point[1])
                            if dist_to_check < lookahead_distance:
                                waypoint_idx += 1
                            else:
                                break

                        target_pt = waypoints[waypoint_idx]
                        dist_dx = target_pt[0] - player_point[0]
                        dist_dy = target_pt[1] - player_point[1]
                        dist = math.hypot(dist_dx, dist_dy)

                        # 如果最终一个点也在这附近，那就认为结束了
                        if dist <= nav_cfg["arrival_radius"] and waypoint_idx == len(waypoints) - 1:
                            waypoint_idx += 1
                        else:
                            desired_angle = math.degrees(math.atan2(dist_dx, -dist_dy)) % 360
                            heading_error = normalize_angle_deg(desired_angle - angle)

                            rad = math.radians(angle)
                            forward_error = dist_dx * math.sin(rad) - dist_dy * math.cos(rad)
                            lateral_error = dist_dx * math.cos(rad) + dist_dy * math.sin(rad)
                            lateral_correction = math.degrees(
                                math.atan2(nav_cfg["lateral_gain"] * lateral_error, max(12.0, abs(forward_error)))
                            )
                            angle_diff = normalize_angle_deg(heading_error + lateral_correction)

                            current_time = time.perf_counter()
                            dt = current_time - last_time
                            if dt < 0.001: dt = 0.001
                            last_time = current_time

                            # ==== 改进的 PD 控制器 ====
                            Kp = 1.0
                            Kd = 0.2  # 调小阻尼的基准系数，使用 dt 计算
                            
                            # 低通滤波微分项，消除视觉模型测量的抖动噪声
                            raw_d_error = (angle_diff - prev_error) / dt
                            smoothed_d_error = (
                                nav_cfg["derivative_alpha"] * raw_d_error
                                + (1.0 - nav_cfg["derivative_alpha"]) * smoothed_d_error
                            )
                            prev_error = angle_diff
                            
                            pid_out = nav_cfg["kp"] * angle_diff + nav_cfg["kd"] * smoothed_d_error

                            turn_enter = nav_cfg["turn_deadband"]
                            turn_exit = max(0.5, turn_enter - nav_cfg["turn_hysteresis"])

                            if abs(angle_diff) >= nav_cfg["hard_turn_angle"]:
                                is_turning_right = angle_diff > 0
                                is_turning_left = angle_diff < 0
                            else:
                                if is_turning_right:
                                    if pid_out < turn_exit:
                                        is_turning_right = False
                                else:
                                    if pid_out > turn_enter:
                                        is_turning_right = True

                                if is_turning_left:
                                    if pid_out > -turn_exit:
                                        is_turning_left = False
                                else:
                                    if pid_out < -turn_enter:
                                        is_turning_left = True

                            # 互斥互锁保护
                            if is_turning_left and is_turning_right:
                                if abs(angle_diff) < turn_enter:
                                    is_turning_left = False
                                    is_turning_right = False
                                elif angle_diff > 0:
                                    is_turning_left = False
                                else:
                                    is_turning_right = False

                            move_forward = abs(angle_diff) < nav_cfg["forward_turn_limit"] or dist < lookahead_distance * 0.75
                            if abs(angle_diff) >= nav_cfg["hard_turn_angle"]:
                                move_forward = False

                            # W = 87 前进, A = 65 左转, D = 68 右转
                            update_keys(move_forward, is_turning_left, is_turning_right)
                    else:
                        logger.info("到达目标地点!")
                        release_all()
                        cv2.destroyAllWindows()
                        return CustomAction.RunResult(success=True)

                # ==== 可视化展示 ====
                map_view = big_map.copy()
                
                # 绘制规划路径
                if len(waypoints) > 0:
                    for i in range(max(0, waypoint_idx - 1), len(waypoints) - 1):
                        pt1 = (int(waypoints[i][0] * big_map_scale), int(waypoints[i][1] * big_map_scale))
                        pt2 = (int(waypoints[i+1][0] * big_map_scale), int(waypoints[i+1][1] * big_map_scale))
                        cv2.line(map_view, pt1, pt2, (255, 0, 0), 2)
                        cv2.circle(map_view, pt1, 2, (255, 255, 0), -1)
                    focus_idx = min(waypoint_idx, len(waypoints) - 1)
                    focus_pt = (int(waypoints[focus_idx][0] * big_map_scale), int(waypoints[focus_idx][1] * big_map_scale))
                    cv2.circle(map_view, focus_pt, 6, (0, 255, 0), -1)

                # 绘制当前位置坐标和朝向
                if player_point is not None:
                    draw_pt = (int(player_point[0] * big_map_scale), int(player_point[1] * big_map_scale))
                    cv2.circle(map_view, draw_pt, 8, (0, 0, 255), -1)
                    if angle is not None:
                        rad = math.radians(angle)
                        dir_pt = (int(draw_pt[0] + 20 * math.sin(rad)), int(draw_pt[1] - 20 * math.cos(rad)))
                        cv2.line(map_view, draw_pt, dir_pt, (0, 255, 255), 2)

                # 窗口预览
                debug_map_width = 800
                if map_view.shape[1] > debug_map_width:
                    scale = debug_map_width / map_view.shape[1]
                    map_view = cv2.resize(map_view, (debug_map_width, int(map_view.shape[0] * scale)))
                
                cv2.imshow("Combined Auto Navigate", map_view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户手动退出可视化预览")
                    release_all()
                    break

                time.sleep(0.01)

        except Exception as e:
            logger.error(f"发生无法预料的任务异常: {e}")
            release_all()
            cv2.destroyAllWindows()
            return CustomAction.RunResult(success=False)

        release_all()
        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)
