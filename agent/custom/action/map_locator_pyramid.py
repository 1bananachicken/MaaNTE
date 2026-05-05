import json
import math
import time

from pathlib import Path

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("map_locator_pyramid")
class MapLocatorPyramid(CustomAction):
    abs_path = Path(__file__).parents[3]
    map_name = "map.jpg"
    if Path.exists(abs_path / "assets"):
        default_big_map = abs_path / f"assets/resource/base/image/map/{map_name}"
    else:
        default_big_map = abs_path / f"resource/base/image/map/{map_name}"

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Map Locator Pyramid Started ===")
        controller = context.tasker.controller

        big_map_path = self.default_big_map
        mini_map_roi = [24, 14, 159, 157]
        frame_interval = 0.1
        nfeatures = 0
        ratio_thresh = 0.8
        min_matches = 6
        min_inliers = 4
        ransac_thresh = 12.0
        circle_padding = 15
        center_radius = 11
        debug_map_width = 900
        
        # Pyramid settings
        max_processing_long_side = 6144
        chunk_size = 2800
        chunk_overlap = 200

        if not big_map_path.exists():
            print(f"大地图不存在: {big_map_path}")
            return CustomAction.RunResult(success=False)

        # 1. 载入原始完整大图
        print(f"正在加载原始大图: {big_map_path}")
        original_map = cv2.imread(str(self.default_big_map), cv2.IMREAD_COLOR)
        if original_map is None:
            print(f"大地图读取失败: {big_map_path}")
            return CustomAction.RunResult(success=False)

        origin_h, origin_w = original_map.shape[:2]
        print(f"原始分辨率: {origin_w}x{origin_h}")

        sift = cv2.SIFT_create(nfeatures=nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        # 2. 构建全局低分辨率视图（大概坐标定位）
        global_scale = min(1.0, max_processing_long_side / max(origin_h, origin_w))
        global_map = cv2.resize(
            original_map,
            (int(origin_w * global_scale), int(origin_h * global_scale)),
            interpolation=cv2.INTER_AREA,
        )
        global_map = cv2.convertScaleAbs(global_map, alpha=2.5, beta=-20)
        global_gray = cv2.cvtColor(global_map, cv2.COLOR_BGR2GRAY)

        cache_path = big_map_path.with_name(f"{big_map_path.stem}.pyramid_global_cache.npz")
        cache_meta = {
            "map_size": int(big_map_path.stat().st_size),
            "map_mtime_ns": big_map_path.stat().st_mtime_ns,
            "origin_w": origin_w,
            "origin_h": origin_h,
            "scale": global_scale,
        }

        global_points = None
        global_des = None
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    cache_meta_raw = cache["meta"]
                    saved_meta = json.loads(cache_meta_raw.item() if hasattr(cache_meta_raw, "item") else str(cache_meta_raw))
                    if saved_meta == cache_meta:
                        global_points = cache["keypoints"].astype(np.float32, copy=False)
                        global_des = cache["descriptors"]
            except Exception as e:
                print(f"全局特征缓存读取失败: {e}")

        if global_points is None or global_des is None:
            print(" 正在计算低分辨率全局特征...")
            kp_global, global_des = sift.detectAndCompute(global_gray, None)
            if global_des is not None:
                global_points = np.float32([kp.pt for kp in kp_global])
                try:
                    np.savez_compressed(
                        cache_path,
                        meta=json.dumps(cache_meta, ensure_ascii=False),
                        keypoints=global_points,
                        descriptors=global_des,
                    )
                except Exception as e:
                    print(f"全局特征缓存保存失败: {e}")

        print(f"全局视图特征点数: {len(global_points)}")

        # 3. 初始化分块缓存信息
        chunk_cols = math.ceil(origin_w / chunk_size)
        chunk_rows = math.ceil(origin_h / chunk_size)
        chunks_cache = {}  # 结构: {(col, row): (points, des, x_offset, y_offset)}

        chunks_cache_dir = big_map_path.with_name(f"{big_map_path.stem}_chunks_cache")
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"开始预加载所有 {chunk_cols}x{chunk_rows} 个分块特征...")
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
                
                chunk_cache_path = chunks_cache_dir / f"chunk_{c}_{r}.npz"
                chunk_pts = None
                des_chunk = None
                
                if chunk_cache_path.exists():
                    try:
                        with np.load(chunk_cache_path, allow_pickle=False) as cache:
                            chunk_pts = cache["keypoints"].astype(np.float32, copy=False)
                            des_chunk = cache["descriptors"]
                            if des_chunk.size == 0:
                                des_chunk = None
                    except Exception as e:
                        pass

                if chunk_pts is None and des_chunk is None:
                    print(f"加载并计算分块 {(c, r)} (大小 {cw}x{ch})...")
                    chunk_img = original_map[cy:cy+ch, cx:cx+cw]
                    chunk_img = cv2.convertScaleAbs(chunk_img, alpha=2.5, beta=-20)
                    chunk_gray = cv2.cvtColor(chunk_img, cv2.COLOR_BGR2GRAY)
                    kp_chunk, des_chunk = sift.detectAndCompute(chunk_gray, None)
                    
                    chunk_pts = np.float32([kp.pt for kp in kp_chunk]) if kp_chunk else np.array([])
                    try:
                        np.savez_compressed(
                            chunk_cache_path,
                            keypoints=chunk_pts,
                            descriptors=des_chunk if des_chunk is not None else np.array([]),
                        )
                    except Exception as e:
                        print(f"分块缓存保存失败: {e}")
                        
                chunks_cache[(c, r)] = (chunk_pts, des_chunk, cx, cy)
        print(" 所有分块特征预加载完毕！")

        last_center = None
        pending_center = None
        pending_count = 0
        lost_frames = 0

        # 当前活跃分块
        current_chunk_idx = None  # (col, row)
        
        print(" press Q to quit")
        
        while True:
            if context.tasker.stopping:
                cv2.destroyAllWindows()
                break

            loop_start = time.perf_counter()
            img = controller.post_screencap().wait().get()

            x, y, w, h = mini_map_roi
            minimap = img[y:y + h, x:x + w]
            
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

            mini_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            kp_mini, des_mini = sift.detectAndCompute(mini_gray, None)

            polygon = None
            player_point = None
            good_matches = []
            inliers = 0
            raw_player_point = None

            if des_mini is not None and len(kp_mini) >= min_matches:
                
                # ------ 层级1: 全局大图匹配 获取大概坐标 ------
                approx_player_point = last_center
                
                # 如果没有上一帧的精确坐标，则使用全局低分辨率图匹配找大概位置
                if approx_player_point is None:
                    knn_global = matcher.knnMatch(des_mini, global_des, k=2)
                    global_good = []
                    for pair in knn_global:
                        if len(pair) < 2:
                            continue
                        m, n = pair
                        if m.distance < ratio_thresh * n.distance:
                            global_good.append(m)

                    if len(global_good) >= min_matches:
                        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in global_good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([global_points[m.trainIdx] for m in global_good]).reshape(-1, 1, 2)
                        
                        M_global, mask_global = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
                        if M_global is not None and mask_global is not None:
                            if int(mask_global.sum()) >= min_inliers:
                                player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                                player_dst = cv2.transform(player_src, M_global)[0, 0]
                                approx_player_point = (
                                    int(player_dst[0] / global_scale),
                                    int(player_dst[1] / global_scale),
                                )
                
                # ------ 层级2: 局部高清分块匹配 ------
                if approx_player_point is not None:
                    # 计算应该使用哪个切割块
                    target_col = max(0, min(chunk_cols - 1, int(approx_player_point[0] // chunk_size)))
                    target_row = max(0, min(chunk_rows - 1, int(approx_player_point[1] // chunk_size)))
                    target_chunk_idx = (target_col, target_row)
                    
                    chunks_cache_dir = big_map_path.with_name(f"{big_map_path.stem}_chunks_cache")
                    chunks_cache_dir.mkdir(parents=True, exist_ok=True)

                    # 计算目标区块及周围四个邻域的分块（上下左右）
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
                    
                    def dist_to_chunk_center(idx):
                        c, r = idx
                        center_x = c * chunk_size + chunk_size / 2
                        center_y = r * chunk_size + chunk_size / 2
                        return math.hypot(approx_player_point[0] - center_x, approx_player_point[1] - center_y)

                    sorted_neighbors = sorted(neighbors, key=dist_to_chunk_center)
                    current_chunk_idx = sorted_neighbors[0] 
                    chunk_pts, des_chunk, cx, cy = chunks_cache[current_chunk_idx]

                    if des_chunk is not None and len(chunk_pts) >= min_matches:
                        cand_matches = []
                        knn_matches = matcher.knnMatch(des_mini, des_chunk, k=2)
                        for pair in knn_matches:
                            if len(pair) < 2:
                                continue
                            m, n = pair
                            if m.distance < ratio_thresh * n.distance:
                                cand_matches.append(m)

                        if len(cand_matches) >= min_matches:
                            src_pts = np.float32([kp_mini[m.queryIdx].pt for m in cand_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([chunk_pts[m.trainIdx] for m in cand_matches]).reshape(-1, 1, 2)
                            
                            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
                            if M is not None and mask is not None:
                                good_matches = cand_matches
                                inliers = int(mask.sum())
                                
                                # 将局部坐标装换为原图全局坐标
                                corners = np.float32([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]]).reshape(-1, 1, 2)
                                polygon_local = cv2.transform(corners, M)
                                polygon = polygon_local.copy()
                                for i in range(len(polygon)):
                                    polygon[i][0][0] = (polygon[i][0][0] + cx) * global_scale
                                    polygon[i][0][1] = (polygon[i][0][1] + cy) * global_scale

                                player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                                player_dst = cv2.transform(player_src, M)[0, 0]
                                
                                # 获得原比例下在完整大图中的坐标
                                player_point = (
                                    int(player_dst[0] + cx),
                                    int(player_dst[1] + cy),
                                )
                                raw_player_point = player_point

            # ------ 过滤与追踪判定 ------
            if player_point is not None:
                accept_point = True
                px, py = player_point

                if px < 0 or py < 0 or px >= origin_w or py >= origin_h:
                    accept_point = False

                if accept_point and last_center is not None:
                    jump = math.hypot(player_point[0] - last_center[0], player_point[1] - last_center[1])
                    max_jump = 60
                    if inliers >= 8 or len(good_matches) >= 14:
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
                            print(
                                f"reject jump raw={player_point} last={last_center} jump={jump:.1f} inliers={inliers}"
                            )
                            accept_point = False
                        else:
                            print(
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

            # ------ 渲染画面 (仍使用等比例缩小的全局图) ------
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
                f"kp={0 if des_mini is None else len(kp_mini)} matches={len(good_matches)} inliers={inliers}",
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

            cv2.imshow("map_locator_pyramid", np.concatenate([mini_view, map_view], axis=1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep_time = frame_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)
