import json
import math
import time

from pathlib import Path

import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("map_locator")
class MapLocator(CustomAction):
    abs_path = Path(__file__).parents[3]
    map_name = "map.jpg"
    if Path.exists(abs_path / "assets"):
        default_big_map = abs_path / f"assets/resource/base/image/map/{map_name}"
    else:
        default_big_map = abs_path / f"map/resource/base/image/{map_name}"

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Map Locator Test Started ===")
        controller = context.tasker.controller

        big_map_path = self.default_big_map
        mini_map_roi = [24, 14, 159, 157]
        frame_interval = 0.1
        nfeatures = 0
        ratio_thresh = 0.8
        min_matches = 8
        min_inliers = 4
        ransac_thresh = 12.0
        circle_padding = 15
        center_radius = 11
        debug_map_width = 900
        max_processing_long_side = 6144

        if not big_map_path.exists():
            print(f"[map_locator_test] 大地图不存在: {big_map_path}")
            return CustomAction.RunResult(success=False)

        big_map = cv2.imread(str(self.default_big_map), cv2.IMREAD_COLOR)
        if big_map is None:
            print(f"[map_locator_test] 大地图读取失败: {big_map_path}")
            return CustomAction.RunResult(success=False)

        origin_h, origin_w = big_map.shape[:2]
        big_map_scale = min(1.0, max_processing_long_side / max(origin_h, origin_w))
        if big_map_scale < 1.0:
            big_map = cv2.resize(
                big_map,
                (int(origin_w * big_map_scale), int(origin_h * big_map_scale)),
                interpolation=cv2.INTER_AREA,
            )

        big_map = cv2.convertScaleAbs(big_map, alpha=2.5, beta=-20)

        sift = cv2.SIFT_create(nfeatures=nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        big_gray = cv2.cvtColor(big_map, cv2.COLOR_BGR2GRAY)
        cache_path = big_map_path.with_name(f"{big_map_path.stem}.sift_cache.npz")
        cache_meta = {
            "map_size": int(big_map_path.stat().st_size),
            "map_mtime_ns": big_map_path.stat().st_mtime_ns,
            "origin_w": origin_w,
            "origin_h": origin_h,
            "proc_w": big_map.shape[1],
            "proc_h": big_map.shape[0],
            "scale": big_map_scale,
        }
        kp_big = None
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
            except Exception as e:
                print(f"[map_locator_test] 特征缓存读取失败: {e}")

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
                except Exception as e:
                    print(f"[map_locator_test] 特征缓存保存失败: {e}")

        if des_big is None or len(big_points) < min_matches:
            print("[map_locator_test] 大地图特征点不足")
            return CustomAction.RunResult(success=False)

        print(f"[map_locator_test] big map keypoints: {len(big_points)}")
        print("[map_locator_test] press Q to quit")

        last_center = None
        pending_center = None
        pending_count = 0

        while True:
            if context.tasker.stopping:
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
                knn_matches = matcher.knnMatch(des_mini, des_big, k=2)
                for pair in knn_matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

                if len(good_matches) >= min_matches:
                    src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([big_points[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
                    if M is not None and mask is not None:
                        inliers = int(mask.sum())
                        if inliers >= min_inliers:
                            corners = np.float32([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]]).reshape(-1, 1, 2)
                            polygon = cv2.transform(corners, M)
                            
                            player_src = np.float32([[mw * 0.5, mh * 0.5]]).reshape(-1, 1, 2)
                            player_dst = cv2.transform(player_src, M)[0, 0]
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
                                f"[map_locator_test] reject jump raw={player_point} last={last_center} jump={jump:.1f} inliers={inliers}"
                            )
                            accept_point = False
                        else:
                            print(
                                f"[map_locator_test] accept delayed jump raw={player_point} last={last_center} jump={jump:.1f}"
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

            if player_point is None:
                player_point = last_center

            map_view = big_map.copy()
            if polygon is not None:
                cv2.polylines(map_view, [np.int32(polygon)], True, (0, 255, 0), 3)
            if player_point is not None:
                draw_player_point = (
                    int(player_point[0] * big_map_scale),
                    int(player_point[1] * big_map_scale),
                )
                cv2.circle(map_view, draw_player_point, 16, (0, 0, 255), -1)
                # cv2.putText(
                #     map_view,
                #     f"player=({player_point[0]}, {player_point[1]})",
                #     (max(0, draw_player_point[0] + 12), max(25, draw_player_point[1] - 12)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     5,
                #     (255, 255, 255),
                #     2,
                #     cv2.LINE_AA,
                # )

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

            cv2.imshow("map_locator_test", np.concatenate([mini_view, map_view], axis=1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sleep_time = frame_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        cv2.destroyAllWindows()
        return CustomAction.RunResult(success=True)
