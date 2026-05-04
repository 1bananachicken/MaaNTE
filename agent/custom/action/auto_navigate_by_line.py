import cv2
import json
import math
import os
import time
import numpy as np
import onnxruntime

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from .Common.utils import get_image


@dataclass
class PointerDetection:
    angle: float
    confidence: float


@dataclass
class LineDetection:
    color_name: str
    target_point: Tuple[int, int]
    heading_error: float
    angle_min: float
    angle_max: float
    area: float
    score: float
    reliability: float
    edge_points: int


class MovementController:
    def __init__(self, controller, config: Dict[str, Any]):
        self.controller = controller
        self.config = config
        self.move_key = int(config["movement"]["forward_key"])
        self.sprint_key = config["movement"].get("sprint_key")
        self.steering_mode = config["steering_mode"]
        self.left_key = int(config["movement"]["left_key"])
        self.right_key = int(config["movement"]["right_key"])
        self.current_turn_key: Optional[int] = None
        self.move_pressed = False
        self.sprint_pressed = False

    def set_forward(self, active: bool) -> None:
        if active and not self.move_pressed:
            self.controller.post_key_down(self.move_key)
            self.move_pressed = True
        elif not active and self.move_pressed:
            self.controller.post_key_up(self.move_key)
            self.move_pressed = False

    def set_sprint(self, active: bool) -> None:
        if self.sprint_key is None:
            return

        sprint_key = int(self.sprint_key)
        if active and not self.sprint_pressed:
            self.controller.post_key_down(sprint_key)
            self.sprint_pressed = True
        elif not active and self.sprint_pressed:
            self.controller.post_key_up(sprint_key)
            self.sprint_pressed = False

    def apply_steering(self, heading_error: float, reliability: float) -> None:
        if self.steering_mode == "ad_key":
            self._apply_key_steering(heading_error)
        else:
            self._clear_turn_key()
            self._apply_mouse_steering(heading_error, reliability)

    def search_turn(self, step: int) -> None:
        if self.steering_mode == "ad_key":
            if step > 0:
                self._set_turn_key(self.right_key)
            elif step < 0:
                self._set_turn_key(self.left_key)
            else:
                self._clear_turn_key()
            return

        self._clear_turn_key()
        if step != 0:
            self.controller.post_relative_move(int(step), 0).wait()

    def stop_all(self) -> None:
        self._clear_turn_key()
        self.set_forward(False)
        self.set_sprint(False)

    def _apply_mouse_steering(self, heading_error: float, reliability: float) -> None:
        mouse_cfg = self.config["mouse_control"]
        deadzone = float(mouse_cfg["deadzone_deg"])
        if abs(heading_error) <= deadzone:
            return

        gain = float(mouse_cfg["x_gain"])
        min_step = int(mouse_cfg["min_step"])
        max_step = int(mouse_cfg["max_step"])
        dx = int(round(heading_error * gain * max(reliability, 0.35)))
        dx = max(-max_step, min(max_step, dx))
        if abs(dx) < min_step:
            dx = min_step if dx >= 0 else -min_step

        self.controller.post_relative_move(dx, 0).wait()

    def _apply_key_steering(self, heading_error: float) -> None:
        deadzone = float(self.config["movement"]["turn_deadzone_deg"])
        if abs(heading_error) <= deadzone:
            self._clear_turn_key()
        elif heading_error > 0:
            self._set_turn_key(self.right_key)
        else:
            self._set_turn_key(self.left_key)

    def _set_turn_key(self, key: int) -> None:
        if self.current_turn_key == key:
            return

        if self.current_turn_key is not None:
            self.controller.post_key_up(self.current_turn_key)

        self.controller.post_key_down(key)
        self.current_turn_key = key

    def _clear_turn_key(self) -> None:
        if self.current_turn_key is not None:
            self.controller.post_key_up(self.current_turn_key)
            self.current_turn_key = None


@AgentServer.custom_action("auto_navigate_by_line")
class AutoNavigateByLine(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            model_path = abs_path / "assets/resource/base/model/navi/pointer_model.onnx"
        else:
            model_path = abs_path / "resource/base/model/navi/pointer_model.onnx"

        self.model_path = model_path
        self._session_cache: Dict[str, Tuple[onnxruntime.InferenceSession, str]] = {}
        self._provider_name_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
            "dml": "DmlExecutionProvider",
        }
        self._debug_state: Dict[str, Any] = {}

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        config = self._build_config(argv.custom_action_param)
        controller = context.tasker.controller
        mover = MovementController(controller, config)

        backend = self._resolve_backend(config["backend"])
        session, provider_name = self._get_session(backend)
        input_name = session.get_inputs()[0].name

        mouse_lock_enabled = False
        if config["steering_mode"] == "mouse_relative" and config["mouse_control"]["enable_mouse_lock_follow"]:
            try:
                mouse_lock_enabled = bool(controller.set_mouse_lock_follow(True))
                print(f"鼠标锁定跟随: {'开启' if mouse_lock_enabled else '开启失败，继续尝试相对移动'}")
            except Exception as exc:
                print(f"开启鼠标锁定跟随失败: {exc}")

        print("自动寻路框架启动")
        print(f"角度识别后端: {backend.upper()} ({provider_name})")
        print("请先在 custom_action_param 中补全导航线 ROI、终点判定等占位配置")

        frame_interval = float(config["frame_interval_sec"])
        last_line_seen_at = time.perf_counter()
        last_pointer_angle: Optional[float] = None
        smoothed_heading_error = 0.0
        scan_direction = 1
        frame_index = 0

        try:
            while not context.tasker.stopping:
                loop_started = time.perf_counter()
                frame = self._get_bgr_frame(controller)
                frame_index += 1

                pointer = self._predict_pointer_angle(frame, config, session, input_name)
                if pointer is not None:
                    last_pointer_angle = pointer.angle

                line = self._detect_navigation_line(frame, config)
                if self._is_goal_reached(frame, config):
                    print("检测到终点到达信号，自动寻路结束")
                    break

                if line is not None:
                    last_line_seen_at = time.perf_counter()
                    smoothed_heading_error = (
                        smoothed_heading_error * (1.0 - float(config["line_follow"]["smoothing"]))
                        + line.heading_error * float(config["line_follow"]["smoothing"])
                    )
                    mover.set_forward(True)
                    mover.set_sprint(bool(config["movement"]["hold_sprint"]))
                    mover.apply_steering(smoothed_heading_error, line.reliability)

                    target_world_angle = None
                    if last_pointer_angle is not None:
                        target_world_angle = (last_pointer_angle + smoothed_heading_error) % 360

                    if frame_index % int(config["log_every_n_frames"]) == 0:
                        player_angle_text = "--" if last_pointer_angle is None else f"{last_pointer_angle:06.2f}"
                        target_angle_text = "--" if target_world_angle is None else f"{target_world_angle:06.2f}"
                        print(
                            f"[导航] color={line.color_name:<6} err={smoothed_heading_error:+06.2f}° "
                            f"player={player_angle_text}° target={target_angle_text}° "
                            f"area={line.area:.0f} pixels={line.edge_points} rel={line.reliability:.2f}"
                        )
                else:
                    lost_for = time.perf_counter() - last_line_seen_at
                    mover.set_sprint(False)
                    if lost_for <= float(config["movement"]["lost_line_keep_forward_sec"]):
                        mover.set_forward(True)
                    else:
                        mover.set_forward(False)

                    if bool(config["movement"]["scan_when_lost"]) and lost_for >= float(config["movement"]["lost_line_scan_after_sec"]):
                        mover.search_turn(int(config["movement"]["lost_line_scan_step"]) * scan_direction)
                        scan_direction *= -1
                    else:
                        mover.search_turn(0)

                    if frame_index % int(config["log_every_n_frames"]) == 0:
                        print(f"[导航] 暂时丢失导航线，lost_for={lost_for:.2f}s")

                if bool(config["debug"]):
                    self._show_debug_window(frame, config, pointer, line, smoothed_heading_error)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("收到调试窗口退出指令，结束自动寻路")
                        break

                self._handle_custom_stuck(frame, config, line, pointer)

                elapsed = time.perf_counter() - loop_started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            mover.stop_all()
            if mouse_lock_enabled:
                try:
                    controller.set_mouse_lock_follow(False)
                except Exception:
                    pass
            if bool(config["debug"]):
                cv2.destroyAllWindows()

        return CustomAction.RunResult(success=True)

    def _build_config(self, custom_action_param: Any) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {
            "backend": os.environ.get("MAA_ONNX_BACKEND", "auto"),
            "debug": True,
            "frame_interval_sec": 0.05,
            "log_every_n_frames": 10,
            "steering_mode": "mouse_relative",
            "pointer_roi": [73, 60, 64, 64],
            "pointer_threshold": 0.5,
            "nav_line_roi": None,
            "line_anchor": None,
            "line_follow": {
                "min_contour_area": 80,
                "smoothing": 0.35,
            },
            "line_ring": {
                "enabled": True,
                "center": [105, 95],
                "inner_radius": 50,
                "outer_radius": 60,
            },
            "movement": {
                "forward_key": 87,
                "left_key": 65,
                "right_key": 68,
                "sprint_key": 16,
                "hold_sprint": False,
                "turn_deadzone_deg": 6.0,
                "lost_line_keep_forward_sec": 0.5,
                "lost_line_scan_after_sec": 0.7,
                "lost_line_scan_step": 36,
                "scan_when_lost": True,
            },
            "mouse_control": {
                "enable_mouse_lock_follow": True,
                "x_gain": 2.4,
                "min_step": 2,
                "max_step": 72,
                "deadzone_deg": 3.0,
            },
            "line_colors": {
                "blue": [
                    [[0, 193, 80], [179, 255, 255]],
                ],
            },
            "goal_check": {
                "enabled": False,
                "roi": None,
                "hsv_ranges": [],
                "min_pixels": 200,
            },
            "hooks": {
                "blocked_check_roi": None,
                "mini_map_center": None,
            },
        }

        user_config = self._parse_param(custom_action_param)
        self._deep_update(default_config, user_config)
        return default_config

    def _parse_param(self, custom_action_param: Any) -> Dict[str, Any]:
        if custom_action_param is None or custom_action_param == "":
            return {}

        if isinstance(custom_action_param, dict):
            return custom_action_param

        try:
            data = json.loads(custom_action_param)
        except Exception as exc:
            print(f"解析 custom_action_param 失败，将使用默认配置: {exc}")
            return {}

        return data if isinstance(data, dict) else {}

    def _deep_update(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _get_bgr_frame(self, controller) -> np.ndarray:
        frame = get_image(controller)
        if frame is None:
            raise RuntimeError("截图失败，未获取到图像")

        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame.copy()

    def _predict_pointer_angle(
        self,
        frame_bgr: np.ndarray,
        config: Dict[str, Any],
        session: onnxruntime.InferenceSession,
        input_name: str,
    ) -> Optional[PointerDetection]:
        roi = self._resolve_roi(config.get("pointer_roi"), frame_bgr.shape)
        if roi is None:
            return None

        x, y, w, h = roi
        crop = frame_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_input = crop_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)

        result = session.run(None, {input_name: img_input})
        output = result[0][0]
        confidence = output[:, 4]
        best_idx = int(np.argmax(confidence))
        best_pred = output[best_idx]
        max_conf = float(confidence[best_idx])

        if max_conf < float(config["pointer_threshold"]):
            return None

        kpts = best_pred[6:].reshape(3, 3)
        tip = kpts[0][:2]
        left = kpts[1][:2]
        right = kpts[2][:2]
        tail_center = (left + right) / 2.0

        dx = float(tip[0] - tail_center[0])
        dy = float(tip[1] - tail_center[1])
        angle = math.degrees(math.atan2(dx, -dy)) % 360

        self._debug_state["pointer_crop"] = crop.copy()
        self._debug_state["pointer_keypoints"] = {
            "tip": (int(tip[0]), int(tip[1])),
            "left": (int(left[0]), int(left[1])),
            "right": (int(right[0]), int(right[1])),
            "tail": (int(tail_center[0]), int(tail_center[1])),
        }
        return PointerDetection(angle=angle, confidence=max_conf)

    def _detect_navigation_line(self, frame_bgr: np.ndarray, config: Dict[str, Any]) -> Optional[LineDetection]:
        roi = self._resolve_roi(config.get("nav_line_roi"), frame_bgr.shape, allow_default=True)
        if roi is None:
            roi = self._default_nav_line_roi(frame_bgr.shape)

        x, y, w, h = roi
        roi_img = frame_bgr[y:y + h, x:x + w]
        if roi_img.size == 0:
            return None

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        anchor = self._resolve_anchor(config.get("line_anchor"), roi)
        ring_mask = self._build_line_ring_mask(roi_img.shape[:2], config.get("line_ring"))
        best_detection: Optional[LineDetection] = None
        best_mask: Optional[np.ndarray] = None
        best_blob_mask: Optional[np.ndarray] = None

        line_cfg = config["line_follow"]
        debug_candidates: List[Dict[str, Any]] = []
        for color_name, ranges in config["line_colors"].items():
            raw_mask = self._build_color_mask(hsv, ranges)
            mask = raw_mask.copy()
            if ring_mask is not None:
                mask = self._intersect_binary_masks(mask, ring_mask)
            detection, debug_info = self._pick_best_line(mask, color_name, anchor, line_cfg)
            debug_info["color_name"] = color_name
            debug_info["raw_mask"] = raw_mask
            debug_info["ring_mask"] = ring_mask
            debug_info["intersection_mask"] = mask
            debug_info["overlay_panel"] = self._build_mask_ring_overlay(raw_mask, ring_mask, mask)
            debug_candidates.append(debug_info)
            if detection is None:
                continue

            if best_detection is None or detection.score > best_detection.score:
                best_detection = detection
                best_mask = mask
                best_blob_mask = debug_info.get("selected_blob_mask")
                self._debug_state["selected_line_debug"] = debug_info

        if best_detection is None and debug_candidates:
            best_debug_info = max(debug_candidates, key=lambda item: item.get("mask_pixels", 0))
            self._debug_state["selected_line_debug"] = best_debug_info

        self._debug_state["nav_roi"] = roi_img.copy()
        self._debug_state["nav_hsv"] = hsv.copy()
        self._debug_state["nav_roi_rect"] = roi
        self._debug_state["line_anchor"] = anchor
        self._debug_state["line_mask"] = best_mask
        self._debug_state["line_blob_mask"] = best_blob_mask
        self._debug_state["line_ring_mask"] = ring_mask
        self._debug_state["line_ring_config"] = config.get("line_ring")
        self._debug_state["line_debug_candidates"] = debug_candidates
        return best_detection

    def _build_color_mask(self, hsv: np.ndarray, ranges: List[List[List[int]]]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_arr, upper_arr))
        return mask

    def _build_line_ring_mask(
        self,
        roi_shape: Tuple[int, int],
        ring_cfg: Optional[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        if not isinstance(ring_cfg, dict) or not bool(ring_cfg.get("enabled")):
            return None

        height, width = roi_shape
        center = self._resolve_circle_center(ring_cfg.get("center"), width, height)
        if center is None:
            return None

        inner_radius = int(ring_cfg.get("inner_radius", -1))
        outer_radius = int(ring_cfg.get("outer_radius", -1))
        if outer_radius <= 0 or outer_radius <= inner_radius:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, outer_radius, 255, thickness=-1)
        if inner_radius > 0:
            cv2.circle(mask, center, inner_radius, 0, thickness=-1)
        return mask

    def _intersect_binary_masks(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.where((left > 0) & (right > 0), 255, 0).astype(np.uint8)

    def _build_mask_ring_overlay(
        self,
        raw_mask: np.ndarray,
        ring_mask: Optional[np.ndarray],
        intersection_mask: np.ndarray,
    ) -> np.ndarray:
        height, width = raw_mask.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[:, :, 2] = raw_mask
        if ring_mask is not None:
            overlay[:, :, 1] = ring_mask
        overlay[:, :, 0] = intersection_mask
        return overlay

    def _pick_best_line(
        self,
        mask: np.ndarray,
        color_name: str,
        anchor: Tuple[int, int],
        line_cfg: Dict[str, Any],
    ) -> Tuple[Optional[LineDetection], Dict[str, Any]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        debug_info: Dict[str, Any] = {
            "color_name": color_name,
            "mask_pixels": int(cv2.countNonZero(mask)),
            "contour_count": len(contours),
            "reject_small_area": 0,
            "best_score": None,
            "message": "",
            "selected_blob_mask": None,
        }
        if not contours:
            debug_info["message"] = "圆环与 HSV 相交后没有轮廓"
            return None, debug_info

        anchor_arr = np.array(anchor, dtype=np.float32)
        best: Optional[LineDetection] = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(line_cfg["min_contour_area"]):
                debug_info["reject_small_area"] += 1
                continue

            pts = contour[:, 0, :].astype(np.float32)
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                target = np.mean(pts, axis=0)
            else:
                target = np.array(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                    dtype=np.float32,
                )

            heading_error = math.degrees(math.atan2(float(target[0] - anchor_arr[0]), float(anchor_arr[1] - target[1])))
            score = area
            reliability = min(1.0, max(0.1, area / 800.0))

            selected_blob_mask = np.zeros_like(mask)
            cv2.drawContours(selected_blob_mask, [contour], -1, 255, thickness=-1)
            detection = LineDetection(
                color_name=color_name,
                target_point=(int(target[0]), int(target[1])),
                heading_error=heading_error,
                angle_min=heading_error,
                angle_max=heading_error,
                area=area,
                score=score,
                reliability=reliability,
                edge_points=int(cv2.countNonZero(selected_blob_mask)),
            )

            if best is None or detection.score > best.score:
                best = detection
                debug_info["best_score"] = float(detection.score)
                debug_info["message"] = "检测成功"
                debug_info["selected_blob_mask"] = selected_blob_mask

        if best is None:
            debug_info["message"] = self._summarize_line_failure(debug_info)
        return best, debug_info

    def _summarize_line_failure(self, debug_info: Dict[str, Any]) -> str:
        if debug_info.get("mask_pixels", 0) == 0:
            return "HSV 与圆环相交后没有像素"
        if debug_info.get("contour_count", 0) == 0:
            return "没有轮廓"

        reject_pairs = [
            ("面积太小", debug_info.get("reject_small_area", 0)),
        ]
        reject_pairs.sort(key=lambda item: item[1], reverse=True)
        label, count = reject_pairs[0]
        if count <= 0:
            return "存在轮廓，但未通过筛选"
        return f"主要被过滤原因: {label} ({count})"

    def _is_goal_reached(self, frame_bgr: np.ndarray, config: Dict[str, Any]) -> bool:
        goal_cfg = config["goal_check"]
        if not bool(goal_cfg["enabled"]):
            return False

        roi = self._resolve_roi(goal_cfg.get("roi"), frame_bgr.shape)
        if roi is None:
            print("goal_check.enabled 已开启，但未填写 goal_check.roi")
            return False

        x, y, w, h = roi
        crop = frame_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self._build_color_mask(hsv, goal_cfg.get("hsv_ranges", []))
        pixels = int(cv2.countNonZero(mask))
        return pixels >= int(goal_cfg.get("min_pixels", 200))

    def _handle_custom_stuck(
        self,
        frame_bgr: np.ndarray,
        config: Dict[str, Any],
        line: Optional[LineDetection],
        pointer: Optional[PointerDetection],
    ) -> None:
        _ = frame_bgr
        _ = line
        _ = pointer
        blocked_roi = config.get("hooks", {}).get("blocked_check_roi")
        if blocked_roi is None:
            return

        # TODO: 用户在这里补自己的卡住判定。
        # 例如：
        # 1. 检测“可交互/F”提示是否长时间不消失
        # 2. 检测角色速度很低但角度持续变化
        # 3. 检测固定 UI 文本/图标
        return

    def _show_debug_window(
        self,
        frame_bgr: np.ndarray,
        config: Dict[str, Any],
        pointer: Optional[PointerDetection],
        line: Optional[LineDetection],
        smoothed_heading_error: float,
    ) -> None:
        nav_roi = self._debug_state.get("nav_roi")
        nav_rect = self._debug_state.get("nav_roi_rect")
        line_mask = self._debug_state.get("line_mask")
        line_blob_mask = self._debug_state.get("line_blob_mask")
        anchor = self._debug_state.get("line_anchor")
        line_ring_mask = self._debug_state.get("line_ring_mask")
        line_ring_config = self._debug_state.get("line_ring_config")
        selected_line_debug = self._debug_state.get("selected_line_debug")

        if nav_roi is None or nav_rect is None:
            preview = frame_bgr.copy()
        else:
            preview = nav_roi.copy()
            if line_ring_mask is not None:
                ring_overlay = np.zeros_like(preview)
                ring_overlay[:, :, 1] = line_ring_mask
                preview = cv2.addWeighted(preview, 0.85, ring_overlay, 0.20, 0)
            if line_mask is not None:
                mask_bgr = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
                preview = cv2.addWeighted(preview, 0.65, mask_bgr, 0.35, 0)
            if line_blob_mask is not None:
                blob_overlay = np.zeros_like(preview)
                blob_overlay[:, :, 2] = line_blob_mask
                preview = cv2.addWeighted(preview, 0.80, blob_overlay, 0.55, 0)
            if anchor is not None:
                cv2.circle(preview, anchor, 4, (0, 255, 255), -1)
            if isinstance(line_ring_config, dict) and bool(line_ring_config.get("enabled")):
                center = self._resolve_circle_center(
                    line_ring_config.get("center"),
                    preview.shape[1],
                    preview.shape[0],
                )
                inner_radius = int(line_ring_config.get("inner_radius", -1))
                outer_radius = int(line_ring_config.get("outer_radius", -1))
                if center is not None and outer_radius > 0:
                    cv2.circle(preview, center, outer_radius, (0, 200, 255), 1)
                    if inner_radius > 0:
                        cv2.circle(preview, center, inner_radius, (0, 120, 255), 1)
            if line is not None:
                cv2.circle(preview, line.target_point, 5, (0, 0, 255), -1)
                cv2.line(preview, anchor, line.target_point, (255, 255, 255), 2)
                cv2.putText(
                    preview,
                    f"{line.color_name} err={smoothed_heading_error:+.1f} area={line.area:.0f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        pointer_crop = self._debug_state.get("pointer_crop")
        keypoints = self._debug_state.get("pointer_keypoints")
        if pointer_crop is not None:
            pointer_preview = pointer_crop.copy()
            if keypoints is not None:
                cv2.line(pointer_preview, keypoints["tail"], keypoints["tip"], (255, 0, 255), 2)
                cv2.circle(pointer_preview, keypoints["tip"], 2, (0, 0, 255), -1)
                cv2.circle(pointer_preview, keypoints["left"], 2, (255, 255, 0), -1)
                cv2.circle(pointer_preview, keypoints["right"], 2, (255, 255, 0), -1)
            pointer_preview = cv2.resize(pointer_preview, (preview.shape[1], max(96, preview.shape[0] // 3)))
            if pointer is not None:
                cv2.putText(
                    pointer_preview,
                    f"pointer={pointer.angle:06.2f} conf={pointer.confidence:.2f}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            preview = np.vstack([preview, pointer_preview])

        cv2.imshow(str(config.get("debug_window_name", "Auto Navigate By Line")), preview)
        self._show_debug_analysis_window(config, selected_line_debug)

    def _show_debug_analysis_window(
        self,
        config: Dict[str, Any],
        selected_line_debug: Optional[Dict[str, Any]],
    ) -> None:
        nav_roi = self._debug_state.get("nav_roi")
        nav_hsv = self._debug_state.get("nav_hsv")
        if nav_roi is None or nav_hsv is None:
            return

        selected_line_debug = selected_line_debug or {}
        raw_mask = selected_line_debug.get("raw_mask")
        ring_mask = selected_line_debug.get("ring_mask")
        intersection_mask = selected_line_debug.get("intersection_mask")
        selected_blob_mask = selected_line_debug.get("selected_blob_mask")
        overlay_panel = selected_line_debug.get("overlay_panel")

        height, width = nav_roi.shape[:2]
        raw_panel = cv2.resize(nav_roi, (width, height))

        hsv_panel = cv2.cvtColor(nav_hsv, cv2.COLOR_HSV2BGR)
        hsv_panel = cv2.resize(hsv_panel, (width, height))

        raw_mask_panel = self._mask_to_bgr(raw_mask, width, height, label="HSV Mask")
        ring_mask_panel = self._mask_to_bgr(ring_mask, width, height, label="Ring Mask")
        intersection_mask_panel = self._mask_to_bgr(intersection_mask, width, height, label="HSV ∩ Ring")
        selected_blob_panel = self._mask_to_bgr(selected_blob_mask, width, height, label="Selected Blob")
        overlay_debug_panel = self._mask_to_bgr(overlay_panel, width, height, label="Overlay")

        top_row = np.hstack([raw_panel, hsv_panel, overlay_debug_panel])
        bottom_row = np.hstack([raw_mask_panel, ring_mask_panel, intersection_mask_panel, selected_blob_panel])

        if top_row.shape[1] < bottom_row.shape[1]:
            pad = bottom_row.shape[1] - top_row.shape[1]
            top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif bottom_row.shape[1] < top_row.shape[1]:
            pad = top_row.shape[1] - bottom_row.shape[1]
            bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        analysis = np.vstack([top_row, bottom_row])

        lines = [
            f"reason: {selected_line_debug.get('message', '无')}",
            f"mask={selected_line_debug.get('mask_pixels', 0)} ring={0 if ring_mask is None else int(cv2.countNonZero(ring_mask))} intersect={0 if intersection_mask is None else int(cv2.countNonZero(intersection_mask))}",
            f"contours={selected_line_debug.get('contour_count', 0)} best_score={selected_line_debug.get('best_score', 0)}",
            f"reject small={selected_line_debug.get('reject_small_area', 0)}",
        ]

        analysis = cv2.copyMakeBorder(analysis, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        base_y = analysis.shape[0] - 52
        for idx, text in enumerate(lines):
            cv2.putText(
                analysis,
                text,
                (10, base_y + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        window_name = str(config.get("debug_analysis_window_name", "Auto Navigate Analysis"))
        cv2.imshow(window_name, analysis)

    def _mask_to_bgr(self, mask: Optional[np.ndarray], width: int, height: int, label: str) -> np.ndarray:
        if mask is None:
            panel = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            if mask.ndim == 2:
                panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                panel = mask.copy()
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST)

        cv2.putText(
            panel,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return panel

    def _default_nav_line_roi(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        roi_w = int(frame_w * 0.45)
        roi_h = int(frame_h * 0.42)
        roi_x = (frame_w - roi_w) // 2
        roi_y = int(frame_h * 0.28)
        return roi_x, roi_y, roi_w, roi_h

    def _resolve_anchor(self, anchor: Optional[List[int]], roi: Tuple[int, int, int, int]) -> Tuple[int, int]:
        _, _, w, h = roi
        if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
            ax, ay = int(anchor[0]), int(anchor[1])
            if ax >= 0 and ay >= 0:
                return min(ax, w - 1), min(ay, h - 1)
        return w // 2, h - 1

    def _resolve_circle_center(
        self,
        center: Optional[List[int]],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            return None

        cx, cy = int(center[0]), int(center[1])
        if cx < 0 or cy < 0:
            return None
        return min(cx, width - 1), min(cy, height - 1)

    def _resolve_roi(
        self,
        roi: Optional[List[int]],
        frame_shape: Tuple[int, ...],
        allow_default: bool = False,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            return None if allow_default else None

        x, y, w, h = [int(v) for v in roi]
        if min(w, h) <= 0 or x < 0 or y < 0:
            return None if allow_default else None

        frame_h, frame_w = frame_shape[:2]
        x = min(x, frame_w - 1)
        y = min(y, frame_h - 1)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

    def _resolve_backend(self, backend: str) -> str:
        backend = str(backend).strip().lower()
        if backend == "auto":
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return "cuda"
            if "DmlExecutionProvider" in available:
                return "directml"
            return "cpu"

        if backend not in self._provider_name_map:
            print(f"未知推理后端 {backend}，将回退到 CPU")
            return "cpu"
        return backend

    def _get_session(self, backend: str) -> Tuple[onnxruntime.InferenceSession, str]:
        if backend in self._session_cache:
            return self._session_cache[backend]

        provider_name = self._provider_name_map[backend]
        available = onnxruntime.get_available_providers()
        if provider_name not in available:
            print(f"请求的后端 {backend.upper()} 不可用，已回退到 CPU，当前 Providers: {available}")
            backend = "cpu"
            provider_name = self._provider_name_map[backend]

        session_options = onnxruntime.SessionOptions()
        providers = [provider_name]
        provider_options = None

        if provider_name in {"CUDAExecutionProvider", "DmlExecutionProvider"}:
            provider_options = [{"device_id": 0}]

        if provider_options is None:
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers,
            )
        else:
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )

        self._session_cache[backend] = (session, provider_name)
        return self._session_cache[backend]
