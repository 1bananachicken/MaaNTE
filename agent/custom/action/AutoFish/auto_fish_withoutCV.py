from __future__ import annotations

import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils.logger import logger

from .fish_control import (
    KEY_A,
    KEY_D,
    choose_tracking_key,
    estimate_error_velocity,
    predict_tracking_interval,
    should_finish_control,
)
from .fish_params import load_fish_control_params
from .fish_vision import detect_control_boxes


@AgentServer.custom_action("auto_fish_without_cv")
class AutoFishWithoutCV(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = load_fish_control_params(argv.custom_action_param)
        safe_margin = params["safe_margin"]
        center_band_ratio = params["center_band_ratio"]
        prediction_ms = params["prediction_ms"]
        velocity_alpha = params["velocity_alpha"]
        green_velocity_alpha = params["green_velocity_alpha"]
        green_center_alpha = params["green_center_alpha"]
        pulse_min_ms = params["pulse_min_ms"]
        pulse_max_ms = params["pulse_max_ms"]
        pulse_ms_per_px = params["pulse_ms_per_px"]
        width_change_threshold = params["width_change_threshold"]
        width_confirm_frames = params["width_confirm_frames"]
        control_end_grace_ms = params["control_end_grace_ms"]
        lost_timeout_ms = params["lost_timeout_ms"]
        lost_abort_ms = params["lost_abort_ms"]
        loop_interval_ms = params["loop_interval_ms"]

        controller = context.tasker.controller
        last_cursor_center = None
        last_sample_time = None
        cursor_velocity = 0.0
        green_center = None
        green_width = None
        green_velocity = 0.0
        last_detected_green_center = None
        last_green_sample_time = None
        width_candidate = None
        width_candidate_hits = 0
        lost_since = None
        last_cursor_seen = None
        has_seen_control = False
        last_status_log = 0.0
        frame_count = 0

        def tap_control_key(key, duration_ms):
            controller.post_key_down(key).wait()
            try:
                time.sleep(max(0.0, duration_ms) / 1000.0)
            finally:
                controller.post_key_up(key).wait()

        def release_control_keys():
            for key in (KEY_A, KEY_D):
                try:
                    controller.post_key_up(key).wait()
                except Exception as exc:
                    logger.warning("释放钓鱼按键失败: key=%s error=%s", key, exc)

        def update_green_bar(box, sample_time):
            nonlocal green_center, green_width, green_velocity
            nonlocal last_detected_green_center, last_green_sample_time
            nonlocal width_candidate, width_candidate_hits

            box_x, _, box_w, _ = box
            detected_center = float(box_x + box_w / 2)
            detected_width = float(box_w)
            if green_center is None or green_width is None:
                green_center = detected_center
                green_width = detected_width
                last_detected_green_center = detected_center
                last_green_sample_time = sample_time
                width_candidate = None
                width_candidate_hits = 0
                logger.debug(
                    "钓鱼绿条边界初始化: left=%.1f right=%.1f",
                    green_center - green_width / 2,
                    green_center + green_width / 2,
                )
                return

            if last_green_sample_time is not None:
                green_velocity = estimate_error_velocity(
                    last_detected_green_center,
                    detected_center,
                    sample_time - last_green_sample_time,
                    green_velocity,
                    green_velocity_alpha,
                )
            last_detected_green_center = detected_center
            last_green_sample_time = sample_time
            green_center += (detected_center - green_center) * green_center_alpha

            if abs(detected_width - green_width) < width_change_threshold:
                green_width += (detected_width - green_width) * 0.2
                width_candidate = None
                width_candidate_hits = 0
            elif (
                width_candidate is not None
                and abs(detected_width - width_candidate) < width_change_threshold / 2
            ):
                width_candidate_hits += 1
            else:
                width_candidate = detected_width
                width_candidate_hits = 1

            if width_candidate_hits >= width_confirm_frames:
                green_width = width_candidate
                logger.debug("钓鱼绿条宽度更新: width=%.1f", green_width)
                width_candidate = None
                width_candidate_hits = 0

        logger.debug("钓鱼开始：进入实时控条阶段")
        try:
            while not context.tasker.stopping:
                frame_started = time.monotonic()
                image = controller.post_screencap().wait().get()
                green_box, cursor_box = detect_control_boxes(image)
                now = time.monotonic()
                frame_count += 1

                if green_box is not None:
                    update_green_bar(green_box, now)
                if cursor_box is not None:
                    last_cursor_seen = now
                if green_box is not None and cursor_box is not None:
                    has_seen_control = True

                if should_finish_control(
                    has_seen_control,
                    last_cursor_seen,
                    now,
                    control_end_grace_ms,
                ):
                    logger.debug("钓鱼光标持续消失，进入结果处理")
                    return CustomAction.RunResult(success=True)

                valid = bool(
                    green_center is not None
                    and green_width is not None
                    and cursor_box is not None
                )

                if not valid:
                    if lost_since is None:
                        lost_since = now
                    lost_ms = (now - lost_since) * 1000
                    if lost_ms > lost_timeout_ms:
                        last_cursor_center = None
                        last_sample_time = None
                        cursor_velocity = 0.0
                    if lost_ms > lost_abort_ms:
                        logger.warning("钓鱼控条识别超时，交给 Pipeline 恢复")
                        return CustomAction.RunResult(success=False)
                    time.sleep(0.02)
                    continue

                lost_since = None
                cursor_x, _, cursor_w, _ = cursor_box
                cursor_center_x = cursor_x + cursor_w / 2

                if last_sample_time is not None:
                    sample_seconds = now - last_sample_time
                    cursor_velocity = estimate_error_velocity(
                        last_cursor_center,
                        cursor_center_x,
                        sample_seconds,
                        cursor_velocity,
                        velocity_alpha,
                    )
                else:
                    sample_seconds = 0.0
                last_cursor_center = cursor_center_x
                last_sample_time = now
                green_left = green_center - green_width / 2
                green_right = green_center + green_width / 2

                lookahead_seconds = max(
                    prediction_ms / 1000.0,
                    min(0.25, sample_seconds * 1.25),
                )
                next_key = choose_tracking_key(
                    None,
                    cursor_center_x,
                    cursor_velocity,
                    green_left,
                    green_right,
                    green_velocity=green_velocity,
                    lookahead_seconds=lookahead_seconds,
                    safe_margin=safe_margin,
                    center_band_ratio=center_band_ratio,
                )

                pulse_ms = 0.0
                if next_key is not None:
                    predicted_cursor = (
                        cursor_center_x + cursor_velocity * lookahead_seconds
                    )
                    safe_left, safe_right = predict_tracking_interval(
                        green_left,
                        green_right,
                        green_velocity,
                        lookahead_seconds,
                        safe_margin,
                        center_band_ratio,
                    )
                    if next_key == KEY_A:
                        outside_distance = max(0.0, predicted_cursor - safe_right)
                    else:
                        outside_distance = max(0.0, safe_left - predicted_cursor)
                    pulse_ms = min(
                        pulse_max_ms,
                        max(
                            pulse_min_ms,
                            pulse_min_ms + outside_distance * pulse_ms_per_px,
                        ),
                    )
                    tap_control_key(next_key, pulse_ms)

                if now - last_status_log >= 0.5:
                    last_status_log = now
                    frame_ms = (time.monotonic() - frame_started) * 1000
                    logger.debug(
                        "钓鱼控条状态: frame=%d frame_ms=%.1f bar=[%.1f, %.1f] "
                        "bar_velocity=%.1f cursor=%.1f cursor_velocity=%.1f "
                        "lookahead_ms=%.1f key=%s pulse_ms=%.1f",
                        frame_count,
                        frame_ms,
                        green_left,
                        green_right,
                        green_velocity,
                        cursor_center_x,
                        cursor_velocity,
                        lookahead_seconds * 1000,
                        next_key,
                        pulse_ms,
                    )

                if loop_interval_ms > 0:
                    time.sleep(loop_interval_ms / 1000.0)

            logger.debug("钓鱼控条因任务停止退出")
            return CustomAction.RunResult(success=False)
        except Exception:
            logger.exception("钓鱼控条异常")
            return CustomAction.RunResult(success=False)
        finally:
            release_control_keys()
