"""新版钓鱼控条的纯控制逻辑。"""

from __future__ import annotations

from typing import Optional

KEY_A = 65
KEY_D = 68


def estimate_error_velocity(
    last_error: Optional[float],
    error: float,
    delta_seconds: float,
    last_velocity: float = 0.0,
    alpha: float = 0.35,
    max_velocity: float = 4000.0,
) -> float:
    """用误差变化估计相对速度，并用 EWMA 限制识别噪声。"""
    if last_error is None or delta_seconds <= 0:
        return last_velocity

    raw_velocity = (error - last_error) / delta_seconds
    raw_velocity = max(-max_velocity, min(max_velocity, raw_velocity))
    alpha = max(0.0, min(1.0, alpha))
    return last_velocity + alpha * (raw_velocity - last_velocity)


def choose_control_key(
    current_key: Optional[int],
    predicted_error: float,
    enter_deadzone: float = 15.0,
    release_deadzone: float = 7.0,
) -> Optional[int]:
    """根据预测误差决定保持、切换或释放 A/D。

    正误差表示光标在绿条右侧，需要按 A；负误差需要按 D。
    ``enter_deadzone`` 与 ``release_deadzone`` 形成施密特滞回，避免边界抖动。
    """
    enter_deadzone = max(0.0, float(enter_deadzone))
    release_deadzone = max(0.0, min(float(release_deadzone), enter_deadzone))

    if current_key is None:
        if predicted_error > enter_deadzone:
            return KEY_A
        if predicted_error < -enter_deadzone:
            return KEY_D
        return None

    if current_key == KEY_A:
        if predicted_error < -enter_deadzone:
            return KEY_D
        if predicted_error < release_deadzone:
            return None
        return KEY_A

    if current_key == KEY_D:
        if predicted_error > enter_deadzone:
            return KEY_A
        if predicted_error > -release_deadzone:
            return None
        return KEY_D

    return choose_control_key(None, predicted_error, enter_deadzone, release_deadzone)


def choose_tracking_key(
    current_key: Optional[int],
    cursor_center: float,
    cursor_velocity: float,
    green_left: float,
    green_right: float,
    green_velocity: float = 0.0,
    lookahead_seconds: float = 0.16,
    safe_margin: float = 5.0,
    center_band_ratio: float = 0.4,
    switch_margin: float = 3.0,
) -> Optional[int]:
    """光标预测会离开绿条中心走廊时才进行控制。

    A 将光标向左移动，D 将光标向右移动。绿条中心与光标使用同一个
    预测时间窗，从而在绿条移动时提前跟随。
    """
    lookahead_seconds = max(0.0, float(lookahead_seconds))
    safe_margin = max(0.0, float(safe_margin))
    center_band_ratio = max(0.0, min(1.0, float(center_band_ratio)))
    switch_margin = max(0.0, float(switch_margin))

    predicted_cursor = cursor_center + cursor_velocity * lookahead_seconds
    safe_left, safe_right = predict_tracking_interval(
        green_left,
        green_right,
        green_velocity,
        lookahead_seconds,
        safe_margin,
        center_band_ratio,
    )

    # 已在安全区内时松键，让下一帧观测决定是否需要跟随。
    if current_key == KEY_A:
        if predicted_cursor < safe_left - switch_margin:
            return KEY_D
        if predicted_cursor <= safe_right:
            return None
        return KEY_A

    if current_key == KEY_D:
        if predicted_cursor > safe_right + switch_margin:
            return KEY_A
        if predicted_cursor >= safe_left:
            return None
        return KEY_D

    if predicted_cursor > safe_right:
        return KEY_A
    if predicted_cursor < safe_left:
        return KEY_D
    return None


def predict_tracking_interval(
    green_left: float,
    green_right: float,
    green_velocity: float,
    lookahead_seconds: float,
    safe_margin: float,
    center_band_ratio: float,
) -> tuple[float, float]:
    """返回预测时刻绿条中心走廊的左右边界。"""
    lookahead_seconds = max(0.0, float(lookahead_seconds))
    safe_margin = max(0.0, float(safe_margin))
    center_band_ratio = max(0.0, min(1.0, float(center_band_ratio)))
    green_width = max(0.0, float(green_right) - float(green_left))
    predicted_center = (float(green_left) + float(green_right)) / 2.0 + float(
        green_velocity
    ) * lookahead_seconds
    usable_width = max(0.0, green_width - safe_margin * 2.0)
    half_band = usable_width * center_band_ratio / 2.0
    return predicted_center - half_band, predicted_center + half_band


def should_finish_control(
    has_seen_control: bool,
    last_cursor_seen: Optional[float],
    now: float,
    grace_ms: float,
) -> bool:
    """控条开始后光标持续消失，视为进入结果阶段。"""
    if not has_seen_control or last_cursor_seen is None:
        return False
    return (float(now) - float(last_cursor_seen)) * 1000 >= max(0.0, float(grace_ms))
