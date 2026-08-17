import unittest
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parents[1] / "agent" / "custom" / "action" / "AutoFish")
)

from fish_control import (
    KEY_A,
    KEY_D,
    choose_control_key,
    choose_tracking_key,
    estimate_error_velocity,
    should_finish_control,
)
from fish_params import load_custom_action_params


class FishControlTests(unittest.TestCase):
    def test_custom_action_params_are_parsed_consistently(self):
        params = {"safe_margin": 6}
        self.assertIs(load_custom_action_params(params), params)
        self.assertEqual(
            load_custom_action_params('{"safe_margin": 8}'),
            {"safe_margin": 8},
        )
        self.assertEqual(load_custom_action_params("invalid"), {})
        self.assertEqual(load_custom_action_params("[]"), {})

    def test_hysteresis_holds_and_releases_a(self):
        self.assertEqual(choose_control_key(None, 20), KEY_A)
        self.assertEqual(choose_control_key(KEY_A, 10), KEY_A)
        self.assertIsNone(choose_control_key(KEY_A, 6))

    def test_hysteresis_holds_and_releases_d(self):
        self.assertEqual(choose_control_key(None, -20), KEY_D)
        self.assertEqual(choose_control_key(KEY_D, -10), KEY_D)
        self.assertIsNone(choose_control_key(KEY_D, -6))

    def test_reverses_only_after_crossing_enter_threshold(self):
        self.assertIsNone(choose_control_key(KEY_A, -10))
        self.assertEqual(choose_control_key(KEY_A, -16), KEY_D)
        self.assertEqual(choose_control_key(KEY_D, 16), KEY_A)

    def test_velocity_is_smoothed_and_clamped(self):
        velocity = estimate_error_velocity(None, 10, 0.01)
        self.assertEqual(velocity, 0.0)
        velocity = estimate_error_velocity(0, 100, 0.1, alpha=1.0)
        self.assertEqual(velocity, 1000.0)
        velocity = estimate_error_velocity(0, 1000, 0.001, alpha=1.0)
        self.assertEqual(velocity, 4000.0)

    def test_delayed_tracking_recovers_from_cursor_disturbance(self):
        """延迟与惯性存在时，光标受扰动后应回到固定绿条内。"""
        dt = 0.01
        observation_interval = 0.08
        input_delay = 0.04
        green_half_width = 48.0
        cursor_x = 680.0
        actual_cursor_velocity = 0.0
        current_key = None
        applied_key = None
        pending_inputs = []
        last_observation = -observation_interval
        last_cursor = None
        estimated_cursor_velocity = 0.0
        hold_until = 0.0
        outside_samples = 0
        measured_samples = 0
        max_overshoot = 0.0

        for step in range(int(8.0 / dt)):
            now = step * dt
            green_center = 640.0

            if step == 150:
                actual_cursor_velocity += 180.0
            elif step == 420:
                actual_cursor_velocity -= 220.0

            while pending_inputs and pending_inputs[0][0] <= now:
                _, applied_key = pending_inputs.pop(0)

            direction = 0.0
            if applied_key == KEY_A:
                direction = -1.0
            elif applied_key == KEY_D:
                direction = 1.0

            actual_cursor_velocity += (
                direction * 900.0 - actual_cursor_velocity * 3.0
            ) * dt
            actual_cursor_velocity = max(-300.0, min(300.0, actual_cursor_velocity))
            cursor_x += actual_cursor_velocity * dt

            if now - last_observation >= observation_interval - 1e-9:
                if now >= hold_until:
                    current_key = None
                estimated_cursor_velocity = estimate_error_velocity(
                    last_cursor,
                    cursor_x,
                    observation_interval,
                    estimated_cursor_velocity,
                    alpha=0.35,
                )
                current_key = choose_tracking_key(
                    current_key,
                    cursor_x,
                    estimated_cursor_velocity,
                    green_center - green_half_width,
                    green_center + green_half_width,
                    lookahead_seconds=0.16,
                    safe_margin=5,
                    switch_margin=3,
                )
                if current_key is not None:
                    hold_until = now + 0.035
                pending_inputs.append((now + input_delay, current_key))
                last_cursor = cursor_x
                last_observation = now

            if now >= 1.0:
                measured_samples += 1
                overshoot = max(0.0, abs(cursor_x - green_center) - green_half_width)
                if overshoot > 0:
                    outside_samples += 1
                max_overshoot = max(max_overshoot, overshoot)

        self.assertLess(outside_samples / measured_samples, 0.12)
        self.assertLess(max_overshoot, 14.0)

    def test_uses_center_corridor_without_chasing_a_single_point(self):
        self.assertIsNone(
            choose_tracking_key(
                None,
                cursor_center=60,
                cursor_velocity=0,
                green_left=0,
                green_right=100,
                lookahead_seconds=0.16,
                safe_margin=5,
                center_band_ratio=0.4,
            )
        )
        self.assertEqual(
            choose_tracking_key(
                None,
                cursor_center=76,
                cursor_velocity=0,
                green_left=0,
                green_right=100,
                lookahead_seconds=0.16,
                safe_margin=5,
                center_band_ratio=0.4,
            ),
            KEY_A,
        )

    def test_follows_moving_green_bar_before_reaching_edge(self):
        self.assertEqual(
            choose_tracking_key(
                None,
                cursor_center=670,
                cursor_velocity=80,
                green_left=655,
                green_right=745,
                green_velocity=120,
                lookahead_seconds=0.14,
                center_band_ratio=0.4,
            ),
            KEY_D,
        )

    def test_cursor_loss_transitions_to_result_instead_of_error_restart(self):
        self.assertFalse(should_finish_control(True, 10.0, 10.299, 300))
        self.assertTrue(should_finish_control(True, 10.0, 10.300, 300))
        self.assertFalse(should_finish_control(False, 10.0, 11.0, 300))


if __name__ == "__main__":
    unittest.main()
