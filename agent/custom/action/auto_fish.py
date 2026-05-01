import time
from pathlib import Path
import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img

def match_template_in_region(img, region, template, min_similarity=0.8):
    if img is None or not isinstance(img, np.ndarray):
        return False, 0.0, 0, 0
    
    x1, y1, x2, y2 = region
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0, 0, 0
        
    roi = img[y1:y2, x1:x2]
    
    if len(roi.shape) == 3 and roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val >= min_similarity:
        return True, max_val, x1 + max_loc[0], y1 + max_loc[1]
    return False, max_val, 0, 0

@AgentServer.custom_action("auto_fish")
class AutoFish(CustomAction):
    abs_path = Path(__file__).parents[3]
    if Path.exists(abs_path / "assets"):
            image_dir = abs_path / "assets/resource/base/image/Fish"
    else:
        image_dir = abs_path / "resource/base/image/Fish"
    continue_img = image_dir / "continue.png"
    valid_region_left_img = image_dir / "valid_region_left.png"
    valid_region_right_img = image_dir / "valid_region_right.png"
    slider_img = image_dir / "slider.png"
    success_catch_img = image_dir / "success_catch.png"

    slider_template = cv2.imread(str(slider_img), cv2.IMREAD_COLOR)
    valid_region_left_template = cv2.imread(str(valid_region_left_img), cv2.IMREAD_COLOR)
    valid_region_right_template = cv2.imread(str(valid_region_right_img), cv2.IMREAD_COLOR)
    continue_template = cv2.imread(str(continue_img), cv2.IMREAD_COLOR)
    success_catch_template = cv2.imread(str(success_catch_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, _argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Autofish Action Started ===")
        controller = context.tasker.controller

        KEY_A = 65
        KEY_D = 68
        KEY_F = 70
        KEY_ESC = 27

        success_region = (350,150,830,200)
        settlement_region = (480, 610, 800, 690)
        game_region = (395, 40, 880, 60)
        deadzone = 15

        # --- 抛竿 ---
        controller.post_key_down(KEY_F)
        time.sleep(0.1)
        controller.post_key_up(KEY_F)
        print("  Casting...")

        # --- 等待鱼上钩 ---
        wait_frame = 0
        while True:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            time.sleep(0.001)
            img = get_image(controller)
            wait_frame += 1
            m_catch, catch_score, _, _ = match_template_in_region(img, success_region, self.success_catch_template, 0.7)
            if wait_frame % 100 == 0:
                print(f"  [wait] frame={wait_frame} catch_score={catch_score:.3f}")
            if m_catch:
                controller.post_key_down(KEY_F)
                time.sleep(0.1)
                controller.post_key_up(KEY_F)
                print(f"  Fish hooked! (score={catch_score:.3f})")
                break

        # --- 小游戏 ---
        frame = 0
        current_ad_key = None
        last_bar_width = 100
        last_target = (game_region[0] + game_region[2]) / 2
        last_x_slider = last_target
        slider_miss_count = 0

        def set_ad_key(key):
            nonlocal current_ad_key
            if current_ad_key == key:
                return
            if current_ad_key is not None:
                controller.post_key_up(current_ad_key)
            if key is not None:
                controller.post_key_down(key)
            current_ad_key = key

        while True:
            if context.tasker.stopping:
                set_ad_key(None)
                return CustomAction.RunResult(success=False)
            time.sleep(0.001)
            img = get_image(controller)
            frame += 1

            m_left, left_score, x_left, _ = match_template_in_region(img, game_region, self.valid_region_left_template, 0.7)
            m_right, right_score, x_right, _ = match_template_in_region(img, game_region, self.valid_region_right_template, 0.7)
            m_slider, slider_score, x_slider, _ = match_template_in_region(img, game_region, self.slider_template, 0.9)

            if frame % 10 == 0:
                if current_ad_key is not None:
                    controller.post_key_up(current_ad_key)
                controller.post_key_down(KEY_F)
                time.sleep(0.05)
                controller.post_key_up(KEY_F)
                if current_ad_key is not None:
                    controller.post_key_down(current_ad_key)

            if m_slider:
                slider_miss_count = 0
                last_x_slider = x_slider
            else:
                slider_miss_count += 1
                if slider_miss_count >= 20:
                    set_ad_key(None)
                    controller.post_key_up(KEY_F)
                    print(f"  [minigame] slider lost {slider_miss_count} frames, minigame ended.")
                    break
                x_slider = last_x_slider

            if frame > 300:
                set_ad_key(None)
                controller.post_key_up(KEY_F)
                print(f"  [minigame] timeout (f={frame}), minigame ended.")
                break

            if m_left and m_right:
                last_bar_width = x_right - x_left
                target = (x_left + x_right) / 2
                last_target = target
            elif m_left and not m_right:
                target = x_left + last_bar_width / 2
                last_target = target
            elif not m_left and m_right:
                target = x_right - last_bar_width / 2
                last_target = target
            else:
                target = last_target

            offset = x_slider - target
            prev_key = current_ad_key
            if offset > deadzone:
                set_ad_key(KEY_A)
            elif offset < -deadzone:
                set_ad_key(KEY_D)
            else:
                set_ad_key(None)

            if frame % 30 == 0 or current_ad_key != prev_key:
                key_name = {None: "-", KEY_A: "A", KEY_D: "D"}.get(current_ad_key, "?")
                print(f"  [minigame] f={frame} slider(x={x_slider:.0f} s={slider_score:.2f}) "
                      f"L({m_left} s={left_score:.2f}) R({m_right} s={right_score:.2f}) "
                      f"bar_w={last_bar_width:.0f} target={target:.0f} offset={offset:+.0f} key={key_name}")

        # --- 小游戏结束，检查是否钓上鱼 ---
        time.sleep(1)
        img = get_image(controller)
        match_settle, _, _, _ = match_template_in_region(img, settlement_region, self.continue_template, 0.8)
        if match_settle:
            print("  Fish caught! Closing settlement...")
            for _ in range(5):
                controller.post_key_down(KEY_ESC)
                time.sleep(0.1)
                controller.post_key_up(KEY_ESC)
                time.sleep(1.5)
                img = get_image(controller)
                m, _, _, _ = match_template_in_region(img, settlement_region, self.continue_template, 0.8)
                if not m:
                    print("  Settlement closed.")
                    break
        else:
            print("  Fish escaped or no settlement.")

        print("  Fishing done.")
        return CustomAction.RunResult(success=True)
