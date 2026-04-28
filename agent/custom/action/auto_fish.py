import time
import json
from pathlib import Path
import cv2
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


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
class Autofish(CustomAction):
    abs_path = Path(__file__).parents[3]
    if Path.exists(abs_path / "assets"):
            image_dir = abs_path / "assets/resource/base/image/auto_fish"
    else:
        image_dir = abs_path / "resource/base/image/auto_fish"
    continue_img = image_dir / "continue.png"
    valid_region_left_img = image_dir / "valid_region_left.png"
    valid_region_right_img = image_dir / "valid_region_right.png"
    slider_img = image_dir / "slider.png"
    success_catch_img = image_dir / "success_catch.png"
    escape_img = image_dir / "escape.png"

    slider_template = cv2.imread(str(slider_img), cv2.IMREAD_COLOR)
    valid_region_left_template = cv2.imread(str(valid_region_left_img), cv2.IMREAD_COLOR)
    valid_region_right_template = cv2.imread(str(valid_region_right_img), cv2.IMREAD_COLOR)
    continue_template = cv2.imread(str(continue_img), cv2.IMREAD_COLOR)
    success_catch_template = cv2.imread(str(success_catch_img), cv2.IMREAD_COLOR)
    escape_template = cv2.imread(str(escape_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Autofish Action Started ===")
        controller = context.tasker.controller
        # 分辨率检测
        try:
            controller.post_screencap().wait()
            w, h = controller.resolution
            logger.warning(f"[分辨率检测] 获取到的分辨率: {w}x{h}")
        except Exception as e:
            logger.error(f"[分辨率检测] 异常: {e}")

        fishing_count = 10
        check_freq = 0.001
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                fishing_count = params.get("count", 10)
                check_freq = params.get("freq", 0.1)
            except:
                pass

        # Key codes for A, D, F, ESC
        KEY_A = 65
        KEY_D = 68
        KEY_F = 70
        KEY_ESC = 27

        # Original coordinates from autofish.py
        success_region = (520, 160, 785, 190)
        settlement_region = (564, 642, 1206, 664)
        game_region = (400, 33, 882, 63)
        escape_region = (590, 349, 689, 371)

        for i in range(fishing_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            print(f"=== Fishing {i + 1}/{fishing_count} ===")

            # 1. Clear settlement screen
            img = get_image(controller)
            match_settle, _, _, _ = match_template_in_region(img, settlement_region, self.continue_template, 0.8)
            if match_settle:
                print("  Closing settlement screen...")
                for _ in range(5):
                    controller.post_key_down(KEY_ESC)
                    time.sleep(0.1)
                    controller.post_key_up(KEY_ESC)
                    time.sleep(1)

                    img = get_image(controller)
                    m, _, _, _ = match_template_in_region(img, settlement_region, self.continue_template, 0.8)
                    if not m:
                        print("  Settlement closed.")
                        break

            # 2+3. Fish-and-reel loop (retry on escape)
            while True:
                # 2. Cast and wait for fish to bite
                controller.post_key_down(KEY_F)
                time.sleep(0.1)
                controller.post_key_up(KEY_F)
                print("  Casting...")

                while True:
                    time.sleep(check_freq)
                    img = get_image(controller)
                    m_catch, _, _, _ = match_template_in_region(img, success_region, self.success_catch_template, 0.8)
                    if m_catch:
                        controller.post_key_down(KEY_F)
                        time.sleep(0.1)
                        controller.post_key_up(KEY_F)
                        print("  Fish hooked!")
                        break

                # 3. Minigame: reel in and balance slider
                start_time = time.time()
                frame = 0
                deadzone = 15

                while time.time() - start_time < 100:
                    time.sleep(check_freq)
                    img = get_image(controller)
                    frame += 1

                    if frame % 10 == 0:
                        m_settle, _, _, _ = match_template_in_region(img, settlement_region, self.continue_template, 0.8)
                        if m_settle:
                            print("  Fish caught!")
                            break
                        m_escape, _, _, _ = match_template_in_region(img, escape_region, self.escape_template, 0.8)
                        if m_escape:
                            print("  Fish escaped! Recasting...")
                            break

                    m_left, _, x_left, _ = match_template_in_region(img, game_region, self.valid_region_left_template, 0.7)
                    m_right, _, x_right, _ = match_template_in_region(img, game_region, self.valid_region_right_template, 0.7)
                    m_slider, _, x_slider, _ = match_template_in_region(img, game_region, self.slider_template, 0.7)

                    if m_slider:
                        # Reel in (throttled)
                        if frame % 10 == 0:
                            controller.post_key_down(KEY_F)
                            time.sleep(0.05)
                            controller.post_key_up(KEY_F)

                        # Balance slider
                        if m_left and m_right:
                            target = (x_left + x_right) / 2
                            offset = x_slider - target
                        elif not m_left and m_right:
                            target = x_right
                            offset = x_slider - target
                        elif m_left and not m_right:
                            target = x_left
                            offset = x_slider - target
                        else:
                            offset = 0

                        if m_left or m_right:
                            if offset > deadzone:
                                controller.post_key_up(KEY_D)
                                controller.post_key_down(KEY_A)
                            elif offset < -deadzone:
                                controller.post_key_up(KEY_A)
                                controller.post_key_down(KEY_D)
                            else:
                                controller.post_key_up(KEY_A)
                                controller.post_key_up(KEY_D)

                # Release all keys
                controller.post_key_up(KEY_D)
                controller.post_key_up(KEY_A)
                controller.post_key_up(KEY_F)

                # Check why the minigame loop ended
                img = get_image(controller)
                time.sleep(0.3)
                m_escape, _, _, _ = match_template_in_region(img, escape_region, self.escape_template, 0.8)
                if m_escape:
                    continue  # Retry: go back to Stage 2 (cast again)
                break  # Fish caught or timeout: move to next fish

            print("  Finished.")

        print("All fishing tasks complete.")
        return CustomAction.RunResult(success=True)
