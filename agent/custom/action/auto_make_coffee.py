import time
import json
from pathlib import Path
import cv2
import numpy as np
import random

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
    
    x1, y1, w, h = region
    x2, y2 = x1 + w, y1 + h
    
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    
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

def click_rect(controller, rect):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    controller.post_click(cx, cy).wait()

@AgentServer.custom_action("auto_make_coffee")
class AutoMakeCoffee(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        if Path.exists(abs_path / "assets"):
            image_dir = abs_path / "assets/resource/base/image/auto_make_coffee"
        else:
            image_dir = abs_path / "resource/base/image/auto_make_coffee"
        start_img = image_dir / "start.png"
        star_img = image_dir / "star.png"
        claim_img = image_dir / "claim.png" 

        self.start_template = cv2.imread(str(start_img), cv2.IMREAD_COLOR)
        self.star_template = cv2.imread(str(star_img), cv2.IMREAD_COLOR)
        self.claim_template = cv2.imread(str(claim_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Auto Make Coffee Action Started ===")
        controller = context.tasker.controller
        
        make_count = 10
        check_freq = 0.5
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                make_count = params.get("count", 10)
                check_freq = params.get("freq", 0.5)
            except:
                pass
                
        KEY_F = 70
        
        # Coordinates mapped from auto_make_coffee.json [x, y, w, h]
        select_level_target = [18, 230, 188, 66]
        click_roi = [28, 272, 65, 56]
        start_roi = [1057, 648, 178, 44]
        star_roi = [1204, 109, 29, 27]
        star_target = [11, 12, 38, 37]
        claim_roi = [681, 539, 187, 38]
        
        for count in range(make_count):
            if context.tasker.stopping:  
                return CustomAction.RunResult(success=False)
            print(f"=== Making Coffee {count + 1}/{make_count} ===")

            # Step 1: 选择关卡
            print("Tapping on select level...")
            click_rect(controller, select_level_target)
            time.sleep(1)
            
            # Step 2: 开始营业
            print("Waiting for start business button...")
            while True:
                if context.tasker.stopping:  
                    return CustomAction.RunResult(success=False)
                img = get_image(controller)
                match_start, _, match_x, match_y = match_template_in_region(img, start_roi, self.start_template, 0.8)
                if match_start:
                    print("Found 'start.png', clicking...")
                    click_rect(controller, [match_x, match_y, self.start_template.shape[1], self.start_template.shape[0]])
                    time.sleep(3) # Post delay from JSON: 3000ms
                    break
                time.sleep(check_freq)

            # Step 3: 达成营业额
            print("Waiting for star to reach sales goal...")
            while True:
                if context.tasker.stopping:  
                    return CustomAction.RunResult(success=False)
                click_rect(controller, click_roi)
                img = get_image(controller)
                match_star, _, _, _ = match_template_in_region(img, star_roi, self.star_template, 0.9)
                if match_star:
                    print("Found 'star.png', clicking target...")
                    click_rect(controller, star_target)
                    time.sleep(1)
                    break
                time.sleep(2)

            # Step 4: 点击领取
            print("Waiting to claim reward...")
            while True:
                if context.tasker.stopping:  
                    return CustomAction.RunResult(success=False)
                img = get_image(controller)
                match_claim, _, match_x, match_y = match_template_in_region(img, claim_roi, self.claim_template, 0.8)
                if match_claim:
                    print("Found 'claim.png', clicking...")
                    click_rect(controller, [match_x, match_y, self.claim_template.shape[1], self.claim_template.shape[0]])
                    time.sleep(1)
                    break
                time.sleep(check_freq)
            
            print("Round finished. Pressing 'F' to continue...")
            controller.post_key_down(KEY_F)
            time.sleep(0.1)
            controller.post_key_up(KEY_F)
            
            time.sleep(2)
            print("Current iteration finished.\n")
            
        print("All coffee tasks complete.")
        return CustomAction.RunResult(success=True)
