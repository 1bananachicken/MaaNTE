import time
import json
import cv2
from pathlib import Path
from .utils import get_image, match_template_in_region, click_rect

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("auto_sell_fish")
class AutoSellFish(CustomAction):
    abs_path = Path(__file__).parents[3]
    if Path.exists(abs_path / "assets"):
            image_dir = abs_path / "assets/resource/base/image/auto_sell_fish"
    else:
        image_dir = abs_path / "resource/base/image/auto_sell_fish"
    
    sell_option_img = image_dir / "sell_option_gray.png"
    sell_option_selected_img = image_dir / "sell_option.png"
    no_fish_to_sell_img = image_dir / "no_fish.png"
    sell_button_img = image_dir / "sell_button.png"
    confirm_sell_img = image_dir / "confirm_sell.png"
    sell_success_img = image_dir / "sell_success.png"
    sell_option_template = cv2.imread(str(sell_option_img), cv2.IMREAD_COLOR)
    sell_option_selected_template = cv2.imread(str(sell_option_selected_img), cv2.IMREAD_COLOR)
    no_fish_to_sell_template = cv2.imread(str(no_fish_to_sell_img), cv2.IMREAD_COLOR)
    sell_button_template = cv2.imread(str(sell_button_img), cv2.IMREAD_COLOR)
    confirm_sell_template = cv2.imread(str(confirm_sell_img), cv2.IMREAD_COLOR)
    sell_success_template = cv2.imread(str(sell_success_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Autofish Action Started ===")
        controller = context.tasker.controller

        KEY_Q = 81
        KEY_ESC = 27
        
        no_fish_to_sell_region = [433, 457, 77, 20]
        sell_option_region = [63, 247, 66, 57]
        sell_option_selected_region = [172, 166, 103, 29]
        sell_button_region = [665, 635, 92, 23]
        confirm_sell_region = [756, 461, 48, 21]
        sell_success_region = [565, 628, 149, 21]
        
        while True:
            img = get_image(controller)
            found_sell_option, _, _, _ = match_template_in_region(img, sell_option_region, self.sell_option_template, 0.7)
            if found_sell_option:
                for _ in range(3):
                    click_rect(controller, sell_option_region)
                    time.sleep(0.1)
                
                img = get_image(controller)
                found_sell_option_selected, _, _, _ = match_template_in_region(img, sell_option_selected_region, self.sell_option_selected_template, 0.8)
                if found_sell_option_selected:
                    break
                time.sleep(1)  
            else:
                controller.post_click_key(KEY_Q).wait()  
                time.sleep(1)  

        print("Sell option detected. Proceeding to sell fish.")

        for _ in range(5):
            img = get_image(controller)
            found_no_fish_to_sell, prob, _, _ = match_template_in_region(img, no_fish_to_sell_region, self.no_fish_to_sell_template, 0.8)
            time.sleep(0.1)
            if found_no_fish_to_sell:
                print("No fish to sell detected. Closing fish shop.")
                controller.post_click_key(KEY_ESC).wait()  
                return CustomAction.RunResult(success=True)
        
        while True:
            img = get_image(controller)
            found_sell_button, _, _, _ = match_template_in_region(img, sell_button_region, self.sell_button_template, 0.8)
            if found_sell_button:
                print("Sell button detected. Clicking to confirm selling fish.")
                while True:
                    for _ in range(3):
                        click_rect(controller, sell_button_region)
                        time.sleep(0.1)
                        
                    img = get_image(controller)
                    found_confirm_sell, _, _, _ = match_template_in_region(img, confirm_sell_region, self.confirm_sell_template, 0.8)
                    if found_confirm_sell:
                        print("Confirm sell button detected. Clicking to confirm selling fish.")
                        for _ in range(3):
                            click_rect(controller, confirm_sell_region)
                            time.sleep(0.1)
                        time.sleep(1)  
                        break
                    else:
                        time.sleep(0.1)
                break
            else:
                time.sleep(0.1)

        while True:
            img = get_image(controller)
            found_sell_success, _, _, _ = match_template_in_region(img, sell_success_region, self.sell_success_template, 0.8)
            if found_sell_success:
                print("Sell success detected. Fish sold successfully.")
                controller.post_click_key(KEY_ESC).wait()  
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                time.sleep(1)
        
        print("All fishing tasks complete.")
        return CustomAction.RunResult(success=True)
