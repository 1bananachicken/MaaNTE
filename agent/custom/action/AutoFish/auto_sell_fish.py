import cv2
import time

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region, click_rect
from utils import screen

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.maafocus import PrintT


@AgentServer.custom_action("auto_sell_fish")
class AutoSellFish(CustomAction):
    abs_path = Path(__file__).parents[4]
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
    sell_fail_img = image_dir / "sell_fail.png"
    sell_option_template = cv2.imread(str(sell_option_img), cv2.IMREAD_COLOR)
    sell_option_selected_template = cv2.imread(
        str(sell_option_selected_img), cv2.IMREAD_COLOR
    )
    no_fish_to_sell_template = cv2.imread(str(no_fish_to_sell_img), cv2.IMREAD_COLOR)
    sell_button_template = cv2.imread(str(sell_button_img), cv2.IMREAD_COLOR)
    confirm_sell_template = cv2.imread(str(confirm_sell_img), cv2.IMREAD_COLOR)
    sell_success_template = cv2.imread(str(sell_success_img), cv2.IMREAD_COLOR)
    sell_fail_template = cv2.imread(str(sell_fail_img), cv2.IMREAD_COLOR)

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        PrintT(context, "autofish.sell_started")
        controller = context.tasker.controller

        KEY_Q = 81
        KEY_ESC = 27

        no_fish_to_sell_region = [433, 457, 77, 20]
        sell_option_region = [63, 247, 66, 57]
        sell_option_selected_region = [172, 166, 103, 29]
        sell_button_region = [665, 635, 92, 23]
        confirm_sell_region = [756, 461, 48, 21]
        sell_success_region = [565, 628, 149, 21]
        sell_fail_region = [739, 349, 202, 24]
        no_valid_fish_region = [509, 350, 261, 22]

        no_fish_to_sell_region = screen.map_rect(no_fish_to_sell_region)
        sell_option_region = screen.map_rect(sell_option_region)
        sell_option_selected_region = screen.map_rect(sell_option_selected_region)
        sell_button_region = screen.map_rect(sell_button_region)
        confirm_sell_region = screen.map_rect(confirm_sell_region)
        sell_success_region = screen.map_rect(sell_success_region)
        sell_fail_region = screen.map_rect(sell_fail_region)
        no_valid_fish_region = screen.map_rect(no_valid_fish_region)

        while True:
            img = get_image(controller)
            found_sell_option, _, _, _ = match_template_in_region(
                img, sell_option_region, self.sell_option_template, 0.7
            )
            if found_sell_option:
                for _ in range(3):
                    click_rect(controller, sell_option_region)
                    time.sleep(0.1)

                img = get_image(controller)
                found_sell_option_selected, _, _, _ = match_template_in_region(
                    img,
                    sell_option_selected_region,
                    self.sell_option_selected_template,
                    0.8,
                )
                if found_sell_option_selected:
                    break
                time.sleep(1)
            else:
                controller.post_click_key(KEY_Q).wait()
                time.sleep(1)

        PrintT(context, "autofish.sell_option_detected")

        for _ in range(5):
            img = get_image(controller)
            found_no_fish_to_sell, prob, _, _ = match_template_in_region(
                img, no_fish_to_sell_region, self.no_fish_to_sell_template, 0.8
            )
            time.sleep(0.1)
            if found_no_fish_to_sell:
                PrintT(context, "autofish.no_fish_detected")
                controller.post_click_key(KEY_ESC).wait()
                return CustomAction.RunResult(success=True)

        time.sleep(1.5)

        while True:
            img = get_image(controller)
            found_sell_button, _, _, _ = match_template_in_region(
                img, sell_button_region, self.sell_button_template, 0.8
            )
            if found_sell_button:
                PrintT(context, "autofish.sell_button_detected")
                while True:
                    click_rect(controller, sell_button_region, 0.1)
                    time.sleep(0.5)
                    img = get_image(controller)
                    found_confirm_sell, _, _, _ = match_template_in_region(
                        img, confirm_sell_region, self.confirm_sell_template, 0.8
                    )
                    sell_fail, _, _, _ = match_template_in_region(
                        img, sell_fail_region, self.sell_fail_template, 0.8
                    )
                    if found_confirm_sell:
                        PrintT(context, "autofish.confirm_sell_detected")
                        click_rect(controller, confirm_sell_region, 0.2)
                        time.sleep(0.5)
                        break
                    elif sell_fail:
                        PrintT(context, "autofish.no_fish_sell_close")
                        controller.post_click_key(KEY_ESC).wait()
                        return CustomAction.RunResult(success=True)
                    else:
                        time.sleep(0.1)
                break
            else:
                time.sleep(0.1)

        while True:
            img = get_image(controller)
            found_sell_success, _, _, _ = match_template_in_region(
                img, sell_success_region, self.sell_success_template, 0.8
            )
            if found_sell_success:
                PrintT(context, "autofish.sell_success")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                time.sleep(1)

        PrintT(context, "autofish.all_done")
        return CustomAction.RunResult(success=True)
