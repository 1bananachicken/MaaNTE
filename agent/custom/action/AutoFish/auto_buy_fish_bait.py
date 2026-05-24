import cv2
import time
import json

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region, click_rect
from utils import screen

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT


@AgentServer.custom_action("auto_buy_fish_bait")
class AutoBuyFishBait(CustomAction):
    abs_path = Path(__file__).parents[4]
    if Path.exists(abs_path / "assets"):
        image_dir = abs_path / "assets/resource/base/image/auto_buy_fish_bait"
    else:
        image_dir = abs_path / "resource/base/image/auto_buy_fish_bait"
    bait_img = image_dir / "bait.png"
    find_bait_success_img = image_dir / "find_bait_success.png"
    select_max_img = image_dir / "select_max.png"
    buy_img = image_dir / "buy.png"
    buy_confirm_img = image_dir / "buy_confirm.png"
    buy_success_img = image_dir / "buy_success.png"
    bait_template = cv2.imread(str(bait_img), cv2.IMREAD_COLOR)
    find_bait_success_template = cv2.imread(
        str(find_bait_success_img), cv2.IMREAD_COLOR
    )
    select_max_template = cv2.imread(str(select_max_img), cv2.IMREAD_COLOR)
    buy_template = cv2.imread(str(buy_img), cv2.IMREAD_COLOR)
    buy_confirm_template = cv2.imread(str(buy_confirm_img), cv2.IMREAD_COLOR)
    buy_success_template = cv2.imread(str(buy_success_img), cv2.IMREAD_COLOR)

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        fish_shop_region = [35, 88, 410, 475]
        find_bait_success_region = [1044, 131, 68, 23]
        select_max_region = [1202, 620, 33, 32]
        buy_region = [1050, 674, 50, 25]
        buy_confirm_region = [749, 462, 47, 25]
        buy_success_region = [569, 629, 145, 19]
        not_enough_shell_region = [1170, 585, 18, 16]
        shell_count_region = [961, 31, 70, 21]

        fish_shop_region = screen.map_rect(fish_shop_region)
        find_bait_success_region = screen.map_rect(find_bait_success_region)
        select_max_region = screen.map_rect(select_max_region)
        buy_region = screen.map_rect(buy_region)
        buy_confirm_region = screen.map_rect(buy_confirm_region)
        buy_success_region = screen.map_rect(buy_success_region)
        not_enough_shell_region = screen.map_rect(not_enough_shell_region)
        shell_count_region = screen.map_rect(shell_count_region)
        KEY_R = 82
        KEY_ESC = 27
        controller = context.tasker.controller

        found_bait_threshold = 0.7
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                found_bait_threshold = params.get("found_bait_threshold", 0.7)
            except:
                pass

        PrintT(context, "autofish.buy_bait_started")

        match_threshold = 0.7
        while True:
            img = get_image(controller)
            found_bait, prob, x, y = match_template_in_region(
                img, fish_shop_region, self.bait_template, found_bait_threshold
            )
            logger.debug(
                "Current found bait threshold: %s, match probability: %.2f, clicked on bait at (%d, %d)",
                found_bait_threshold,
                prob,
                x + 15,
                y + 5,
            )
            if found_bait:
                controller.post_touch_move(
                    x, y
                )  # 先移动到指定位置再进行点击，否则可能会触发滑动买到别的东东
                for _ in range(3):
                    click_rect(controller, [x, y, 30, 10])
                    time.sleep(0.1)

                time.sleep(1)  # 120hz下可能会过快地出现触发检测。适当的延时
                img = get_image(controller)
                found_bait_success, _, _, _ = match_template_in_region(
                    img,
                    find_bait_success_region,
                    self.find_bait_success_template,
                    match_threshold,
                )
                if found_bait_success:
                    time.sleep(0.5)
                    break
                else:
                    PrintT(context, "autofish.bait_retry_click")
                    time.sleep(1)
            else:
                PrintT(context, "autofish.bait_not_found")
                controller.post_click_key(KEY_R).wait()
                time.sleep(1)
                continue

        while True:
            img = get_image(controller)
            found_select_max, prob, _, _ = match_template_in_region(
                img, select_max_region, self.select_max_template, match_threshold
            )
            logger.debug("Looking for select max option, match probability: %.2f", prob)
            if found_select_max:
                PrintT(context, "autofish.select_max_found")
                for _ in range(5):
                    click_rect(controller, select_max_region, 0.3)
                    time.sleep(0.1)
                time.sleep(1)
                break
            else:
                PrintT(context, "autofish.select_max_retry")
                time.sleep(1)

        while True:
            img = get_image(controller)
            found_buy, _, _, _ = match_template_in_region(
                img, buy_region, self.buy_template, match_threshold
            )
            if found_buy:
                PrintT(context, "autofish.buy_button_click")
                for _ in range(3):
                    click_rect(controller, buy_region, 0.3)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                PrintT(context, "autofish.buy_button_retry")
                time.sleep(1)

        for _ in range(5):
            img = get_image(controller)
            found_buy_confirm, _, _, _ = match_template_in_region(
                img, buy_confirm_region, self.buy_confirm_template, match_threshold
            )
            if found_buy_confirm:
                PrintT(context, "autofish.buy_confirm_click")
                for _ in range(3):
                    click_rect(controller, buy_confirm_region)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                PrintT(context, "autofish.buy_confirm_retry")
                time.sleep(1)

        while True:
            img = get_image(controller)
            found_buy_success, _, _, _ = match_template_in_region(
                img, buy_success_region, self.buy_success_template, match_threshold
            )
            if found_buy_success:
                PrintT(context, "autofish.buy_success")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                PrintT(context, "autofish.buy_success_retry")
                time.sleep(1)

        return CustomAction.RunResult(success=True)
