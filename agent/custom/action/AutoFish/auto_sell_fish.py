import cv2
import time

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region, click_rect
from ..Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = get_logger("auto_sell_fish")


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
    sell_option_selected_template = cv2.imread(str(sell_option_selected_img), cv2.IMREAD_COLOR)
    no_fish_to_sell_template = cv2.imread(str(no_fish_to_sell_img), cv2.IMREAD_COLOR)
    sell_button_template = cv2.imread(str(sell_button_img), cv2.IMREAD_COLOR)
    confirm_sell_template = cv2.imread(str(confirm_sell_img), cv2.IMREAD_COLOR)
    sell_success_template = cv2.imread(str(sell_success_img), cv2.IMREAD_COLOR)
    sell_fail_template = cv2.imread(str(sell_fail_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        logger.info("卖鱼流程开始")
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
        
        logger.info("步骤 1/3: 切换到卖鱼选项")
        sell_option_attempt = 0
        while True:
            sell_option_attempt += 1
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
                logger.debug(f"  未找到卖鱼选项 (第 {sell_option_attempt} 次)，按 Q 后重试")
                controller.post_click_key(KEY_Q).wait()
                time.sleep(1)

        logger.info("已进入卖鱼界面")

        logger.info("步骤 2/3: 检查是否有可卖鱼")
        for check_n in range(1, 6):
            img = get_image(controller)
            found_no_fish_to_sell, prob, _, _ = match_template_in_region(img, no_fish_to_sell_region, self.no_fish_to_sell_template, 0.8)
            logger.debug(f"  第 {check_n}/5 次检查“无鱼可卖”: 匹配度={prob:.2f}")
            time.sleep(0.1)
            if found_no_fish_to_sell:
                logger.info("无鱼可卖，关闭店铺")
                controller.post_click_key(KEY_ESC).wait()
                return CustomAction.RunResult(success=True)

        time.sleep(1.5)

        logger.info("步骤 3/3: 卖出当前鱼")
        sell_button_attempt = 0
        while True:
            sell_button_attempt += 1
            img = get_image(controller)
            found_sell_button, _, _, _ = match_template_in_region(img, sell_button_region, self.sell_button_template, 0.8)
            if found_sell_button:
                logger.info("  已找到“卖出”按钮，点击中")
                confirm_attempt = 0
                while True:
                    confirm_attempt += 1
                    click_rect(controller, sell_button_region, 0.1)
                    time.sleep(0.5)
                    img = get_image(controller)
                    found_confirm_sell, _, _, _ = match_template_in_region(img, confirm_sell_region, self.confirm_sell_template, 0.8)
                    sell_fail, _, _, _ = match_template_in_region(img, sell_fail_region, self.sell_fail_template, 0.8)
                    if found_confirm_sell:
                        logger.info(f"  已找到“确认卖出”按钮 (第 {confirm_attempt} 次)，确认中")
                        click_rect(controller, confirm_sell_region, 0.2)
                        time.sleep(0.5)
                        break
                    elif sell_fail:
                        logger.info("无鱼可卖，关闭店铺")
                        controller.post_click_key(KEY_ESC).wait()
                        return CustomAction.RunResult(success=True)
                    else:
                        time.sleep(0.1)
                break
            else:
                logger.debug(f"  未找到“卖出”按钮 (第 {sell_button_attempt} 次)")
                time.sleep(0.1)

        success_attempt = 0
        while True:
            success_attempt += 1
            img = get_image(controller)
            found_sell_success, _, _, _ = match_template_in_region(img, sell_success_region, self.sell_success_template, 0.8)
            if found_sell_success:
                logger.info("卖鱼成功，关闭店铺")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                logger.debug(f"  未检测到卖出成功 (第 {success_attempt} 次)，继续等待...")
                time.sleep(1)

        logger.info("卖鱼流程完成")
        return CustomAction.RunResult(success=True)
