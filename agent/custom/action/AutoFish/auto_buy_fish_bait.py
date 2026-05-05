import cv2
import time
import json

from pathlib import Path
from ..Common.utils import get_image, match_template_in_region, click_rect
from ..Common.logger import get_logger

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = get_logger("auto_buy_fish_bait")


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
    find_bait_success_template = cv2.imread(str(find_bait_success_img), cv2.IMREAD_COLOR)
    select_max_template = cv2.imread(str(select_max_img), cv2.IMREAD_COLOR)
    buy_template = cv2.imread(str(buy_img), cv2.IMREAD_COLOR)
    buy_confirm_template = cv2.imread(str(buy_confirm_img), cv2.IMREAD_COLOR)
    buy_success_template = cv2.imread(str(buy_success_img), cv2.IMREAD_COLOR)

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        fish_shop_region = [35, 88, 410, 475]
        find_bait_success_region = [1044, 131, 68, 23]
        select_max_region = [1202, 620, 33, 32]
        buy_region = [1050, 674, 50, 25]
        buy_confirm_region = [749, 462, 47, 25]
        buy_success_region = [569, 629, 145, 19]
        not_enough_shell_region = [1170, 585, 18, 16]
        shell_count_region = [961, 31, 70, 21]
        KEY_R = 82
        KEY_ESC = 27
        controller = context.tasker.controller  
        
        found_bait_threshold = 0.8
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                found_bait_threshold = params.get("found_bait_threshold", 0.8)
            except:
                pass
        
        logger.info(f"买鱼饵流程开始 (阈值={found_bait_threshold:.2f})")

        match_threshold = 0.7
        logger.info("步骤 1/4: 在鱼店中定位鱼饵")
        find_bait_attempt = 0
        while True:
            find_bait_attempt += 1
            img = get_image(controller)
            found_bait, prob, x, y = match_template_in_region(img, fish_shop_region, self.bait_template, found_bait_threshold)
            logger.debug(f"第 {find_bait_attempt} 次尝试定位鱼饵: 匹配度={prob:.2f}, 阈值={found_bait_threshold}")
            if found_bait:
                logger.info(f"鱼饵已定位 ({x+15},{y+5}), 点击中")
                for _ in range(3):
                    click_rect(controller, [x, y, 30, 10])
                    time.sleep(0.1)

                img = get_image(controller)
                found_bait_success, _, _, _ = match_template_in_region(img, find_bait_success_region, self.find_bait_success_template, match_threshold)
                if found_bait_success:
                    time.sleep(0.5)
                    break
                else:
                    logger.warning("鱼饵已找到但点击未生效，重试...")
                    time.sleep(1)
            else:
                logger.warning(f"鱼店中未找到鱼饵 (第 {find_bait_attempt} 次)，刷新店铺后重试")
                controller.post_click_key(KEY_R).wait()
                time.sleep(1)
                continue

        logger.info("步骤 2/4: 选择最大购买数量")
        select_max_attempt = 0
        while True:
            select_max_attempt += 1
            img = get_image(controller)
            found_select_max, prob, _, _ = match_template_in_region(img, select_max_region, self.select_max_template, match_threshold)
            logger.debug(f"第 {select_max_attempt} 次查找“最大数量”按钮: 匹配度={prob:.2f}")
            if found_select_max:
                logger.info("已找到“最大数量”按钮，点击中")
                for _ in range(5):
                    click_rect(controller, select_max_region, 0.3)
                    time.sleep(0.1)
                time.sleep(1)
                break
            else:
                logger.debug("未找到“最大数量”按钮，重试...")
                time.sleep(1)

        logger.info("步骤 3/4: 点击“购买”")
        buy_attempt = 0
        while True:
            buy_attempt += 1
            img = get_image(controller)
            found_buy, _, _, _ = match_template_in_region(img, buy_region, self.buy_template, match_threshold)
            if found_buy:
                logger.info("已找到“购买”按钮，点击中")
                for _ in range(3):
                    click_rect(controller, buy_region, 0.3)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                logger.debug(f"未找到“购买”按钮 (第 {buy_attempt} 次)，重试...")
                time.sleep(1)

        confirm_attempt = 0
        for _ in range(5):
            confirm_attempt += 1
            img = get_image(controller)
            found_buy_confirm, _, _, _ = match_template_in_region(img, buy_confirm_region, self.buy_confirm_template, match_threshold)
            if found_buy_confirm:
                logger.info("已找到“确认购买”按钮，点击中")
                for _ in range(3):
                    click_rect(controller, buy_confirm_region)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                logger.debug(f"未找到“确认购买”按钮 (第 {confirm_attempt} 次)，重试...")
                time.sleep(1)

        logger.info("步骤 4/4: 等待购买成功提示")
        success_attempt = 0
        while True:
            success_attempt += 1
            img = get_image(controller)
            found_buy_success, _, _, _ = match_template_in_region(img, buy_success_region, self.buy_success_template, match_threshold)
            if found_buy_success:
                logger.info("买鱼饵成功，关闭店铺")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                logger.debug(f"未检测到购买成功 (第 {success_attempt} 次)，继续等待...")
                time.sleep(1)

        return CustomAction.RunResult(success=True)
