import time
import json

from ..Common.utils import get_image, click_rect

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("auto_buy_fish_bait")
class AutoBuyFishBait(CustomAction):

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        select_max_region = [1202, 620, 33, 32]
        buy_region = [1050, 674, 50, 25]
        buy_confirm_region = [749, 462, 47, 25]
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

        print("=== AutoBuyFishBait Action Started ===")

        match_threshold = 0.7
        while True:
            img = get_image(controller)
            bait_result = context.run_recognition(
                "BuyFishBaitItem",
                img,
                pipeline_override={
                    "BuyFishBaitItem": {
                        "recognition": {"param": {"threshold": found_bait_threshold}}
                    }
                },
            )
            if bait_result and bait_result.hit:
                x, y = bait_result.box.x, bait_result.box.y
                controller.post_touch_move(
                    x, y
                )  # 先移动到指定位置再进行点击，否则可能会触发滑动买到别的东东
                for _ in range(3):
                    click_rect(controller, [x, y, 30, 10])
                    time.sleep(0.1)

                time.sleep(1)
                img = get_image(controller)
                bait_success_result = context.run_recognition(
                    "BuyFishBaitFindSuccess", img
                )
                if bait_success_result and bait_success_result.hit:
                    time.sleep(0.5)
                    break
                else:
                    print("Bait found but not click correctly, retrying...")
                    time.sleep(1)
            else:
                print("Bait not found in fish shop, retrying...")
                controller.post_click_key(KEY_R).wait()
                time.sleep(1)
                continue

        while True:
            img = get_image(controller)
            select_max_result = context.run_recognition("BuyFishBaitSelectMax", img)
            if select_max_result and select_max_result.hit:
                print("Select max option found, clicking...")
                for _ in range(5):
                    click_rect(controller, select_max_region, 0.3)
                    time.sleep(0.1)
                time.sleep(1)
                break
            else:
                print("Select max option not found, retrying...")
                time.sleep(1)

        while True:
            img = get_image(controller)
            buy_result = context.run_recognition("BuyFishBaitBuy", img)
            if buy_result and buy_result.hit:
                print("Buy button found, clicking...")
                for _ in range(3):
                    click_rect(controller, buy_region, 0.3)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                print("Buy button not found, retrying...")
                time.sleep(1)

        for _ in range(5):
            img = get_image(controller)
            buy_confirm_result = context.run_recognition("BuyFishBaitConfirm", img)
            if buy_confirm_result and buy_confirm_result.hit:
                print("Buy confirm button found, clicking...")
                for _ in range(3):
                    click_rect(controller, buy_confirm_region)
                    time.sleep(0.1)
                time.sleep(0.5)
                break
            else:
                print("Buy confirm button not found, retrying...")
                time.sleep(1)

        while True:
            img = get_image(controller)
            buy_success_result = context.run_recognition("BuyFishBaitSuccess", img)
            if buy_success_result and buy_success_result.hit:
                print("Buy success.")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                print("Buy success confirmation not found, retrying...")
                time.sleep(1)

        return CustomAction.RunResult(success=True)
