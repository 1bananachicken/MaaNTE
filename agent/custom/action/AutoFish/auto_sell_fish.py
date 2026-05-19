import time

from ..Common.utils import get_image, click_rect

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("auto_sell_fish")
class AutoSellFish(CustomAction):

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
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
        sell_fail_region = [739, 349, 202, 24]
        no_valid_fish_region = [509, 350, 261, 22]

        while True:
            img = get_image(controller)
            sell_option_result = context.run_recognition("SellFishOptionGray", img)
            if sell_option_result and sell_option_result.hit:
                for _ in range(3):
                    click_rect(controller, sell_option_region)
                    time.sleep(0.1)

                img = get_image(controller)
                sell_option_sel_result = context.run_recognition("SellFishOption", img)
                if sell_option_sel_result and sell_option_sel_result.hit:
                    break
                time.sleep(1)
            else:
                controller.post_click_key(KEY_Q).wait()
                time.sleep(1)

        print("Sell option detected. Proceeding to sell fish.")

        for _ in range(5):
            img = get_image(controller)
            no_fish_result = context.run_recognition("SellFishNoFish", img)
            time.sleep(0.1)
            if no_fish_result and no_fish_result.hit:
                print("No fish to sell detected. Closing fish shop.")
                controller.post_click_key(KEY_ESC).wait()
                return CustomAction.RunResult(success=True)

        time.sleep(1.5)

        while True:
            img = get_image(controller)
            sell_button_result = context.run_recognition("SellFishButton", img)
            if sell_button_result and sell_button_result.hit:
                print("Sell button detected. Clicking to confirm selling fish.")
                while True:
                    click_rect(controller, sell_button_region, 0.1)
                    time.sleep(0.5)
                    img = get_image(controller)
                    confirm_sell_result = context.run_recognition(
                        "SellFishConfirm", img
                    )
                    sell_fail_result = context.run_recognition("SellFishFail", img)
                    if confirm_sell_result and confirm_sell_result.hit:
                        print(
                            "Confirm sell button detected. Clicking to confirm selling fish."
                        )
                        click_rect(controller, confirm_sell_region, 0.2)
                        time.sleep(0.5)
                        break
                    elif sell_fail_result and sell_fail_result.hit:
                        print("no fish to sell, closing fish shop.")
                        controller.post_click_key(KEY_ESC).wait()
                        return CustomAction.RunResult(success=True)
                    else:
                        time.sleep(0.1)
                break
            else:
                time.sleep(0.1)

        while True:
            img = get_image(controller)
            sell_success_result = context.run_recognition("SellFishSuccess", img)
            if sell_success_result and sell_success_result.hit:
                print("Sell success detected. Fish sold successfully.")
                controller.post_click_key(KEY_ESC).wait()
                time.sleep(0.5)
                controller.post_click_key(KEY_ESC).wait()
                break
            else:
                time.sleep(1)

        print("All fishing tasks complete.")
        return CustomAction.RunResult(success=True)
