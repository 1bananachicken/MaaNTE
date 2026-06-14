import time
import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.maafocus import PrintT
from ..Common.utils import get_image, click_rect_multiple
from .utils import wait_and_claim, press_key_f


@AgentServer.custom_action("auto_make_coffee")
class AutoMakeCoffee(CustomAction):

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        PrintT(context, "coffee.started")
        controller = context.tasker.controller
        make_count = 10
        check_freq = 0.5
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                make_count = params.get("count", 10)
                check_freq = params.get("freq", 0.5)
            except json.JSONDecodeError as e:  
                PrintT(context,  
                    f"[AutoCoffee] Failed to parse custom_action_param as JSON: {e}. "  
                    f"Raw value: {argv.custom_action_param!r}"  
                )  

        click_roi = [28, 272, 65, 56]
        exit_roi = [11, 12, 38, 37]

        for count in range(make_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            PrintT(context, "coffee.making", count + 1, make_count)

            # Step 1: 选择关卡
            PrintT(context, "coffee.step_wait_start")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                img = get_image(controller)
                start_result = context.run_recognition("MakeCoffeeStart", img)
                if start_result and start_result.hit:
                    while True:
                        if context.tasker.stopping:
                            return CustomAction.RunResult(success=False)
                        context.run_action("MakeCoffeeScrollToTop")
                        time.sleep(1.2)
                        img = get_image(controller)
                        target_result = context.run_recognition(
                            "MakeCoffeeTargetCoffeeMaster", img
                        )
                        if target_result and target_result.hit:
                            break

                    click_rect_multiple(
                        controller,
                        [
                            target_result.box.x,
                            target_result.box.y,
                            target_result.box.w,
                            target_result.box.h,
                        ],
                    )
                    img = get_image(controller)
                    start_result = context.run_recognition("MakeCoffeeStart", img)
                    if not (start_result and start_result.hit):
                        time.sleep(check_freq)
                        continue

                    PrintT(context, "coffee.step_start_click")
                    click_rect_multiple(
                        controller,
                        [
                            start_result.box.x,
                            start_result.box.y,
                            start_result.box.w,
                            start_result.box.h,
                        ],
                    )
                    time.sleep(3)
                    break
                time.sleep(check_freq)

            # Step 2: 达成营业额
            PrintT(context, "coffee.step_wait_star")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                click_rect_multiple(controller, click_roi)
                img = get_image(controller)
                star_result = context.run_recognition("MakeCoffeeStar", img)
                if star_result and star_result.hit:
                    PrintT(context, "coffee.step_star_click")
                    click_rect_multiple(controller, exit_roi)
                    time.sleep(1)
                    break
                time.sleep(2)

            # Step 3: 点击领取
            PrintT(context, "coffee.step_wait_claim")
            wait_and_claim(context, controller, check_freq)

            PrintT(context, "coffee.round_finished")
            press_key_f(controller)

            time.sleep(2)
            PrintT(context, "coffee.iteration_done")

        PrintT(context, "coffee.all_done")
        return CustomAction.RunResult(success=True)
