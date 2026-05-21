import time
import json
import random

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.maafocus import PrintT


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img


def click_rect(controller, rect):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    for _ in range(3):
        controller.post_touch_down(cx, cy).wait()
        time.sleep(0.001)
        controller.post_touch_up().wait()


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
            except:
                pass

        KEY_F = 70

        # Coordinates mapped from auto_make_coffee.json [x, y, w, h]
        select_level_target = [18, 230, 188, 66]
        click_roi = [28, 272, 65, 56]
        start_roi = [1057, 648, 178, 44]
        star_roi = [1204, 109, 29, 27]
        exit_roi = [11, 12, 38, 37]
        claim_roi = [681, 539, 187, 38]

        for count in range(make_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            PrintT(context, "coffee.making", count + 1, make_count)

            # Step 1: 选择关卡
            PrintT(context, "coffee.step_select_level")
            click_rect(controller, select_level_target)
            time.sleep(1)

            # Step 2: 开始营业
            PrintT(context, "coffee.step_wait_start")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                img = get_image(controller)
                start_result = context.run_recognition("MakeCoffeeStart", img)
                if start_result and start_result.hit:
                    PrintT(context, "coffee.step_start_click")
                    click_rect(
                        controller,
                        [
                            start_result.box.x,
                            start_result.box.y,
                            start_result.box.w,
                            start_result.box.h,
                        ],
                    )
                    time.sleep(3)  # Post delay from JSON: 3000ms
                    break
                time.sleep(check_freq)

            # Step 3: 达成营业额
            PrintT(context, "coffee.step_wait_star")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                click_rect(controller, click_roi)
                img = get_image(controller)
                star_result = context.run_recognition("MakeCoffeeStar", img)
                if star_result and star_result.hit:
                    PrintT(context, "coffee.step_star_click")
                    click_rect(controller, exit_roi)
                    time.sleep(1)
                    break
                time.sleep(2)

            # Step 4: 点击领取
            PrintT(context, "coffee.step_wait_claim")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)
                img = get_image(controller)
                claim_result = context.run_recognition("MakeCoffeeClaim", img)
                if claim_result and claim_result.hit:
                    PrintT(context, "coffee.step_claim_click")
                    click_rect(
                        controller,
                        [
                            claim_result.box.x,
                            claim_result.box.y,
                            claim_result.box.w,
                            claim_result.box.h,
                        ],
                    )
                    time.sleep(1)
                    break
                time.sleep(check_freq)

            PrintT(context, "coffee.round_finished")
            controller.post_key_down(KEY_F)
            time.sleep(0.1)
            controller.post_key_up(KEY_F)

            time.sleep(2)
            PrintT(context, "coffee.iteration_done")

        PrintT(context, "coffee.all_done")
        return CustomAction.RunResult(success=True)
