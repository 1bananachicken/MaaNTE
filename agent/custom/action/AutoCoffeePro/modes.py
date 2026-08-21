import time
import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT
from .utils import get_image, click_rect_multiple, press_key_f


@AgentServer.custom_action("nanally_pro")
class NanallyProAction(CustomAction):
    """Mode 1 (Nanally / BaiCang): repeatedly click hammer until win template appears.

    Pipeline usage:
    ```jsonc
    {
        "NanallyProRun": {
            "action": "Custom",
            "custom_action": "nanally_pro",
            "custom_action_param": {
                "max_loop_seconds": 300,
                "loop_interval": 0.2
            }
        }
    }
    ```
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        max_loop_seconds = 300.0
        loop_interval = 0.2
        hammer_recognition_node = "NanallyProCheckHammer"

        if argv.custom_action_param:
            params = json.loads(argv.custom_action_param)
            max_loop_seconds = params.get("max_loop_seconds", max_loop_seconds)
            loop_interval = params.get("loop_interval", loop_interval)
            hammer_recognition_node = params.get("hammer_recognition_node", hammer_recognition_node)

        start_time = time.time()
        PrintT(context, "NanallyPro: started hammer clicking mode")

        while True:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            # Check timeout
            if time.time() - start_time > max_loop_seconds:
                logger.warning("NanallyPro timed out after %.1f seconds", max_loop_seconds)
                return CustomAction.RunResult(success=False)

            # Click hammer
            img = get_image(controller)
            hammer_result = context.run_recognition(hammer_recognition_node, img)
            if hammer_result and hammer_result.hit:
                click_rect_multiple(
                    controller,
                    [hammer_result.box.x, hammer_result.box.y, hammer_result.box.w, hammer_result.box.h],
                )
            else:
                logger.debug("NanallyPro: hammer not found, waiting")
                time.sleep(loop_interval)


@AgentServer.custom_action("lacrimosa_pro")
class LacrimosaProAction(CustomAction):
    """Mode 2 (Lacrimosa): detect juice dish above customers and serve them.

    Detection flow:
    1. Use BackgroundDiffPro (CustomRecognition) to find customer positions
    2. For each customer, check if a juice icon appears above them
    3. If found, click glasses -> click tomato to serve

    Pipeline usage:
    ```jsonc
    {
        "LacrimosaProRun": {
            "action": "Custom",
            "custom_action": "lacrimosa_pro",
            "custom_action_param": {
                "max_services": 10,
                "max_loop_seconds": 300,
                "idle_sleep": 0.15
            }
        }
    }
    ```
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        max_services = 10
        max_loop_seconds = 300.0
        idle_sleep = 0.15
        tool_interval = 0.12
        after_service_delay = 0.18
        post_service_cooldown = 0.3
        juice_recognition_node = "LacrimosaProCheckJuice"
        glasses_recognition_node = "LacrimosaProCheckGlasses"
        tomato_recognition_node = "LacrimosaProCheckTomato"

        if argv.custom_action_param:
            params = json.loads(argv.custom_action_param)
            max_services = params.get("max_services", max_services)
            max_loop_seconds = params.get("max_loop_seconds", max_loop_seconds)
            idle_sleep = params.get("idle_sleep", idle_sleep)
            tool_interval = params.get("tool_interval", tool_interval)
            after_service_delay = params.get("after_service_delay", after_service_delay)
            post_service_cooldown = params.get("post_service_cooldown", post_service_cooldown)
            juice_recognition_node = params.get("juice_recognition_node", juice_recognition_node)
            glasses_recognition_node = params.get("glasses_recognition_node", glasses_recognition_node)
            tomato_recognition_node = params.get("tomato_recognition_node", tomato_recognition_node)

        service_count = 0
        start_time = time.time()
        PrintT(context, "LacrimosaPro: started juice detection mode")

        while service_count < max_services:
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            if time.time() - start_time > max_loop_seconds:
                logger.warning("LacrimosaPro timed out after %.1f seconds", max_loop_seconds)
                return CustomAction.RunResult(success=False)

            # Step 1: Detect juice demand (customer head icon)
            img = get_image(controller)
            juice_result = context.run_recognition(juice_recognition_node, img)
            if not (juice_result and juice_result.hit):
                time.sleep(idle_sleep)
                continue

            logger.info("LacrimosaPro: juice demand detected, score=%.3f", getattr(juice_result, "score", 0))

            # Step 2: Click glasses tool
            img = get_image(controller)
            glasses_result = context.run_recognition(glasses_recognition_node, img)
            if not (glasses_result and glasses_result.hit):
                logger.warning("LacrimosaPro: glasses not found, skipping service")
                time.sleep(idle_sleep)
                continue

            click_rect_multiple(
                controller,
                [glasses_result.box.x, glasses_result.box.y, glasses_result.box.w, glasses_result.box.h],
            )
            time.sleep(tool_interval)

            # Step 3: Click tomato tool
            img = get_image(controller)
            tomato_result = context.run_recognition(tomato_recognition_node, img)
            if not (tomato_result and tomato_result.hit):
                logger.warning("LacrimosaPro: tomato not found, skipping service")
                continue

            click_rect_multiple(
                controller,
                [tomato_result.box.x, tomato_result.box.y, tomato_result.box.w, tomato_result.box.h],
            )
            time.sleep(after_service_delay)

            # Cooldown to avoid re-triggering on the same customer
            time.sleep(post_service_cooldown)

            service_count += 1
            logger.info("LacrimosaPro: completed service %d/%d", service_count, max_services)

        PrintT(context, "LacrimosaPro: all services completed")
        return CustomAction.RunResult(success=True)
