import time
import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT
from .utils import get_image, click_rect_multiple, press_key_f


@AgentServer.custom_action("auto_coffee_pro")
class AutoCoffeePro(CustomAction):
    """Orchestrator: full game automation loop.

    Original logic from myshit/game_bot.py GameAutomationBot.run():
    1. Level selection (find target level, click enter, wait for start)
    2. Run game mode (delegated to sub-actions)
    3. Exit level and handle result (retry on fail, finish on success)
    4. Loop until max_rounds reached or task stopped

    Pipeline usage:
    ```jsonc
    {
        "AutoCoffeeProRun": {
            "action": "Custom",
            "custom_action": "auto_coffee_pro",
            "custom_action_param": {
                "mode": "nanally",
                "max_rounds": 20,
                "level_search_timeout": 60,
                "start_button_timeout": 8,
                "exit_button_timeout": 8,
                "result_timeout": 4
            }
        }
    }
    ```
    """

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        # ── Parse parameters from custom_action_param ──
        mode = "nanally"
        max_rounds = 20
        level_search_timeout = 60.0
        start_button_timeout = 8.0
        exit_button_timeout = 8.0
        result_timeout = 4.0
        level_node = "AutoCoffeeProFindLevel"
        start_node = "AutoCoffeeProCheckStart"
        exit_node = "AutoCoffeeProCheckExit"
        success_node = "AutoCoffeeProCheckSuccess"
        fail_node = "AutoCoffeeProCheckFail"
        retry_node = "AutoCoffeeProCheckRetry"
        finish_node = "AutoCoffeeProCheckFinish"

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                mode = params.get("mode", mode)
                max_rounds = params.get("max_rounds", max_rounds)
                level_search_timeout = params.get("level_search_timeout", level_search_timeout)
                start_button_timeout = params.get("start_button_timeout", start_button_timeout)
                exit_button_timeout = params.get("exit_button_timeout", exit_button_timeout)
                result_timeout = params.get("result_timeout", result_timeout)
                level_node = params.get("level_node", level_node)
                start_node = params.get("start_node", start_node)
                exit_node = params.get("exit_node", exit_node)
                success_node = params.get("success_node", success_node)
                fail_node = params.get("fail_node", fail_node)
                retry_node = params.get("retry_node", retry_node)
                finish_node = params.get("finish_node", finish_node)
            except json.JSONDecodeError as e:
                logger.error("AutoCoffeePro: failed to parse params: %s", e)

        PrintT(context, "AutoCoffeePro started (mode=%s, max_rounds=%d)", mode, max_rounds)
        logger.info("AutoCoffeePro: mode=%s, max_rounds=%d", mode, max_rounds)

        # ── Main round loop ──
        for round_index in range(1, max_rounds + 1):
            if context.tasker.stopping:
                logger.info("AutoCoffeePro: task stopping, exiting at round %d", round_index)
                return CustomAction.RunResult(success=True)

            PrintT(context, "AutoCoffeePro: round %d/%d", round_index, max_rounds)

            # ── Step 1: Level Selection ──
            if not self._select_level(
                context, controller,
                level_node, level_search_timeout,
                start_node, start_button_timeout,
            ):
                logger.error("AutoCoffeePro: level selection failed on round %d", round_index)
                return CustomAction.RunResult(success=False)

            # ── Step 2: Run Game Mode ──
            if not self._run_mode(context, controller, mode):
                logger.warning("AutoCoffeePro: mode '%s' failed on round %d", mode, round_index)
                # Continue to exit handling anyway

            # ── Step 3: Exit Level and Handle Result ──
            self._exit_and_handle_result(
                context, controller,
                exit_node, exit_button_timeout,
                success_node, fail_node, retry_node, finish_node, result_timeout,
            )

        PrintT(context, "AutoCoffeePro: all %d rounds completed", max_rounds)
        return CustomAction.RunResult(success=True)

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _select_level(
        self,
        context: Context,
        controller,
        level_node: str,
        level_timeout: float,
        start_node: str,
        start_timeout: float,
    ) -> bool:
        """Find and click target level, then click start button."""
        logger.info("AutoCoffeePro: searching for target level")

        deadline = time.time() + level_timeout
        while time.time() < deadline:
            if context.tasker.stopping:
                return False

            img = get_image(controller)
            level_result = context.run_recognition(level_node, img)
            if level_result and level_result.hit:
                logger.info("AutoCoffeePro: target level found")
                click_rect_multiple(
                    controller,
                    [level_result.box.x, level_result.box.y, level_result.box.w, level_result.box.h],
                )
                time.sleep(0.5)
                break
            time.sleep(0.3)
        else:
            logger.error("AutoCoffeePro: level not found within %.1f seconds", level_timeout)
            return False

        # Wait for start button
        deadline = time.time() + start_timeout
        while time.time() < deadline:
            if context.tasker.stopping:
                return False

            img = get_image(controller)
            start_result = context.run_recognition(start_node, img)
            if start_result and start_result.hit:
                click_rect_multiple(
                    controller,
                    [start_result.box.x, start_result.box.y, start_result.box.w, start_result.box.h],
                )
                logger.info("AutoCoffeePro: start button clicked")
                time.sleep(1.5)
                return True
            time.sleep(0.3)

        logger.error("AutoCoffeePro: start button not found")
        return False

    def _run_mode(
        self, context: Context, controller, mode: str
    ) -> bool:
        """Run the specified game mode via pipeline sub-actions.

        The mode is executed by running a pipeline action node.
        The user must have pipeline nodes defined for each mode.
        """
        mode_node_map = {
            "nanally": "NanallyProRun",
            "lacrimosa": "LacrimosaProRun",
        }

        node_name = mode_node_map.get(mode)
        if not node_name:
            logger.error("AutoCoffeePro: unknown mode '%s'", mode)
            return False

        # Run the mode via pipeline action
        context.run_action(node_name)
        logger.info("AutoCoffeePro: mode '%s' execution completed", mode)
        return True

    def _exit_and_handle_result(
        self,
        context: Context,
        controller,
        exit_node: str,
        exit_timeout: float,
        success_node: str,
        fail_node: str,
        retry_node: str,
        finish_node: str,
        result_timeout: float,
    ) -> None:
        """Exit the current level and handle result screen.

        - Detects fail/combo_break → click retry if available
        - Otherwise → click finish
        """
        # Click exit
        deadline = time.time() + exit_timeout
        while time.time() < deadline:
            if context.tasker.stopping:
                return

            img = get_image(controller)
            exit_result = context.run_recognition(exit_node, img)
            if exit_result and exit_result.hit:
                click_rect_multiple(
                    controller,
                    [exit_result.box.x, exit_result.box.y, exit_result.box.w, exit_result.box.h],
                )
                logger.info("AutoCoffeePro: exit button clicked")
                time.sleep(0.8)
                break
            time.sleep(0.3)
        else:
            logger.warning("AutoCoffeePro: exit button not found, continuing")

        # Detect result state
        deadline = time.time() + result_timeout
        state = None
        while time.time() < deadline:
            if context.tasker.stopping:
                return

            img = get_image(controller)
            fail_result = context.run_recognition(fail_node, img)
            if fail_result and fail_result.hit:
                state = "fail"
                logger.warning("AutoCoffeePro: detected fail result")
                break

            success_result = context.run_recognition(success_node, img)
            if success_result and success_result.hit:
                state = "success"
                logger.info("AutoCoffeePro: detected success result")
                break

            time.sleep(0.25)

        # Handle retry on fail
        if state == "fail":
            deadline = time.time() + 4.0  # retry_button_timeout
            while time.time() < deadline:
                if context.tasker.stopping:
                    return

                img = get_image(controller)
                retry_result = context.run_recognition(retry_node, img)
                if retry_result and retry_result.hit:
                    click_rect_multiple(
                        controller,
                        [retry_result.box.x, retry_result.box.y, retry_result.box.w, retry_result.box.h],
                    )
                    logger.info("AutoCoffeePro: retry clicked after fail")
                    time.sleep(1.5)
                    return
                time.sleep(0.3)

        # Click finish (success or fallback)
        deadline = time.time() + 8.0  # finish_button_timeout
        while time.time() < deadline:
            if context.tasker.stopping:
                return

            img = get_image(controller)
            finish_result = context.run_recognition(finish_node, img)
            if finish_result and finish_result.hit:
                click_rect_multiple(
                    controller,
                    [finish_result.box.x, finish_result.box.y, finish_result.box.w, finish_result.box.h],
                )
                logger.info("AutoCoffeePro: finish clicked")
                time.sleep(1.2)
                return
            time.sleep(0.3)

        logger.warning("AutoCoffeePro: finish button not found")
