import json
import re
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils.logger import logger
from utils.maafocus import PrintT

from ..Common.utils import click_rect_multiple, get_image
from .utils import press_key_f, wait_and_claim

DEFAULT_MAKE_COUNT = 10
DEFAULT_CHECK_FREQ = 0.5
COUNTDOWN_DETECT_TIMEOUT = 20
COUNTDOWN_POLL_INTERVAL = 0.5
SECOND_GUEST_REMAINING_SECONDS = 111
SECOND_GUEST_FALLBACK_DELAY = 8.5
TOMATO_JUICE_SERVINGS = 2


def _positive_number(value, default, value_type):
    try:
        parsed = value_type(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _load_params(raw_param):
    if not raw_param:
        return DEFAULT_MAKE_COUNT, DEFAULT_CHECK_FREQ

    try:
        params = raw_param if isinstance(raw_param, dict) else json.loads(raw_param)
        if not isinstance(params, dict):
            raise TypeError("custom_action_param must be a JSON object")
    except (json.JSONDecodeError, TypeError) as error:
        logger.warning(
            "[AutoTomatoJuice] Failed to parse custom_action_param: %s; raw value: %r",
            error,
            raw_param,
        )
        return DEFAULT_MAKE_COUNT, DEFAULT_CHECK_FREQ

    make_count = _positive_number(params.get("count"), DEFAULT_MAKE_COUNT, int)
    check_freq = _positive_number(params.get("freq"), DEFAULT_CHECK_FREQ, float)
    return make_count, check_freq


def _result_rect(result):
    return [result.box.x, result.box.y, result.box.w, result.box.h]


def _make_tomato_juice(context):
    """依次选择空玻璃杯和番茄汁，制作一杯特调番茄汁。"""
    if context.tasker.stopping:
        return False
    context.run_action("MakeTomatoJuiceSelectGlass")

    if context.tasker.stopping:
        return False
    context.run_action("MakeTomatoJuiceAddTomato")
    return True


def _remaining_seconds(result):
    """从营业倒计时的 OCR 结果中解析剩余秒数。"""
    best_result = result.best_result if result and result.hit else None
    text = getattr(best_result, "text", "")
    if not text:
        return None

    minute_second = re.search(
        r"(\d+)\s*(?:分|분|m(?:in)?)\s*(\d+)\s*(?:秒|초|s(?:ec)?)",
        text,
        re.IGNORECASE,
    )
    if minute_second:
        minutes, seconds = map(int, minute_second.groups())
        return minutes * 60 + seconds

    colon_time = re.search(r"(\d+)\s*[:：]\s*(\d+)", text)
    if colon_time:
        minutes, seconds = map(int, colon_time.groups())
        return minutes * 60 + seconds

    return None


def _wait_for_customers_ready(context, controller, check_freq):
    """等待第二位客人到店，确保连续制作的两杯番茄汁都能被接收。"""
    start_time = time.monotonic()
    countdown_seen_at = None
    poll_interval = min(check_freq, COUNTDOWN_POLL_INTERVAL)

    while time.monotonic() - start_time <= COUNTDOWN_DETECT_TIMEOUT:
        if context.tasker.stopping:
            return False

        img = get_image(controller)
        ready_result = context.run_recognition("MakeTomatoJuiceBusinessReady", img)
        now = time.monotonic()
        if ready_result and ready_result.hit:
            remaining = _remaining_seconds(ready_result)
            if countdown_seen_at is None:
                countdown_seen_at = now
                logger.debug(
                    "[AutoTomatoJuice] Countdown detected after %.2fs; "
                    "remaining=%s",
                    now - start_time,
                    remaining,
                )

            if (
                remaining is not None
                and remaining <= SECOND_GUEST_REMAINING_SECONDS
            ):
                logger.debug(
                    "[AutoTomatoJuice] Second guest ready at %ds remaining",
                    remaining,
                )
                return True

        if (
            countdown_seen_at is not None
            and now - countdown_seen_at >= SECOND_GUEST_FALLBACK_DELAY
        ):
            logger.warning(
                "[AutoTomatoJuice] Countdown target was not recognized; "
                "using %.1fs fallback delay",
                SECOND_GUEST_FALLBACK_DELAY,
            )
            return True

        time.sleep(poll_interval)

    logger.warning(
        "[AutoTomatoJuice] Business countdown was not detected within %.1fs",
        COUNTDOWN_DETECT_TIMEOUT,
    )
    return False


@AgentServer.custom_action("auto_make_tomato_juice")
class AutoMakeTomatoJuice(CustomAction):

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        PrintT(context, "coffee.started")
        controller = context.tasker.controller
        make_count, check_freq = _load_params(argv.custom_action_param)
        exit_roi = [11, 12, 38, 37]

        for count in range(make_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)
            PrintT(context, "coffee.making", count + 1, make_count)

            # Step 1: 选择“新品练习 I”并开始营业。
            PrintT(context, "coffee.step_wait_start")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                start_result = context.run_recognition("MakeCoffeeStart", img)
                if not (start_result and start_result.hit):
                    time.sleep(check_freq)
                    continue

                while True:
                    if context.tasker.stopping:
                        return CustomAction.RunResult(success=False)

                    context.run_action("MakeCoffeeScrollToTop")
                    time.sleep(1)
                    img = get_image(controller)
                    target_result = context.run_recognition(
                        "MakeCoffeeTargetCoffeeMaster", img
                    )
                    if target_result and target_result.hit:
                        break

                click_rect_multiple(controller, _result_rect(target_result))
                time.sleep(check_freq)

                img = get_image(controller)
                start_result = context.run_recognition("MakeCoffeeStart", img)
                if not (start_result and start_result.hit):
                    continue

                PrintT(context, "coffee.step_start_click")
                click_rect_multiple(controller, _result_rect(start_result))
                if not _wait_for_customers_ready(context, controller, check_freq):
                    return CustomAction.RunResult(success=False)
                break

            # Step 2: 第二位客人到店后，连续制作两杯特调番茄汁。
            PrintT(context, "coffee.step_making_dishes")
            for _ in range(TOMATO_JUICE_SERVINGS):
                if not _make_tomato_juice(context):
                    return CustomAction.RunResult(success=False)

            # Step 3: 检测营业额星标；未达标时继续制作番茄汁。
            PrintT(context, "coffee.step_wait_star")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                star_result = context.run_recognition("MakeCoffeeStar", img)
                if star_result and star_result.hit:
                    PrintT(context, "coffee.step_star_click")
                    click_rect_multiple(controller, exit_roi)
                    time.sleep(1)
                    break

                if not _make_tomato_juice(context):
                    return CustomAction.RunResult(success=False)

            # Step 4: 领取奖励并返回咖啡店，准备下一轮。
            PrintT(context, "coffee.step_wait_claim")
            if not wait_and_claim(context, controller, check_freq):
                return CustomAction.RunResult(success=False)

            PrintT(context, "coffee.round_finished")
            press_key_f(controller)
            time.sleep(2)
            PrintT(context, "coffee.iteration_done")

        PrintT(context, "coffee.all_done")
        return CustomAction.RunResult(success=True)
