import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils.logger import logger
from utils.maafocus import PrintT
from .Common.utils import load_params
from .auto_volleyball import (
    _K_KEY,
    _KEY_PRESS_INTERVAL_SECONDS,
    _RESULT_CHECK_INTERVAL_SECONDS,
    _MAX_GAME_SECONDS,
    _match_state,
)

_WEEKLY_SELECT_NODE = "VolleyballWeeklySelectTeammates"

# 选人面板第一行可解锁角色的头像点击坐标（1280x720，从左到右）。
_WEEKLY_CHARACTERS = {
    1: (782, 162),
    2: (892, 162),
    3: (1003, 162),
}

_PICK_SETTLE_SECONDS = 0.5   # 面板打开后的等待
_MOVE_SETTLE_SECONDS = 0.2   # 指针移动到位后的稳定等待
_PRESS_SECONDS = 0.12        # 按下与抬起之间的间隔
_PICK_INTERVAL_SECONDS = 0.6  # 两次选择之间的间隔
_DISMISS_SETTLE_SECONDS = 0.8  # 收起面板后的等待

# 面板外空白处（左下角），选完角色后点它收起选人面板
_PANEL_DISMISS_POINT = (200, 650)


def _click_at(controller, x: int, y: int) -> None:
    """移动、按下、抬起三步分开等待。

    按下前必须保证指针已停稳，否则游戏会把按下后的移动识别成拖动，
    导致点不上角色。
    """
    controller.post_touch_move(x, y).wait()
    time.sleep(_MOVE_SETTLE_SECONDS)
    controller.post_touch_down(x, y).wait()
    time.sleep(_PRESS_SECONDS)
    controller.post_touch_up().wait()


def _resolve_character_id(params: dict, key: str, default: int) -> int:
    try:
        character_id = int(params.get(key, default))
    except (TypeError, ValueError):
        return default
    return character_id if character_id in _WEEKLY_CHARACTERS else default


@AgentServer.custom_action("volleyball_weekly_reset")
class VolleyballWeeklyReset(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("AutoVolleyballWeekly: started")
        PrintT(context, "volleyball_weekly.started")
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_weekly_select_teammates")
class VolleyballWeeklySelectTeammates(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = load_params(argv.custom_action_param)
        try:
            node_data = context.get_node_data(_WEEKLY_SELECT_NODE) or {}
        except Exception:
            logger.exception("AutoVolleyballWeekly: failed to read selection config")
            node_data = {}
        attach = node_data.get("attach") or {}
        first_id = _resolve_character_id(
            attach if "first" in attach else params, "first", 1
        )
        second_id = _resolve_character_id(
            attach if "second" in attach else params, "second", 2
        )

        if first_id == second_id:
            logger.error(
                "AutoVolleyballWeekly: first and second must be different characters"
            )
            return CustomAction.RunResult(success=False)

        controller = context.tasker.controller
        time.sleep(_PICK_SETTLE_SECONDS)
        for character_id in (first_id, second_id):
            x, y = _WEEKLY_CHARACTERS[character_id]
            _click_at(controller, x, y)
            time.sleep(_PICK_INTERVAL_SECONDS)

        # 面板不会自动收起，点面板外空白处关闭，后续节点才能点到开始比赛
        x, y = _PANEL_DISMISS_POINT
        _click_at(controller, x, y)
        time.sleep(_DISMISS_SETTLE_SECONDS)

        PrintT(context, "volleyball_weekly.teammates_selected")
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_weekly_play")
class VolleyballWeeklyPlay(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        tasker = context.tasker
        started_at = time.monotonic()
        next_key_at = time.monotonic()
        next_check_at = next_key_at + _RESULT_CHECK_INTERVAL_SECONDS

        PrintT(context, "volleyball_weekly.playing")

        try:
            while not tasker.stopping:
                now = time.monotonic()

                if now - started_at >= _MAX_GAME_SECONDS:
                    logger.error(
                        "AutoVolleyballWeekly: game exceeded %.0f seconds",
                        _MAX_GAME_SECONDS,
                    )
                    return CustomAction.RunResult(success=False)

                if now >= next_key_at:
                    controller.post_click_key(_K_KEY).wait()
                    next_key_at = now + _KEY_PRESS_INTERVAL_SECONDS

                if now >= next_check_at:
                    controller.post_screencap().wait()
                    state = _match_state(context, controller.cached_image)
                    if state is not None:
                        logger.info(
                            "AutoVolleyballWeekly: detected state=%s", state
                        )
                        return CustomAction.RunResult(success=True)
                    next_check_at = now + _RESULT_CHECK_INTERVAL_SECONDS

                sleep_until = min(next_key_at, next_check_at)
                time.sleep(max(0.01, min(0.05, sleep_until - time.monotonic())))
        except Exception:
            logger.exception("AutoVolleyballWeekly: game loop failed")
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=False)
