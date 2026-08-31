import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JTemplateMatch

from utils.logger import logger
from utils.maafocus import PrintT
from .Common.utils import get_image

_K_KEY = 0x4B
_KEY_PRESS_INTERVAL_SECONDS = 0.6
_RESULT_CHECK_INTERVAL_SECONDS = 5.0
_MAX_GAME_SECONDS = 600.0
_TEAMMATE_SELECTION_TIMEOUT_SECONDS = 30.0
_TEAMMATE_CONFIRM_TIMEOUT_SECONDS = 2.0
_TEAMMATE_FAILURE_LIMIT = 5
_TEMPLATE_THRESHOLD = 0.8

_DIFFICULTY_ROIS = {
    1: (158, 254, 91, 86),
    2: (458, 337, 74, 77),
    3: (766, 264, 76, 75),
    4: (1067, 337, 69, 70),
}

_GAME_END_STATES = (
    ("skip", "Volleyball/SkipButton.png", (1223, 29, 28, 26)),
    ("win", "Volleyball/Win.png", (937, 71, 308, 110)),
    ("loss", "Volleyball/Lose.png", (879, 72, 363, 105)),
)

_FIRST_TEAMMATE_CONFIRMED_TEMPLATE = "Volleyball/FirstTeammateConfirmed.png"
_SECOND_TEAMMATE_CONFIRMED_TEMPLATE = "Volleyball/SecondTeammateConfirmed.png"

# 1-7 与 task 选项中 first/second 的数值一致。
# 每个角色：(显示名, 已选中确认 ROI, 头像点击 ROI)
_CHARACTERS = {
    1: ("薄荷", (323, 94, 103, 76), (407, 166, 1, 1)),
    2: ("零", (443, 98, 89, 70), (519, 167, 1, 1)),
    3: ("娜娜莉", (554, 98, 89, 70), (633, 170, 1, 1)),
    4: ("残虹", (332, 208, 89, 69), (407, 276, 1, 1)),
    5: ("卡厄斯", (444, 209, 88, 68), (516, 279, 1, 1)),
    6: ("真红", (556, 208, 88, 70), (630, 278, 1, 1)),
    7: ("伊洛伊", (332, 318, 89, 70), (410, 386, 1, 1)),
}

_current_difficulty = 1


def _load_params(custom_action_param) -> dict:
    if isinstance(custom_action_param, dict):
        return custom_action_param
    if not custom_action_param:
        return {}
    try:
        params = json.loads(custom_action_param)
    except (TypeError, json.JSONDecodeError):
        return {}
    return params if isinstance(params, dict) else {}


def _match_state(context: Context, frame) -> str | None:
    if frame is None or getattr(frame, "size", 0) == 0:
        return None

    for state, template, roi in _GAME_END_STATES:
        result = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                template=[template],
                roi=tuple(roi),
                threshold=[_TEMPLATE_THRESHOLD],
            ),
            frame,
        )
        if result is not None and result.hit:
            return state
    return None


def _template_hit(context: Context, frame, template: str, roi: tuple) -> bool:
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    result = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            template=[template],
            roi=tuple(roi),
            threshold=[_TEMPLATE_THRESHOLD],
        ),
        frame,
    )
    return result is not None and result.hit


def _select_teammate(
    context: Context,
    controller,
    name: str,
    confirmed_template: str,
    confirmed_roi: tuple,
    click_roi: tuple,
    deadline: float,
    failures: list[int],
) -> bool:
    """Check selection before each click and verify it after the click."""
    while time.monotonic() < deadline and not context.tasker.stopping:
        frame = get_image(controller)
        if _template_hit(context, frame, confirmed_template, confirmed_roi):
            logger.info("AutoVolleyball: %s already selected", name)
            return True

        if failures[0] >= _TEAMMATE_FAILURE_LIMIT:
            return False

        x, y, width, height = click_roi
        try:
            controller.post_click(x + width // 2, y + height // 2).wait()
        except Exception:
            logger.exception("AutoVolleyball: %s click failed", name)

        confirm_deadline = min(
            deadline, time.monotonic() + _TEAMMATE_CONFIRM_TIMEOUT_SECONDS
        )
        while time.monotonic() < confirm_deadline and not context.tasker.stopping:
            frame = get_image(controller)
            if _template_hit(context, frame, confirmed_template, confirmed_roi):
                logger.info(
                    "AutoVolleyball: %s selected after attempt %d",
                    name,
                    failures[0] + 1,
                )
                return True
            time.sleep(0.1)

        failures[0] += 1
        logger.warning(
            "AutoVolleyball: %s selection attempt %d/%d not confirmed",
            name,
            failures[0],
            _TEAMMATE_FAILURE_LIMIT,
        )

    return False


def _resolve_character_id(params: dict, key: str, default: int) -> int:
    try:
        character_id = int(params.get(key, default))
    except (TypeError, ValueError):
        character_id = default
    return character_id if character_id in _CHARACTERS else default


@AgentServer.custom_action("volleyball_reset")
class VolleyballReset(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _current_difficulty

        params = _load_params(argv.custom_action_param)
        try:
            start_difficulty = int(params.get("start_difficulty", 1))
        except (TypeError, ValueError):
            start_difficulty = 1

        _current_difficulty = min(4, max(1, start_difficulty))
        PrintT(context, "volleyball.started", _current_difficulty)
        logger.info("AutoVolleyball: start difficulty=%d", _current_difficulty)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_select_difficulty")
class VolleyballSelectDifficulty(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        roi = _DIFFICULTY_ROIS.get(_current_difficulty)
        if roi is None:
            logger.error(
                "AutoVolleyball: invalid current difficulty=%r", _current_difficulty
            )
            return CustomAction.RunResult(success=False)

        x, y, width, height = roi
        context.tasker.controller.post_click(x + width // 2, y + height // 2).wait()
        PrintT(context, "volleyball.selecting_difficulty", _current_difficulty)
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_select_teammates")
class VolleyballSelectTeammates(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = _load_params(argv.custom_action_param)
        try:
            node_data = context.get_node_data("VolleyballSelectTeammates") or {}
        except Exception:
            logger.exception("AutoVolleyball: failed to read selection config")
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
                "AutoVolleyball: first and second must be different characters"
            )
            return CustomAction.RunResult(success=False)

        controller = context.tasker.controller
        deadline = time.monotonic() + _TEAMMATE_SELECTION_TIMEOUT_SECONDS
        failures = [0]

        selections = (
            (first_id, _FIRST_TEAMMATE_CONFIRMED_TEMPLATE),
            (second_id, _SECOND_TEAMMATE_CONFIRMED_TEMPLATE),
        )
        for character_id, confirmed_template in selections:
            name, confirmed_roi, click_roi = _CHARACTERS[character_id]
            if not _select_teammate(
                context,
                controller,
                name,
                confirmed_template,
                confirmed_roi,
                click_roi,
                deadline,
                failures,
            ):
                logger.error(
                    "AutoVolleyball: teammate selection stopped after %d/%d failed clicks",
                    failures[0],
                    _TEAMMATE_FAILURE_LIMIT,
                )
                return CustomAction.RunResult(success=False)

        PrintT(context, "volleyball.teammates_selected")
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("volleyball_play")
class VolleyballPlay(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller
        tasker = context.tasker
        started_at = time.monotonic()
        next_key_at = time.monotonic()
        next_check_at = next_key_at + _RESULT_CHECK_INTERVAL_SECONDS

        PrintT(context, "volleyball.playing", _current_difficulty)

        try:
            while not tasker.stopping:
                now = time.monotonic()

                if now - started_at >= _MAX_GAME_SECONDS:
                    logger.error(
                        "AutoVolleyball: game exceeded %.0f seconds",
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
                            "AutoVolleyball: detected state=%s at difficulty=%d",
                            state,
                            _current_difficulty,
                        )
                        return CustomAction.RunResult(success=True)
                    next_check_at = now + _RESULT_CHECK_INTERVAL_SECONDS

                sleep_until = min(next_key_at, next_check_at)
                time.sleep(max(0.01, min(0.05, sleep_until - time.monotonic())))
        except Exception:
            logger.exception("AutoVolleyball: game loop failed")
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=False)


@AgentServer.custom_action("volleyball_advance_difficulty")
class VolleyballAdvanceDifficulty(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _current_difficulty

        completed_difficulty = _current_difficulty
        _current_difficulty += 1
        PrintT(context, "volleyball.difficulty_done", completed_difficulty)

        if _current_difficulty > 4:
            context.override_next(
                "VolleyballAdvanceDifficulty", ["VolleyballTaskComplete"]
            )
            PrintT(context, "volleyball.task_done")
        else:
            context.override_next(
                "VolleyballAdvanceDifficulty", ["VolleyballWaitReenterStartRacing"]
            )

        return CustomAction.RunResult(success=True)
