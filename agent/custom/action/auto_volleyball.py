import json
import time

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.pipeline import JRecognitionType, JTemplateMatch

from utils.logger import logger
from utils.maafocus import PrintT
from .Common.utils import get_image

_J_KEY = 0x4A
_KEY_PRESS_INTERVAL_SECONDS = 0.6
_RESULT_CHECK_INTERVAL_SECONDS = 5.0
_MAX_GAME_SECONDS = 600.0
_TEAMMATE_SELECTION_TIMEOUT_SECONDS = 30.0
_TEAMMATE_CONFIRM_TIMEOUT_SECONDS = 2.0
_TEAMMATE_FAILURE_LIMIT = 5
_TEMPLATE_THRESHOLD = 0.8
_CHARACTER_TEMPLATE_THRESHOLD = 0.72
_CHARACTER_SEARCH_ROI = (320, 95, 365, 335)
_DEFAULT_LOOP_COUNT = 99
_MAX_LOOP_COUNT = 9999

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
# 头像顺序会变化，因此用角色头像模板定位，不把格位当作角色身份。
_CHARACTERS = {
    1: ("薄荷", "Volleyball/Character/Mint.png"),
    2: ("零", "Volleyball/Character/Zero.png"),
    3: ("真红", "Volleyball/Character/Zhenhong.png"),
    4: ("娜娜莉", "Volleyball/Character/Nanally.png"),
    5: ("残虹", "Volleyball/Character/Canhong.png"),
    6: ("卡厄斯", "Volleyball/Character/Chaos.png"),
    7: ("伊洛伊", "Volleyball/Character/Yiluoyi.png"),
}

# 格位只用于确定主控/队友标签的局部确认区域，角色与格位没有绑定关系。
_CHARACTER_SLOTS = (
    ((407, 166), (323, 94, 103, 76)),
    ((519, 167), (443, 98, 89, 70)),
    ((633, 170), (554, 98, 89, 70)),
    ((407, 276), (332, 208, 89, 69)),
    ((516, 279), (444, 209, 88, 68)),
    ((630, 278), (556, 208, 88, 70)),
    ((410, 386), (332, 318, 89, 70)),
)
_MAX_SLOT_DISTANCE_SQUARED = 45 * 45

_current_difficulty = 1
_target_loop_count = _DEFAULT_LOOP_COUNT
_completed_loop_count = 0


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


def _locate_character(
    context: Context, frame, name: str, template: str
) -> tuple[tuple, tuple] | None:
    """Locate a character portrait, then map it to the nearest visible grid slot."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None

    result = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            template=[template],
            roi=_CHARACTER_SEARCH_ROI,
            threshold=[_CHARACTER_TEMPLATE_THRESHOLD],
        ),
        frame,
    )
    best = result.best_result if result is not None and result.hit else None
    if best is None:
        logger.debug("AutoVolleyball: portrait not found for %s", name)
        return None

    x, y, width, height = tuple(best.box)
    center = (x + width // 2, y + height // 2)
    slot_center, confirmed_roi = min(
        _CHARACTER_SLOTS,
        key=lambda slot: (center[0] - slot[0][0]) ** 2 + (center[1] - slot[0][1]) ** 2,
    )
    distance_squared = (center[0] - slot_center[0]) ** 2 + (
        center[1] - slot_center[1]
    ) ** 2
    if distance_squared > _MAX_SLOT_DISTANCE_SQUARED:
        logger.warning(
            "AutoVolleyball: %s portrait matched outside known slots at %s",
            name,
            center,
        )
        return None

    logger.debug(
        "AutoVolleyball: located %s at %s score=%.3f",
        name,
        center,
        getattr(best, "score", 0.0),
    )
    return confirmed_roi, (center[0], center[1], 1, 1)


def _select_teammate(
    context: Context,
    controller,
    name: str,
    portrait_template: str,
    confirmed_template: str,
    deadline: float,
    failures: list[int],
) -> bool:
    """Check selection before each click and verify it after the click."""
    while time.monotonic() < deadline and not context.tasker.stopping:
        frame = get_image(controller)
        location = _locate_character(context, frame, name, portrait_template)
        if location is None:
            failures[0] += 1
            logger.warning(
                "AutoVolleyball: %s portrait detection failed %d/%d",
                name,
                failures[0],
                _TEAMMATE_FAILURE_LIMIT,
            )
            if failures[0] >= _TEAMMATE_FAILURE_LIMIT:
                return False
            time.sleep(0.2)
            continue

        confirmed_roi, click_roi = location
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
        global _completed_loop_count, _current_difficulty, _target_loop_count

        params = _load_params(argv.custom_action_param)
        try:
            start_difficulty = int(params.get("start_difficulty", 1))
        except (TypeError, ValueError):
            start_difficulty = 1

        _current_difficulty = min(4, max(1, start_difficulty))
        try:
            loop_count = int(params.get("loop_count", _DEFAULT_LOOP_COUNT))
        except (TypeError, ValueError):
            loop_count = _DEFAULT_LOOP_COUNT
        _target_loop_count = min(_MAX_LOOP_COUNT, max(1, loop_count))
        _completed_loop_count = 0
        context.override_next(
            "VolleyballCountCompletedLoop", ["VolleyballRestartButton"]
        )
        PrintT(context, "volleyball.started", _target_loop_count)
        logger.info(
            "AutoVolleyball: start difficulty=%d loop_count=%d",
            _current_difficulty,
            _target_loop_count,
        )
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
        PrintT(context, "volleyball.selecting_difficulty")
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
            name, portrait_template = _CHARACTERS[character_id]
            if not _select_teammate(
                context,
                controller,
                name,
                portrait_template,
                confirmed_template,
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

        PrintT(
            context,
            "volleyball.playing",
            _completed_loop_count + 1,
            _target_loop_count,
        )

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
                    controller.post_click_key(_J_KEY).wait()
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


@AgentServer.custom_action("volleyball_count_completed_loop")
class VolleyballCountCompletedLoop(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _completed_loop_count

        _completed_loop_count += 1
        PrintT(
            context,
            "volleyball.loop_progress",
            _completed_loop_count,
            _target_loop_count,
        )
        logger.info(
            "AutoVolleyball: completed loop %d/%d",
            _completed_loop_count,
            _target_loop_count,
        )

        if _completed_loop_count >= _target_loop_count:
            context.override_next(
                "VolleyballCountCompletedLoop", ["VolleyballTaskComplete"]
            )
            PrintT(context, "volleyball.loop_task_done", _completed_loop_count)
        else:
            context.override_next(
                "VolleyballCountCompletedLoop", ["VolleyballRestartButton"]
            )

        return CustomAction.RunResult(success=True)


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
