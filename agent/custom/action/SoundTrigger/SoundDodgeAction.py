import json
import queue
import threading
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.action.Common.logger import get_logger
from custom.action.SoundTrigger.DodgeCounterTrigger import Dodger
from custom.action.SoundTrigger.SoundListener import Ear
from utils.maafocus import PrintT

logger = get_logger(__name__)


def _parse_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "enable",
            "enabled",
        }
    return bool(default)


def _parse_params(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise TypeError("custom_action_param must be a JSON object")
    return parsed


def _get_config_value(context, node_name, key, default):
    try:
        node_data = context.get_node_data(node_name) or {}
    except Exception as exc:
        logger.warning("Failed to read %s: %s", node_name, exc)
        return default
    attach = node_data.get("attach") if isinstance(node_data, dict) else None
    if not isinstance(attach, dict):
        return default
    return attach.get(key, default)


class Ctx:
    def __init__(self):
        self._stop_event = threading.Event()
        self._action_queue = queue.Queue(maxsize=1)
        self.ear = None
        self.dodger = None
        self.active = False

    def _stopped(self):
        return self._stop_event.is_set()

    def setup(
        self,
        controller,
        threshold=0.13,
        counter_threshold=0.12,
        dodge_all_attacks=True,
    ):
        if self.active:
            return

        base = Path(__file__).parents[4] / "assets" / "resource" / "base"
        if not base.exists():
            base = Path(__file__).parents[4] / "resource" / "base"

        sample = str(base / "sounds" / "dodge.wav")
        counter = str(base / "sounds" / "counter.wav")

        self.ear = Ear(
            sample_path=sample,
            counter_path=counter,
            threshold=threshold,
            counter_threshold=counter_threshold,
            stop_check=self._stopped,
        )
        self.dodger = Dodger(controller=controller, stop_check=self._stopped)
        self.ear.on_dodge = self._on_dodge
        self.ear.on_counter = (
            self._on_dodge if dodge_all_attacks else self._on_counter
        )
        self.active = True
        logger.debug(
            "Ctx initialized, dodge_all_attacks=%s", dodge_all_attacks
        )

    def enter(self):
        self._stop_event.clear()
        if not self.active or not self.ear:
            return False
        self.ear.start()
        logger.debug("Ctx entered")
        return True

    def exit(self):
        self._stop_event.set()
        if self.ear:
            self.ear.stop()
            self.ear = None
        self.dodger = None
        self.active = False
        logger.debug("Ctx exited")

    def _enqueue_action(self, action):
        if self._stopped():
            return
        try:
            self._action_queue.put_nowait(action)
        except queue.Full:
            logger.debug("Dropping stale sound action: %s", action)

    def _on_dodge(self):
        self._enqueue_action("dodge")

    def _on_counter(self):
        self._enqueue_action("counter")

    def process_next(self, timeout=0.05):
        if self._stopped() or not self.dodger:
            return False
        try:
            action = self._action_queue.get(timeout=timeout)
        except queue.Empty:
            return False
        if action == "dodge":
            self.dodger.dodge()
        else:
            self.dodger.counter()
        return True


@AgentServer.custom_action("SoundDodgeAction")
class SoundDodgeAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        enable_sound_trigger = True
        dodge_all_attacks = True
        threshold = 0.13
        counter_threshold = 0.12
        if argv.custom_action_param:
            try:
                p = _parse_params(argv.custom_action_param)
                parsed_enable = _parse_bool(p.get("enable_sound_trigger"), True)
                parsed_dodge_all = _parse_bool(p.get("dodge_all_attacks"), True)
                parsed_threshold = float(p.get("threshold", 0.13))
                parsed_counter_threshold = float(
                    p.get("counter_attack_threshold", 0.12)
                )
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid custom_action_param: {argv.custom_action_param!r}, error: {e}. Using defaults."
                )
            else:
                enable_sound_trigger = parsed_enable
                dodge_all_attacks = parsed_dodge_all
                threshold = parsed_threshold
                counter_threshold = parsed_counter_threshold

        enable_sound_trigger = _parse_bool(
            _get_config_value(
                context,
                "SoundDodgeEnableConfig",
                "enable_sound_trigger",
                enable_sound_trigger,
            ),
            enable_sound_trigger,
        )
        dodge_all_attacks = _parse_bool(
            _get_config_value(
                context,
                "SoundDodgeModeConfig",
                "dodge_all_attacks",
                dodge_all_attacks,
            ),
            dodge_all_attacks,
        )
        try:
            threshold = float(
                _get_config_value(
                    context,
                    "SoundDodgeThresholdConfig",
                    "threshold",
                    threshold,
                )
            )
            counter_threshold = float(
                _get_config_value(
                    context,
                    "SoundCounterThresholdConfig",
                    "counter_attack_threshold",
                    counter_threshold,
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid sound dodge config: %s", exc)

        if not enable_sound_trigger:
            logger.info("Sound dodge is disabled")
            return CustomAction.RunResult(success=True)

        PrintT(context, "sound_dodge.started")
        logger.info(
            "Sound dodge mode: %s",
            "dodge-only" if dodge_all_attacks else "dodge+counter",
        )

        ctx = Ctx()
        try:
            ctx.setup(
                context.tasker.controller,
                threshold=threshold,
                counter_threshold=counter_threshold,
                dodge_all_attacks=dodge_all_attacks,
            )
            if not ctx.enter():
                return CustomAction.RunResult(success=False)

            PrintT(context, "sound_dodge.monitoring")
            while not context.tasker.stopping:
                ctx.process_next(timeout=0.05)

            PrintT(context, "sound_dodge.interrupted")
            return CustomAction.RunResult(success=True)
        except Exception as e:
            logger.error("Error: %s", e)
            return CustomAction.RunResult(success=False)
        finally:
            ctx.exit()
            PrintT(context, "sound_dodge.done")
