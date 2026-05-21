import json
import threading
import time
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from custom.action.Common.logger import get_logger
from custom.action.SoundTrigger.DodgeCounterTrigger import Dodger
from custom.action.SoundTrigger.SoundListener import Ear
from utils.maafocus import PrintT

logger = get_logger(__name__)


class Ctx:
    def __init__(self):
        self._stop_event = threading.Event()
        self.ear = None
        self.dodger = None
        self.active = False

    def _stopped(self):
        return self._stop_event.is_set()

    def setup(self, controller, threshold=0.13, counter_threshold=0.12):
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
        self.ear.on_counter = self._on_counter
        self.active = True
        logger.debug("Ctx initialized")

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

    def _on_dodge(self):
        if self._stopped():
            return
        if self.dodger:
            threading.Thread(target=self.dodger.dodge, daemon=True).start()

    def _on_counter(self):
        if self._stopped():
            return
        if self.dodger:
            threading.Thread(target=self.dodger.counter, daemon=True).start()


@AgentServer.custom_action("SoundDodgeAction")
class SoundDodgeAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        PrintT(context, "sound_dodge.started")

        threshold = 0.13
        counter_threshold = 0.12
        if argv.custom_action_param:
            try:
                p = json.loads(argv.custom_action_param)
                threshold = float(p.get("threshold", 0.13))
                counter_threshold = float(p.get("counter_attack_threshold", 0.12))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid custom_action_param: {argv.custom_action_param!r}, error: {e}. Using defaults."
                )

        context = Ctx()
        try:
            context.setup(
                context.tasker.controller,
                threshold=threshold,
                counter_threshold=counter_threshold,
            )
            if not context.enter():
                return CustomAction.RunResult(success=False)

            PrintT(context, "sound_dodge.monitoring")
            while not context.tasker.stopping:
                time.sleep(0.1)

            PrintT(context, "sound_dodge.interrupted")
            return CustomAction.RunResult(success=True)
        except Exception as e:
            logger.error("Error: %s", e)
            return CustomAction.RunResult(success=False)
        finally:
            context.exit()
            PrintT(context, "sound_dodge.done")
