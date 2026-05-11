from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from .player import AutoPianoPlayer, AutoPianoSettings, AutoPianoStopped


def extract_param(argv: CustomAction.RunArg) -> dict:
    value = getattr(argv, "custom_action_param", None)
    if value is None:
        value = getattr(argv, "param", None)
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value) if value.strip() else {}
    if isinstance(value, dict):
        return value
    return dict(value)


@AgentServer.custom_action("auto_play_piano")
class AutoPlayPiano(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        try:
            self._run(context, argv)
        except AutoPianoStopped:
            print("AutoPiano stopped.", flush=True)
            return CustomAction.RunResult(success=False)
        except Exception as exc:
            print(f"AutoPiano failed: {exc}", flush=True)
            traceback.print_exc()
            return CustomAction.RunResult(success=False)

        return CustomAction.RunResult(success=True)

    def _run(self, context: Context, argv: CustomAction.RunArg) -> None:
        param = extract_param(argv)
        settings = AutoPianoSettings(
            song=str(param.get("song", "")).strip(),
            speed=float(param.get("speed", 1.0)),
            transpose=int(param.get("transpose", 0)),
        )
        AutoPianoPlayer(PROJECT_ROOT).play(context, settings)
