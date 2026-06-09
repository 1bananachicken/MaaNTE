import json
import random
import re
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


def _split(text):
    return [t.strip() for t in re.split(r"[;；]", text) if t.strip()]


_bagel_spam_cached_index = -1


@AgentServer.custom_action("bagel_spam_pick_index")
class BagelSpamPickIndex(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _bagel_spam_cached_index

        _bagel_spam_cached_index = random.randint(0, 999999)
        logger.debug("cached index: %d", _bagel_spam_cached_index)

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("bagel_spam_output_text")
class BagelSpamOutputText(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        global _bagel_spam_cached_index

        params = _parse_params(argv)
        titles_str = params.get("titles", "")
        bodies_str = params.get("bodies", "")
        output = params.get("output", "title")

        titles = _split(titles_str)
        bodies = _split(bodies_str)

        if not titles or not bodies:
            logger.error("titles or bodies is empty")
            return CustomAction.RunResult(success=False)

        pairs = list(zip(titles, bodies))
        idx = (
            _bagel_spam_cached_index % len(pairs)
            if _bagel_spam_cached_index >= 0
            else 0
        )
        text = pairs[idx][0] if output == "title" else pairs[idx][1]

        logger.debug("output %s[%d]: %s", output, idx, text)

        controller = context.tasker.controller
        controller.post_press_key(0x24).wait()
        for _ in range(50):
            controller.post_press_key(0x2E).wait()
        controller.post_input_text(text).wait()

        return CustomAction.RunResult(success=True)


def _parse_params(argv):
    if argv.custom_action_param:
        try:
            return json.loads(argv.custom_action_param)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}
