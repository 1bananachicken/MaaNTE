import json
import random
import re
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger

# LLM 生成模式下，BagelSpamLLMGenerate 会把结果存在这两个变量里
# 我们优先使用 LLM 生成的内容，没有才走预设随机
from . import bagel_spam_llm as _llm_mod


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
        output = params.get("output", "title")

        if _llm_mod._bagel_spam_llm_title and _llm_mod._bagel_spam_llm_body:
            titles_str = _llm_mod._bagel_spam_llm_title
            bodies_str = _llm_mod._bagel_spam_llm_body
        else:
            titles_str = params.get("titles", "")
            bodies_str = params.get("bodies", "")

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
        controller.post_input_text(text).wait()

        return CustomAction.RunResult(success=True)


def _parse_params(argv):
    if argv.custom_action_param:
        try:
            return json.loads(argv.custom_action_param)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}
