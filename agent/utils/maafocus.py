"""通过 MaaFramework pipeline focus 协议向 MXU 客户端发送用户可见消息。"""

from maa.context import Context
from utils.logger import logger

_FOCUS_NODE = "_MAANTE_FOCUS_"


def Print(ctx: Context, content: str):
    """向 MXU 发送 focus 消息。ctx 为 MaaFramework Context 对象。"""
    if ctx is None:
        logger.warning("context is None, skip sending focus")
        return

    pipeline_override = {
        _FOCUS_NODE: {
            "focus": {"Node.Action.Starting": content},
            "action": "DoNothing",
            "pre_delay": 0,
            "post_delay": 0,
        }
    }

    try:
        ctx.run_action(_FOCUS_NODE, pipeline_override=pipeline_override)
    except Exception as e:
        logger.warning(f"failed to send focus: {e}")
