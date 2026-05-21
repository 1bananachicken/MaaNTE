"""通过 MaaFramework pipeline focus 协议向 MXU 客户端发送用户可见消息。"""

from maa.context import Context
from utils.logger import logger
from utils.i18n import T

_FOCUS_NODE = "_MAANTE_FOCUS_"


def Print(context: Context, content: str):
    """向 MXU 发送 focus 消息（原样文本，不做 i18n）。context 为 MaaFramework Context 对象。"""
    if context is None:
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
        context.run_action(_FOCUS_NODE, pipeline_override=pipeline_override)
    except Exception as e:
        logger.warning(f"failed to send focus: {e}")


def PrintT(context: Context, key: str, *args):
    """向 MXU 发送 focus 消息（i18n 版本）。key 为翻译键，*args 为 %-格式化参数。"""
    Print(context, T(key, *args))
