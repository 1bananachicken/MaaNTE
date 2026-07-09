# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import sys
from .test_UF import UF_Logger, UF_Count, UF_CountClean, UF_GetImage
from .test_AUF import UF_ActionLogger,UF_ActionMoveScreen
from .logger import change_console_level
from .test_DF import DF_Action

from .test_AF import AF_ChooseNormalInstance, AF_AutoFightCls,AF_AutoFightClsTest

from .CharacterController import MouseMoveAction

# 显式列出需要处理的函数
FUNCTIONS_RECOGNITION = [
    # 来自 test_UF 的函数
    UF_Logger,
    UF_Count,
    UF_CountClean,
    UF_GetImage,

    # 来自 test_AF 的函数
    AF_ChooseNormalInstance,
    AF_AutoFightCls,
    AF_AutoFightClsTest,
    # 来自 test_DF 的函数
    DF_Action,



]

FUNCTIONS_ACTION = [
    # 来自 test_AUF 的函数
    UF_ActionLogger,
UF_ActionMoveScreen,

    MouseMoveAction,
]

flag = getattr(sys.modules["__main__"], "Agent_FLAG", True)
if flag:
    from maa.agent.agent_server import AgentServer

    for func in FUNCTIONS_RECOGNITION:
        decorated_func = AgentServer.custom_recognition(func.__name__)(func)
        globals()[func.__name__] = decorated_func

    for func in FUNCTIONS_ACTION:
        decorated_func = AgentServer.custom_action(func.__name__)(func)
        globals()[func.__name__] = decorated_func
    __all__ = [func.__name__ for func in FUNCTIONS_RECOGNITION]  # 添加实例到导出列表
else:
    from maa.resource import Resource

    resource = Resource()
    for func in FUNCTIONS_RECOGNITION:
        decorated_func = resource.custom_recognition(func.__name__)(func)
        globals()[func.__name__] = decorated_func

    for func in FUNCTIONS_ACTION:
        decorated_func = resource.custom_action(func.__name__)(func)
        globals()[func.__name__] = decorated_func
    __all__ = [func.__name__ for func in FUNCTIONS_RECOGNITION] + ["resource"]  # 添加实例到导出列表
__all__.append("change_console_level")
# 定义导出列表：包含所有函数和recognition_handler实例
