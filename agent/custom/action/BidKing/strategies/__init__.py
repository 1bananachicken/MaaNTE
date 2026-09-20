"""
出价策略注册表

新增一种策略只要两步：
  1. 在本目录新建一个模块，用 @register("策略名") 装饰一个 decide 函数
  2. 在文件末尾 import 一次（导入即注册），再到前端任务选项里加一个 case

策略约定：decide(context, controller, params, ui) -> Optional[int]
  - params: 该策略自己的参数（前端 strategy_param），不要读别的策略的键
  - ui:     公共输入层解析出来的东西（input_roi / clear_btn / debug）
  - 屏幕坐标（ROI）一律从 ..layout 导入，不走 params / ui
  - 返回正整数 = 本轮出价额；返回 None = 本轮放弃

策略只负责"算出这个数"：不决定点击顺序、不点击数字、不提交确认。
"""

from typing import Any, Callable, Optional

# 接收ABCD四个参数，返回int(或None)对象
Strategy = Callable[[Any, Any, dict, dict], Optional[int]]

# 注册表，键是策略名称，值是对应的策略函数
STRATEGIES: dict[str, Strategy] = {}

# 默认策略：等价于改造前"每轮固定出价 1"的行为
DEFAULT_STRATEGY = "fixed"

# 注册装饰器
# 外层register(name)接收策略名，中层decorator(fn)接受策略函数，内层STRAGIERS[name] = fn登记进字典，返回原函数
# @register("fixed")修饰一个函数后，就被登记到STRATEGIES["fixed"]
def register(name: str) -> Callable[[Strategy], Strategy]:
    """把一个策略实现注册到指定名字下"""

    def decorator(fn: Strategy) -> Strategy:
        STRATEGIES[name] = fn
        return fn

    return decorator


def get_strategy(name: str) -> tuple[Strategy, bool]:
    """返回 (策略实现, 是否发生了回落)；名字未知时回落到默认策略"""
    fn = STRATEGIES.get(name)
    if fn is not None:
        return fn, False
    return STRATEGIES[DEFAULT_STRATEGY], True


def available() -> list[str]:
    """已注册的策略名（用于日志与排错）"""
    return sorted(STRATEGIES)


# 导入即注册，必须放在 register 定义之后
from . import fixed_price, valuation  # noqa: E402,F401
