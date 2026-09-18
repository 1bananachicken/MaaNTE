"""
固定价格策略

每轮都出用户配置的固定金额；默认 1，即改造前"每轮出价 1"的行为。
参数（strategy_param）：
  - price: 正整数，缺省 1；非法值回落默认值并写日志
"""

from typing import Optional

from utils.logger import logger

from ..utils import to_positive_int
from . import register

DEFAULT_PRICE = 1


@register("fixed")
def decide(context, controller, params: dict, ui: dict) -> Optional[int]:
    raw = params.get("price", DEFAULT_PRICE)
    price = to_positive_int(raw)
    if price is None:
        logger.warning(f"⚠️ 固定价格参数无效（{raw!r}），回落到默认值 {DEFAULT_PRICE}")
        return DEFAULT_PRICE
    return price
