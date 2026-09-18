"""
拍卖之王 - 出价入口（策略调度 + 公共输入层）

职责边界：
  - 本动作：按策略名取到策略 → 取得本轮出价额 → 清空输入框 → 逐位点数字 → 回读校验
  - Pipeline：点"确认出价"、处理 >100万 二次确认弹窗、面板残留自愈
    （见 BidKing.json 的 BidKingConfirmBid 与 BidKingStatus.json）
  - 策略：只负责算出"本轮出多少"，不决定点击顺序、不提交确认（见 strategies/）

Pipeline 参数（custom_action_param）：
  - strategy:       策略名，默认 fixed（固定价格 1，即改造前的行为）
  - strategy_param: 该策略自己的参数（前端每个策略 case 提供自己那套）
  - debug:          是否打印 OCR 调试信息

屏幕坐标（数字键盘、输入框、清空按钮等）一律来自 .layout，
不再通过 custom_action_param 传入，避免被前端覆盖时丢失。
假设分辨率：1280×720，坐标直接用
"""

import time
from typing import Optional

from maa.context import Context
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction

from ..Common.utils import click_rect, load_params
from utils.maafocus import PrintT
from utils.logger import logger

from .layout import CLEAR_BTN, DIGIT_BUTTONS, INPUT_ROI
from .strategies import DEFAULT_STRATEGY, get_strategy
from .utils import extract_price, read_region_text



# ============ 公共输入层 ============

def clear_and_type(controller, clear_btn, target: int) -> None:
    """清空输入框后逐位点击数字按钮"""
    click_rect(controller, clear_btn, delay=0.2)

    #将target转换为str，遍历每个字符并且转为整数
    digits = [int(d) for d in str(target)]
    logger.info(f"⌨️ 输入数字序列: {digits}")
    for i, d in enumerate(digits):
        btn = DIGIT_BUTTONS.get(str(d))
        if btn is None:
            raise RuntimeError(f"未找到数字 {d} 的按钮坐标")
        logger.info(f"  [{i + 1}/{len(digits)}] 点击 {d}")
        click_rect(controller, btn, delay=0.15)


def verify_readback(
    context: Context,
    controller,
    input_roi,
    clear_btn,
    target: int,
    debug: bool,
    retried: bool = False,
) -> bool:
    """
    回读输入框并校验；不一致时清空重输一次。

    失败策略：
    - 完全读不到 → 不重输（大概率是 OCR 的问题，不是点错）
    - 不一致     → 重输一次；再不一致则停止重试，交由 Pipeline 继续确认
    """
    time.sleep(0.2)  # 等最后一位数字上屏
    text = read_region_text(context, controller, input_roi, debug)
    actual = extract_price(text)

    if actual is None:
        logger.warning("⚠️ 回读读不到数字，按已输入内容继续")
        PrintT(context, "bidking.readback_unreadable")
        return False

    if actual == target:
        return True

    logger.warning(f"⚠️ 回读不一致: 目标 {target}, 实际 {actual}")
    PrintT(context, "bidking.readback_mismatch", target, actual)

    if retried:
        logger.warning("⚠️ 已重输过一次，不再重试，交由 Pipeline 确认")
        return False

    clear_and_type(controller, clear_btn, target)
    return verify_readback(
        context, controller, input_roi, clear_btn, target, debug, retried=True
    )


# ============ 自定义动作 ============

@AgentServer.custom_action("place_bid")
class PlaceBid(CustomAction):
    """按所选策略算出本轮出价额并输入；确认出价由 Pipeline 点击"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        try:
            param = load_params(getattr(argv, "custom_action_param", None))

            strategy_param = param.get("strategy_param")
            if not isinstance(strategy_param, dict):
                if strategy_param is not None:
                    logger.warning(
                        f"⚠️ strategy_param 不是字典（{type(strategy_param).__name__}），按空参数处理"
                    )
                strategy_param = {}

            requested = str(param.get("strategy") or DEFAULT_STRATEGY)
            input_roi = INPUT_ROI
            clear_btn = CLEAR_BTN
            debug = bool(param.get("debug", False))

            # ---------- 1. 取策略（未知策略回落到默认策略） ----------
            decide, fell_back = get_strategy(requested)
            strategy_name = DEFAULT_STRATEGY if fell_back else requested
            if fell_back:
                logger.warning(
                    f"⚠️ 未知策略 {requested!r}，回落到 {DEFAULT_STRATEGY!r}"
                )
                PrintT(context, "bidking.strategy_unknown", requested, DEFAULT_STRATEGY)

            controller = context.tasker.controller
            ui = {"debug": debug}

            # ---------- 2. 让策略算出本轮出价额 ----------
            target = decide(context, controller, strategy_param, ui)

            if target is None or target <= 0:
                if target is not None:
                    logger.warning(f"⚠️ 策略返回非正数 {target}，按放弃本轮处理")
                logger.info("🛑 本轮不出价，面板交由 Pipeline 的 BidKingClosePanel 收起")
                PrintT(context, "bidking.skip_round", strategy_name)
                return CustomAction.RunResult(success=True)

            logger.info(f"🎯 本轮出价额: {target}（策略={strategy_name}）")

            # ---------- 3. 公共输入层：清空 + 逐位输入 + 回读校验 ----------
            clear_and_type(controller, clear_btn, target)
            verify_readback(context, controller, input_roi, clear_btn, target, debug)

            PrintT(context, "bidking.bid_done", strategy_name, target)
            return CustomAction.RunResult(success=True)

        except Exception as exc:
            logger.error(f"❌ PlaceBid 异常: {exc}")
            import traceback

            logger.error(traceback.format_exc())
            PrintT(context, "bidking.failed", str(exc))
            return CustomAction.RunResult(success=False)
