"""
保底出价策略

每轮按游戏系统根据当轮情报披露算出的最低估价出价：
  1. 等画面稳定（披露动画期间顶栏显示的还是上一轮的值，稳定后才会跳新值）
  2. 读顶栏估价
兜底（与 bidking-bidding 能力契约一致）：
  - 等待画面稳定并短重试后估价仍为 0 → 出 1
  - 估价读不到 → 点面板内"上轮出价"沿用上一轮报价 → 仍为空/0 → 出 1

屏幕区域一律来自 ..layout（不通过参数传入），
稳定判定的时长与阈值是本策略自己的常量（见下方）：
  - SETTLE_TIME / SETTLE_TIMEOUT: 等画面稳定的判定与上限
  - SETTLE_EXTRA:                 情报卡片静止后，再等系统更新估价的上界
本策略目前不读 strategy_param（形参保留以符合策略接口）。
"""

import time
from typing import Optional

import numpy as np

from ...Common.utils import click_rect, get_image
from utils.maafocus import PrintT
from utils.logger import logger

from ..layout import PRICE_ROI, LAST_BID_BTN, SETTLE_ROI, INPUT_ROI
from ..utils import crop, extract_price, read_region_text
from . import register


# 判定画面稳定参数
SETTLE_TIME = 0.8          # 连续静止多久才认为披露动画结束
SETTLE_TIMEOUT = 12.0      # 等待上限（秒）
SETTLE_INTERVAL = 0.2      # 两次截图的间隔

# 情报卡片静止后的等待上限：此后系统还要更新包裹、再更新顶栏估价，
# 5 秒是实测足够的上界（估价更新本身看不见，只能等）。
SETTLE_EXTRA = 5.0

# 宁可"多等"也不要"误判稳定"：误判稳定会读到旧值，多等的代价只是等待变长
CHANGE_THRESHOLD = 10      # 单像素差异超过该值算"变了"（0-255）
CHANGE_RATIO = 0.001       # 变化像素占比低于该值算"没变"（容忍鼠标指针等）

# 稳定后仍读到 0/读不到时的短重试（新值可能晚一拍上屏）
PRICE_RETRY = 1.5
PRICE_RETRY_INTERVAL = 0.4

# 估价与"上轮出价"都无效时的最小出价（保证仍能参与后续轮次）
FALLBACK_PRICE = 1


@register("valuation")
def decide(context, controller, params: dict, ui: dict) -> Optional[int]:
    # 布局常量：不读 params，避免被前端覆盖时丢失
    price_roi = PRICE_ROI
    last_bid_btn = LAST_BID_BTN
    settle_roi = SETTLE_ROI

    input_roi = INPUT_ROI
    debug = bool(ui.get("debug", False))

    # ---------- 1. 等画面稳定 ----------
    if wait_until_stable(controller, settle_roi, SETTLE_TIME, SETTLE_TIMEOUT):
        time.sleep(SETTLE_EXTRA)

    # ---------- 2. 读系统最低估价 ----------
    text, price = read_price(context, controller, price_roi, debug)
    logger.info(f"💰 估价 OCR 原文: {text!r} -> {price}")

    if price is None:
        logger.warning(f"⚠️ 无法从 {text!r} 解析出估价")
        PrintT(context, "bidking.valuation_unreadable")
        last_bid = read_last_bid(
            context, controller, last_bid_btn, input_roi, debug
        )
        if last_bid:
            PrintT(context, "bidking.fallback_last_bid", last_bid)
            return last_bid
        logger.warning("⚠️ 上一轮报价也为空，改为出 1")
        PrintT(context, "bidking.fallback_one", FALLBACK_PRICE)
        return FALLBACK_PRICE

    if price <= 0:
        # 正常情况下等到画面稳定后估价就已更新；走到这里说明等待与重试后仍为 0
        logger.info("💰 等待稳定与短重试后估价仍为 0，本轮出 1")
        PrintT(context, "bidking.valuation_zero", FALLBACK_PRICE)
        return FALLBACK_PRICE

    return price


# ============ 内部实现 ============


def wait_until_stable(controller, settle_roi, settle_time, settle_timeout) -> bool:
    """
    等 settle_roi 区域连续静止 settle_time 秒，即情报披露动画播放完毕。
    返回是否等到了稳定；超时返回 False，按现状继续。
    """
    start = time.time()
    prev = None
    stable_since = None

    while time.time() - start < settle_timeout:
        img = get_image(controller)
        if img is None:
            logger.warning("⚠️ 截图失败，跳过稳定等待")
            return False

        cur = crop(img, settle_roi)
        if prev is not None and cur.shape == prev.shape:
            diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
            changed = float(np.count_nonzero(diff > CHANGE_THRESHOLD)) / diff.size
            if changed <= CHANGE_RATIO:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= settle_time:
                    logger.info(f"🖼️ 画面已稳定（等待 {time.time() - start:.1f}s）")
                    return True
            else:
                stable_since = None

        prev = cur
        time.sleep(SETTLE_INTERVAL)

    logger.warning(f"⚠️ 等待画面稳定超时（{settle_timeout:.1f}s），按当前画面继续")
    return False


def read_price(context, controller, price_roi, debug: bool):
    """
    读估价；读到 0 或读不到时在 PRICE_RETRY 秒内短重试
    （画面稳定后系统可能再晚一拍才把新值写上屏）
    """
    deadline = time.time() + PRICE_RETRY
    while True:
        text = read_region_text(context, controller, price_roi, debug)
        price = extract_price(text)
        if price or time.time() >= deadline:
            return text, price
        logger.info("⏳ 估价尚未更新（读数空或为 0），稍后重试")
        time.sleep(PRICE_RETRY_INTERVAL)


def read_last_bid(context, controller, last_bid_btn, input_roi, debug: bool):
    """点击面板内"上轮出价"并回读输入框；无效时返回 None"""
    if not input_roi:
        return None
    click_rect(controller, last_bid_btn, delay=0.2)
    time.sleep(0.3)  # 等输入框刷新

    text = read_region_text(context, controller, input_roi, debug)
    value = extract_price(text)
    if value is None or value <= 0:
        return None
    return value
