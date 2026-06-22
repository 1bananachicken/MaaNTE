# -*- coding: utf-8 -*-
"""羁绊送礼 — 自动给指定角色送指定礼物。

用法:
    配置字符串: 角色ID:礼物ID1,礼物ID2; 角色ID:礼物ID3
    格式: nanali:test_gift,spicy_strip

流程:
    Pipeline: 确认ESC菜单 → 找羁遇按钮 → 进界面 → 找赠礼按钮
    Python:   滑动找角色 → 滑动找礼物 → 点赠送按钮 → 检查上限 → 循环
    点击:     Common/utils (click_rect / post_swipe)
"""

from __future__ import annotations

import cv2
import time
from dataclasses import dataclass, field
from typing import Any

from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JRecognitionType, JTemplateMatch

from utils.logger import logger
from utils.maafocus import Print
from agent.custom.action.Common.utils import click_rect, get_image
from wrap import task_action


# ═══════════════════════════════════════════════════════════════════════
# 内置角色-礼物映射
# ═══════════════════════════════════════════════════════════════════════

BOND_GIFT_PRESETS: dict[str, dict] = {
    "nanali": {
        "template": "BondGift/nanali.png",
        "gifts": {
            "test_gift": "BondGift/nanali_test.png",
            "spicy_strip": "BondGift/nanali_latiao.png",
        },
    },
    # 新角色在这里加: 模板图 + 可用礼物 + 礼物模板图
}


# ═══════════════════════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════════════════════

ItemListROI = [510, 215, 605, 300]
CharacterListROI = [1115, 15, 165, 545]

CHAR_SWIPE_BEGIN = (1195, 420)
CHAR_SWIPE_END = (1195, 280)
GIFT_SWIPE_BEGIN = (810, 420)
GIFT_SWIPE_END = (810, 320)

GIFT_LIMIT_ROI = [154, 338, 972, 36]
GIFT_LIMIT_TEMPLATE = "BondGift/gift_to_limit.png"
GIFT_LIMIT_THRESHOLD = 0.7

FIND_THRESHOLD = 0.75
FIND_MAX_SCROLLS = 15
FIND_SWIPE_DURATION_MS = 400
FIND_POST_DELAY_MS = 600
SCROLL_STUCK_THRESHOLD = 0.92


# ═══════════════════════════════════════════════════════════════════════
# 配置解析
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BondGiftTarget:
    character: str
    gifts: list[str] = field(default_factory=list)


def _parse_config(raw: str) -> list[BondGiftTarget]:
    if not raw or not raw.strip():
        return []
    targets: list[BondGiftTarget] = []
    raw = raw.replace("\uff1a", ":").replace("\uff0c", ",").replace("\uff1b", ";")
    for segment in raw.split(";"):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        parts = segment.split(":", 1)
        char = parts[0].strip()
        gifts = [g.strip() for g in parts[1].split(",") if g.strip()]
        if not char or not gifts:
            continue
        if char not in BOND_GIFT_PRESETS:
            logger.warning("BondGift: 未知角色 '%s', 跳过", char)
            continue
        valid_gifts = []
        for g in gifts:
            if g in BOND_GIFT_PRESETS[char]["gifts"]:
                valid_gifts.append(g)
            else:
                logger.warning("BondGift: 未知礼物 '%s' (角色 '%s'), 跳过", g, char)
        if valid_gifts:
            targets.append(BondGiftTarget(character=char, gifts=valid_gifts))
            logger.info("BondGift: '%s' ← %s", char, valid_gifts)
    logger.info("BondGift: 解析 %d 个角色", len(targets))
    return targets


# ═══════════════════════════════════════════════════════════════════════
# 控制辅助
# ═══════════════════════════════════════════════════════════════════════

def _screencap(context: Context):
    img = get_image(context.tasker.controller)
    return img.copy()  # 深拷贝，防止内部缓存覆盖导致 access violation


def _swipe(context: Context, begin, end, duration_ms: int = 300):
    context.tasker.controller.post_swipe(
        begin[0], begin[1], end[0], end[1], duration=duration_ms
    ).wait()
    time.sleep(0.15)


def _swipe_to_top(context: Context, roi: list[int], swipe_begin, swipe_end, times: int = FIND_MAX_SCROLLS):
    """反向滑动重置列表到顶部，检测到尽头提前退出。"""
    prev_gray = None
    for _ in range(times):
        if context.tasker.stopping:
            return
        image = _screencap(context)
        is_stuck, prev_gray = _is_list_stuck(image, roi, prev_gray)
        if is_stuck:
            logger.info("_swipe_to_top: 已到顶部, 停止")
            return
        _swipe(context, swipe_end, swipe_begin, FIND_SWIPE_DURATION_MS)


def _is_list_stuck(image, roi: list[int], prev_gray) -> tuple[bool, Any]:
    """比较 ROI 区域灰度图判断列表是否已滑动到尽头。返回 (是否到尽头, 当前灰度图)。"""
    x, y, w, h = roi
    curr_gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return False, curr_gray
    result = cv2.matchTemplate(prev_gray, curr_gray, cv2.TM_CCOEFF_NORMED)
    score = float(result.max())
    return score >= SCROLL_STUCK_THRESHOLD, curr_gray


# ═══════════════════════════════════════════════════════════════════════
# 滚动查找辅助
# ═══════════════════════════════════════════════════════════════════════

def _find_and_click_in_list(
    context: Context,
    roi: list[int],
    template: str,
    swipe_begin: tuple[int, int],
    swipe_end: tuple[int, int],
    label: str = "",
    threshold: float = FIND_THRESHOLD,
    max_scrolls: int = FIND_MAX_SCROLLS,
    swipe_duration_ms: int = FIND_SWIPE_DURATION_MS,
    post_delay_ms: int = FIND_POST_DELAY_MS,
) -> bool:
    """边滚边识别：截图 → TemplateMatch。命中 → click → return True。未命中 → 检测到尽头 → swipe → 继续。"""
    controller = context.tasker.controller
    prev_gray = None
    for i in range(max_scrolls):
        if context.tasker.stopping:
            return False

        # 截图 + 识别
        image = _screencap(context)
        result = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                roi=tuple(roi),
                template=[template],
                threshold=[threshold],
                order_by="Score",
            ),
            image,
        )
        if result and result.hit and result.box:
            box = result.box
            logger.info(
                "_find_click: 命中 '%s' scroll=%d box=(%d,%d,%d,%d)",
                label, i, box.x, box.y, box.w, box.h,
            )
            click_rect(controller, [box.x, box.y, box.w, box.h])
            time.sleep(post_delay_ms / 1000.0)
            return True

        # 未命中 → 截 ROI 灰度判断是否到尽头
        is_stuck, prev_gray = _is_list_stuck(image, roi, prev_gray)
        if is_stuck:
            logger.info("_find_click: '%s' scroll=%d 已到尽头, 停止", label, i)
            break

        # 没到底 → 滑一次
        logger.info("_find_click: '%s' scroll=%d 未命中, 执行 swipe", label, i)
        _swipe(context, swipe_begin, swipe_end, swipe_duration_ms)

    logger.warning("_find_click: '%s' 未找到, 达到上限 %d", label, max_scrolls)
    return False


def _check_char_limit(context: Context) -> bool:
    """检查单角色是否已达上限（TemplateMatch gift_to_limit.png）。"""
    image = _screencap(context)
    result = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            roi=tuple(GIFT_LIMIT_ROI),
            template=[GIFT_LIMIT_TEMPLATE],
            threshold=[GIFT_LIMIT_THRESHOLD],
            order_by="Score",
        ),
        image,
    )
    if result and result.hit:
        logger.info("BondGift: 识别到单角色上限标记")
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# 配置 dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BondGiftConfig:
    config: str = ""
    max_per_char: int = 3
    max_total: int = 10


# ═══════════════════════════════════════════════════════════════════════
# 主 Action
# ═══════════════════════════════════════════════════════════════════════

@task_action("auto_bonding_gift", BondGiftConfig)
class AutoBondingGift(CustomAction):
    """羁绊送礼 — Python 负责滑动找角色、找礼物、点赠送、检查上限。"""

    def run(self, context: Context, cfg: BondGiftConfig, argv: CustomAction.RunArg) -> bool:
        t_start = time.perf_counter()

        targets = _parse_config(cfg.config)
        if not targets:
            Print(context, "羁绊送礼: 未配置或配置无效, 跳过")
            return True

        logger.info(
            "BondGift: 开始, %d个角色, per_char=%d, total=%d",
            len(targets), cfg.max_per_char, cfg.max_total,
        )
        Print(context, f"羁绊送礼: 开始 ({len(targets)}角色)")

        total_given = 0
        per_char_given: dict[str, int] = {}

        for ti, target in enumerate(targets):
            if context.tasker.stopping:
                break
            if total_given >= cfg.max_total:
                Print(context, "羁绊送礼: 已达全局上限")
                break

            char_name = target.character
            if per_char_given.get(char_name, 0) >= cfg.max_per_char:
                continue

            char_template = BOND_GIFT_PRESETS[char_name]["template"]
            logger.info("BondGift: [%d/%d] '%s'", ti + 1, len(targets), char_name)

            # ── 步骤4: 滑动找角色并点击 ──
            _swipe_to_top(context, CharacterListROI, CHAR_SWIPE_BEGIN, CHAR_SWIPE_END)
            found = _find_and_click_in_list(
                context,
                roi=CharacterListROI,
                template=char_template,
                swipe_begin=CHAR_SWIPE_BEGIN,
                swipe_end=CHAR_SWIPE_END,
                label=char_name,
            )
            if not found:
                Print(context, f"羁绊送礼: 未找到 '{char_name}', 跳过")
                continue

            # ── 步骤5-6: 遍历礼物，找礼物→点赠送→检查上限 ──
            for gift_name in target.gifts:
                if context.tasker.stopping:
                    break
                if total_given >= cfg.max_total:
                    break
                if per_char_given.get(char_name, 0) >= cfg.max_per_char:
                    break

                gift_template = BOND_GIFT_PRESETS[char_name]["gifts"][gift_name]

                # 步骤5: 滑动找礼物并点击
                _swipe_to_top(context, ItemListROI, GIFT_SWIPE_BEGIN, GIFT_SWIPE_END)
                found = _find_and_click_in_list(
                    context,
                    roi=ItemListROI,
                    template=gift_template,
                    swipe_begin=GIFT_SWIPE_BEGIN,
                    swipe_end=GIFT_SWIPE_END,
                    label=gift_name,
                )
                if not found:
                    Print(context, f"羁绊送礼: 未找到礼物 '{gift_name}'")
                    continue

                # 步骤6: 点赠送按钮（pipeline 节点）
                ret = context.run_task("BondGiftClickGive")
                if not ret:
                    Print(context, f"羁绊送礼: 赠送按钮未找到, 跳过 '{gift_name}'")
                    continue

                # 步骤6: 计数 + 检查上限
                total_given += 1
                per_char_given[char_name] = per_char_given.get(char_name, 0) + 1
                Print(context, f"'{char_name}' ← '{gift_name}' ✓ ({total_given}/{cfg.max_total})")

                # B: TemplateMatch 检查单角色上限
                if _check_char_limit(context):
                    Print(context, f"羁绊送礼: '{char_name}' 已达单角色上限")
                    per_char_given[char_name] = cfg.max_per_char
                    break

        elapsed = time.perf_counter() - t_start
        logger.info("BondGift: 完成, 送出%d个, %.1fs", total_given, elapsed)
        Print(context, f"羁绊送礼: 完成! {total_given}个 ({elapsed:.0f}s)")
        return True
