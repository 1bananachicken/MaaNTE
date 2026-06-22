# -*- coding: utf-8 -*-
"""羁绊送礼 — 自动给指定角色送指定礼物。

用法:
    配置字符串: 星见雅:巧克力,花束; 其他角色:万能礼物
    格式: 角色名:礼物1,礼物2;角色名:礼物3

流程:
    Pipeline: 确认主页 → 按ESC → 等菜单稳定 → 进 Python
    Python:   context.run_task + pipeline_override 跑 TemplateMatch+Click → 进界面 → 送礼
    点击:     Common/utils (click_rect / click_rect_multiple)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch

from utils.logger import logger
from utils.maafocus import Print
from agent.custom.action.Common.utils import click_rect, click_rect_multiple, get_image
import cv2


# ═══════════════════════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════════════════════

CHAR_LIST_ROI = [900, 100, 350, 600]
CHAR_SWIPE_BEGIN = (1070, 400)
CHAR_SWIPE_END = (1070, 200)
CHAR_SWIPE_DURATION_MS = 300
CHAR_MAX_SCROLLS = 20
CHAR_RESET_SWIPES = 3

GIFT_TITLE_ROI = [200, 50, 400, 60]
GIFT_TITLE_WAIT_MS = 0.6

GIFT_LIST_ROI = [50, 150, 400, 500]
GIFT_LONG_PRESS_MS = 800
GIFT_POPUP_NAME_ROI = [100, 200, 300, 80]
GIFT_GIVE_BUTTON_ROI = [100, 500, 200, 80]
GIFT_SWIPE_BEGIN = (250, 500)
GIFT_SWIPE_END = (250, 300)
GIFT_SWIPE_DURATION_MS = 300
GIFT_MAX_SCROLLS = 15
GIFT_GIVE_WAIT_MS = 1.2

FULL_HINT_TEXT = "0/3"
FULL_HINT_ROI = [100, 480, 200, 60]
GLOBAL_LIMIT_EXPECTED = ["上限", "次数", "无法赠送"]
ERROR_EXPECTED = ["无法", "不能", "失败", "Cannot", "failed"]

# 羁遇按钮识别参数
JIYU_TEMPLATE = "BondGift/jiyu.png"
# JIYU_TEMPLATE = "BagelSpam/camera.png"
JIYU_TEMPLATE_THRESHOLD = 0.65
JIYU_ENTRY_ATTEMPTS = 3          # 最多重试点击次数
JIYU_CLICK_INTERVAL_MS = 1.5     # 两次点击间隔

ItemListROI = [510, 215, 605, 300]
CharacterListROI = [1115, 15, 165, 545]

AnHunQu_Template = "BondGift/anhunqu.png"
Zengli_Template = "BondGift/zengli.png"

Nanali_Template = "BondGift/nanali.png"
Nanali_Gift_Templates = (
    "BondGift/nanali_test.png",
    "BondGift/nanali_latiao.png",
)



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
    for segment in re.split(r"[;\uff1b]", raw):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        parts = segment.split(":", 1)
        char = parts[0].strip()
        gifts = [g.strip() for g in parts[1].split(",") if g.strip()] if len(parts) > 1 else []
        if char and gifts:
            targets.append(BondGiftTarget(character=char, gifts=gifts))
            logger.info("BondGift: '%s' ← %s", char, gifts)
    logger.info("BondGift: 解析 %d 个角色", len(targets))
    return targets


# ═══════════════════════════════════════════════════════════════════════
# 控制辅助
# ═══════════════════════════════════════════════════════════════════════

def _screencap(context: Context):
    img = get_image(context.tasker.controller)
    return img.copy()  # 深拷贝，防止内部缓存覆盖导致 access violation


def _press_esc(context: Context):
    context.tasker.controller.post_press_key(27).wait()
    time.sleep(0.6)


def _swipe(context: Context, begin, end, duration_ms: int = 300):
    context.tasker.controller.post_swipe(begin[0], begin[1], end[0], end[1],
                                         duration=duration_ms).wait()
    time.sleep(0.15)


def _click_at(context: Context, x: int, y: int):
    ctrl = context.tasker.controller
    click_rect(ctrl, [x, y, 1, 1], delay=0.05)
    time.sleep(0.05)


def _long_press_at(context: Context, x: int, y: int, duration_ms: int = 800):
    ctrl = context.tasker.controller
    ctrl.post_touch_down(x, y).wait()
    time.sleep(duration_ms / 1000.0)
    ctrl.post_touch_up().wait()
    time.sleep(0.3)


def _click_blank(context: Context, x: int = 40, y: int = 40):
    _click_at(context, x, y)
    time.sleep(0.2)


def _ocr_text(context: Context, image, roi: list[int],
              expected: list[str], threshold: float = 0.6) -> Optional[str]:
    result = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(roi=tuple(roi), expected=expected, threshold=threshold),
        image,
    )
    if result and result.hit:
        text = result.detail.get("text", "")
        logger.info("OCR hit roi=%s → '%s'", roi, text)
        return text
    logger.info("OCR miss roi=%s expected=%s", roi, expected)
    return None


def _ocr_matches(context: Context, image, roi: list[int],
                 expected: list[str], threshold: float = 0.6) -> bool:
    result = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(roi=tuple(roi), expected=expected, threshold=threshold),
        image,
    )
    return bool(result and result.hit)


# ═══════════════════════════════════════════════════════════════════════
# 滚动辅助
# ═══════════════════════════════════════════════════════════════════════

def _find_and_click_in_list(context, roi, template, swipe_begin, swipe_end,
                           label="", threshold=0.75, max_scrolls=10,
                           swipe_duration_ms=400, post_delay_ms=600):
    """边滚边识别：每次 swipe 后截图 → TemplateMatch。命中 → click → return True。"""
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
            logger.info("_find_click: 命中 '%s' scroll=%d box=(%d,%d,%d,%d)",
                        label, i, box.x, box.y, box.w, box.h)
            ret = context.run_task("_ClickTarget", pipeline_override={
                "_ClickTarget": {
                    "recognition": {"type": "DirectHit"},
                    "action": {
                        "type": "Click",
                        "param": {"target": [box.x, box.y, box.w, box.h]},
                    },
                }
            })
            logger.info("_find_click: '%s' run_task → %s", label, ret)
            return True

        # 没找到 → 滑一次
        logger.info("_find_click: '%s' scroll=%d 未命中, 执行 swipe", label, i)
        context.run_task("_SwipeNode", pipeline_override={
            "_SwipeNode": {
                "recognition": {"type": "DirectHit"},
                "action": {
                    "type": "Swipe",
                    "begin": [swipe_begin[0], swipe_begin[1]],
                    "end": [swipe_end[0], swipe_end[1]],
                    "duration": swipe_duration_ms,
                },
                "post_delay": post_delay_ms,
            }
        })

    logger.warning("_find_click: '%s' 未找到, 达到上限 %d", label, max_scrolls)
    return False


# ═══════════════════════════════════════════════════════════════════════
# 主 Action
# ═══════════════════════════════════════════════════════════════════════

@AgentServer.custom_action("auto_bonding_gift")
class AutoBondingGift(CustomAction):
    """羁绊送礼 — 自闭环：找羁遇按钮 → 点击 → 进界面 → 送礼。"""

    def _test_main_logic(self, context: Context) -> bool:
        """测试主逻辑：OCR 找「赠礼」+ TemplateMatch 找暗魂曲，点击。

        稍后要把方法名改回来，现在先收集数据。
        """
        ctrl = context.tasker.controller
        image = _screencap(context)
        logger.info("_test_main_logic: 开始调试, 截图=%dx%d", image.shape[1], image.shape[0])

        # ── TemplateMatch: 找「赠礼」 ──
        logger.info("_test_main_logic: TemplateMatch 搜索「赠礼」...")
        zengli_roi = [0, 0, 1280, 720]  # 全屏搜，稍后缩窄
        zengli_result = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                roi=tuple(zengli_roi),
                template=[Zengli_Template],
                threshold=[0.65],
                order_by="Score",
            ),
            image,
        )
        if zengli_result and zengli_result.hit and zengli_result.box:
            zbox = zengli_result.box
            logger.info("_test_main_logic: TemplateMatch 命中「赠礼」 box=(%d,%d,%d,%d)",
                        zbox.x, zbox.y, zbox.w, zbox.h)
            logger.info("_test_main_logic: run_task 点击「赠礼」")
            ret = context.run_task("_ClickZengli", pipeline_override={
                "_ClickZengli": {
                    "recognition": {"type": "DirectHit"},
                    "action": {
                        "type": "Click",
                        "param": {"target": [zbox.x, zbox.y, zbox.w, zbox.h]},
                    },
                }
            })
            logger.info("_test_main_logic: 「赠礼」run_task → %s", ret)
            Print(context, "测试: TemplateMatch 点「赠礼」 ✓")
        else:
            logger.info("_test_main_logic: TemplateMatch 未命中「赠礼」")
            Print(context, "测试: TemplateMatch 未命中「赠礼」 ✗")

        # # ── TemplateMatch: 找暗魂曲 ──
        # logger.info("_test_main_logic: TemplateMatch 搜索暗魂曲...")
        # anhunqu_roi = [1150, 75, 130, 550]  # [x, y, w, h] — 稍后测量
        # # anhunqu_roi = [0, 0, 1280, 720]
        # anhunqu_result = context.run_recognition_direct(
        #     JRecognitionType.TemplateMatch,
        #     JTemplateMatch(
        #         roi=tuple(anhunqu_roi),
        #         template=[AnHunQu_Template],
        #         threshold=[0.65],
        #         order_by="Score",
        #     ),
        #     image,
        # )
        # if anhunqu_result and anhunqu_result.hit and anhunqu_result.box:
        #     abox = anhunqu_result.box
        #     logger.info("_test_main_logic: TemplateMatch 命中暗魂曲 box=(%d,%d,%d,%d)",
        #                 abox.x, abox.y, abox.w, abox.h)
        #     logger.info("_test_main_logic: run_task 点击「暗魂曲」")
        #     ret = context.run_task("_ClickAnHunQu", pipeline_override={
        #         "_ClickAnHunQu": {
        #             "recognition": {"type": "DirectHit"},
        #             "action": {
        #                 "type": "Click",
        #                 "param": {"target": [abox.x, abox.y, abox.w, abox.h]},
        #             },
        #         }
        #     })
        #     logger.info("_test_main_logic: 「暗魂曲」run_task → %s", ret)
        #     Print(context, "测试: TemplateMatch 点暗魂曲 ✓")
        # else:
        #     logger.info("_test_main_logic: TemplateMatch 未命中暗魂曲")
        #     Print(context, "测试: TemplateMatch 未命中暗魂曲 ✗")

        # ── ① 边滚边找 Nanali 角色 ──
        logger.info("_test_main_logic: 边滚边找 Nanali 角色...")
        found = _find_and_click_in_list(
            context,
            roi=CharacterListROI,
            template=Nanali_Template,
            swipe_begin=(1195, 420),
            swipe_end=(1195, 280),
            label="Nanali角色",
        )
        if found:
            Print(context, "测试: 点 Nanali 角色 ✓")
        else:
            Print(context, "测试: 未找到 Nanali 角色 ✗")

        # ── ② 边滚边找 Nanali 礼物 ──
        logger.info("_test_main_logic: 边滚边找 Nanali 礼物...")
        found = _find_and_click_in_list(
            context,
            roi=ItemListROI,
            template=Nanali_Gift_Templates[0],
            swipe_begin=(810, 420),
            swipe_end=(810, 320),
            label="Nanali礼物",
        )
        if found:
            Print(context, "测试: 点 Nanali 礼物 ✓")
        else:
            Print(context, "测试: 未找到 Nanali 礼物 ✗")
        logger.info("_test_main_logic: 调试结束")
        return True

    # ── 进入羁绊界面 ──

    def _enter_bonding_interface(self, context: Context) -> bool:
        """用 MAA pipeline_override 跑 TemplateMatch + Click, 测试框架点击。"""
        for attempt in range(JIYU_ENTRY_ATTEMPTS):
            if context.tasker.stopping:
                return False
            Print(context, "羁绊送礼: 换long press")
            override = {
                "BondGiftCameraTest": {
                    "recognition": {
                        "type": "TemplateMatch",
                        "param": {
                            "roi": [0, 0, 1280, 720],
                            "template": JIYU_TEMPLATE,
                            "threshold": JIYU_TEMPLATE_THRESHOLD,
                            "order_by": "Score",
                        },
                    },
                    "action": {
                        "type": "LongPress",
                        "duration": 100,
                    },
                    "pre_delay": 200,
                    "post_delay": 1500,
                }
            }
            Print(context, f"羁绊送礼: MAA 找+点 羁遇 (第{attempt+1}次)")
            ret = context.run_task("BondGiftCameraTest", pipeline_override=override)
            logger.info("BondGift: run_task → %s", ret)

            if context.tasker.stopping:
                return False

            # 验证 ESC 菜单是否消失
            time.sleep(0.5)
            image = _screencap(context)
            still_in_esc = _ocr_matches(
                context, image,
                roi=[918, 101, 312, 165],
                expected=["猎人等级", "獵人等級", "(?i)Hunter"],
                threshold=0.7,
            )
            if not still_in_esc:
                Print(context, "羁绊送礼: 进入羁绊界面 ✓")
                return True

            Print(context, f"羁绊送礼: 仍在ESC菜单 (第{attempt+1}次)")

        Print(context, "羁绊送礼: 无法进入羁绊界面 ✗")
        return False

    # ── 主入口 ──

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        t_start = time.perf_counter()

        # 解析参数
        params = self._parse_params(argv)
        config_str = params.get("config", "")
        max_per_char = int(params.get("max_per_char", "3") or "3")
        max_total = int(params.get("max_total", "10") or "10")

        targets = _parse_config(config_str)
        if not targets:
            Print(context, "羁绊送礼: 未配置, 跳过")
            return CustomAction.RunResult(success=True)

        logger.info("BondGift: 开始, %d个角色, per_char=%d, total=%d",
                     len(targets), max_per_char, max_total)
        Print(context, f"羁绊送礼: 开始 ({len(targets)}角色)")

        self._test_main_logic(context)
        # ── Phase 1: 进入羁绊界面 ──
        # if not self._enter_bonding_interface(context):
        #     logger.warning("BondGift: 未能进入羁绊界面")
        #     return CustomAction.RunResult(success=True)
        return CustomAction.RunResult(success=True)

        # ── Phase 2: 送礼 ──
        total_given = 0
        per_char_given: dict[str, int] = {}

        for ti, target in enumerate(targets):
            if context.tasker.stopping:
                break
            if total_given >= max_total:
                Print(context, "羁绊送礼: 已达全局上限")
                break

            char_name = target.character
            if per_char_given.get(char_name, 0) >= max_per_char:
                continue

            logger.info("BondGift: [%d/%d] '%s'", ti + 1, len(targets), char_name)

            found = self._select_character(context, char_name)
            if not found:
                Print(context, f"羁绊送礼: 未找到 '{char_name}', 跳过")
                continue

            for gift_name in target.gifts:
                if context.tasker.stopping:
                    break
                if total_given >= max_total:
                    break

                char_count = per_char_given.get(char_name, 0)
                if char_count >= max_per_char:
                    break

                for gift_attempt in range(2):
                    if context.tasker.stopping:
                        break
                    result = self._give_gift(context, gift_name)

                    if result == "ok":
                        total_given += 1
                        per_char_given[char_name] = char_count + 1
                        Print(context, f"'{char_name}' ← '{gift_name}' ✓ ({total_given}/{max_total})")
                        break
                    elif result == "char_full":
                        per_char_given[char_name] = max_per_char
                        break
                    elif result == "global_limit":
                        total_given = max_total
                        break
                    elif result == "not_found":
                        Print(context, f"未找到礼物 '{gift_name}'")
                        break
                    else:
                        if gift_attempt == 0:
                            _press_esc(context)
                            time.sleep(0.5)
                            if not self._select_character(context, char_name):
                                break
                        else:
                            logger.warning("BondGift: 恢复后仍失败, 跳过 '%s'", gift_name)

                if self._check_global_limit(context):
                    total_given = max_total
                    break
                if self._check_char_full(context):
                    per_char_given[char_name] = max_per_char
                    break

        elapsed = time.perf_counter() - t_start
        logger.info("BondGift: 完成, 送出%d个, %.1fs", total_given, elapsed)
        Print(context, f"羁绊送礼: 完成! {total_given}个 ({elapsed:.0f}s)")
        return CustomAction.RunResult(success=True)

    # ── 参数解析 ──

    @staticmethod
    def _parse_params(argv: CustomAction.RunArg) -> dict:
        param = argv.custom_action_param
        if isinstance(param, str):
            try:
                return json.loads(param)
            except json.JSONDecodeError:
                return {}
        if isinstance(param, dict):
            return param
        return {}

    # ── 角色选择 ──

    def _select_character(self, context: Context, char_name: str) -> bool:
        for scroll_idx in range(CHAR_MAX_SCROLLS + 1):
            if context.tasker.stopping:
                return False
            step_y = 80
            for i in range(6):
                if context.tasker.stopping:
                    return False
                cy = CHAR_LIST_ROI[1] + 40 + i * step_y
                cx = CHAR_LIST_ROI[0] + CHAR_LIST_ROI[2] // 2
                _click_at(context, cx, cy)
                time.sleep(GIFT_TITLE_WAIT_MS)
                image = _screencap(context)
                title = _ocr_text(context, image, GIFT_TITLE_ROI, [char_name], threshold=0.5)
                if title:
                    logger.info("BondGift: 选中 '%s' (scroll=%d, idx=%d)", char_name, scroll_idx, i)
                    return True

            if scroll_idx >= CHAR_MAX_SCROLLS - CHAR_RESET_SWIPES:
                for _ in range(CHAR_RESET_SWIPES):
                    if context.tasker.stopping:
                        return False
                    _swipe(context, CHAR_SWIPE_END, CHAR_SWIPE_BEGIN, CHAR_SWIPE_DURATION_MS)
                scroll_idx = 0
                continue
            if scroll_idx < CHAR_MAX_SCROLLS:
                _swipe(context, CHAR_SWIPE_BEGIN, CHAR_SWIPE_END, CHAR_SWIPE_DURATION_MS)

        logger.warning("BondGift: 未找到角色 '%s'", char_name)
        return False

    # ── 礼物赠送 ──

    def _give_gift(self, context: Context, gift_name: str) -> str:
        for scroll_idx in range(GIFT_MAX_SCROLLS + 1):
            if context.tasker.stopping:
                return "fail"
            step_y = 80
            for i in range(6):
                if context.tasker.stopping:
                    return "fail"
                cy = GIFT_LIST_ROI[1] + 40 + i * step_y
                cx = GIFT_LIST_ROI[0] + GIFT_LIST_ROI[2] // 2
                _long_press_at(context, cx, cy, duration_ms=GIFT_LONG_PRESS_MS)
                time.sleep(0.3)
                image = _screencap(context)
                name = _ocr_text(context, image, GIFT_POPUP_NAME_ROI, [gift_name], threshold=0.5)
                if name:
                    logger.info("BondGift: 找到礼物 '%s' (scroll=%d, idx=%d)", gift_name, scroll_idx, i)
                    _click_at(
                        context,
                        GIFT_GIVE_BUTTON_ROI[0] + GIFT_GIVE_BUTTON_ROI[2] // 2,
                        GIFT_GIVE_BUTTON_ROI[1] + GIFT_GIVE_BUTTON_ROI[3] // 2,
                    )
                    time.sleep(GIFT_GIVE_WAIT_MS)
                    return "ok"

            if scroll_idx >= GIFT_MAX_SCROLLS - 3:
                for _ in range(3):
                    if context.tasker.stopping:
                        return "fail"
                    _swipe(context, GIFT_SWIPE_END, GIFT_SWIPE_BEGIN, GIFT_SWIPE_DURATION_MS)
                scroll_idx = 0
                continue
            if scroll_idx < GIFT_MAX_SCROLLS:
                _swipe(context, GIFT_SWIPE_BEGIN, GIFT_SWIPE_END, GIFT_SWIPE_DURATION_MS)

        logger.warning("BondGift: 未找到礼物 '%s'", gift_name)
        return "not_found"

    # ── 限制检查 ──

    def _check_char_full(self, context: Context) -> bool:
        image = _screencap(context)
        return _ocr_matches(context, image, FULL_HINT_ROI, [FULL_HINT_TEXT], threshold=0.6)

    def _check_global_limit(self, context: Context) -> bool:
        image = _screencap(context)
        for expected in GLOBAL_LIMIT_EXPECTED:
            if _ocr_matches(context, image, [0, 0, 1280, 720], [expected], threshold=0.6):
                Print(context, f"羁绊送礼: 全局限制 (OCR: '{expected}')")
                return True
        return False

