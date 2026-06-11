# -*- coding: utf-8 -*-
"""角色都市技能同步 — Custom Action

由 Pipeline 节点 ``SyncCharacterAbilityCityAbilityMain`` 驱动，
自定义动作名 ``SyncCharacterAbilityCityAbilityMainAction``。

遍历角色列表，OCR 识别每个角色的名字和两个都市技能等级，
通过 ``CharacterAbility_CityAbility`` 管理器持久化数据。

``custom_action_param``::

    {"fresh_record": true}   -- 全新记录模式：全部扫描完毕后清空旧数据再存入
    {} 或不传                -- 默认模式：逐角色保存
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JRecognitionType, JTemplateMatch

from custom.action.Common.CharacterAbility_CityAbility import (
    clear_all,
    set_character_abilities,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 子节点名称（对应 SyncCharacterAbilityCityAbility.json）
# ---------------------------------------------------------------------------
_NODE_OPEN_INFO = "SyncCharacterAbilityCityAbilityOpenInfoPage"
_NODE_OPEN_ABILITY = "SyncCharacterAbilityCityAbilityOpenAbilityPage"
_NODE_NEXT = "SyncCharacterAbilityCityAbilityNextCharacter"
_NODE_OCR_NAME = "SyncCharacterAbilityCityAbilityOCRName"
_NODE_OCR_SKILL0 = "SyncCharacterAbilityCityAbilityOCRSkill0"
_NODE_OCR_SKILL1 = "SyncCharacterAbilityCityAbilityOCRSkill1"
_NODE_MATCH_CHAR = "SyncCharacterAbilityCityAbilityMatchCharacter"
_CLICK_NODES = [
    "SyncCharacterAbilityCityAbilityCharacter2Move",
    "SyncCharacterAbilityCityAbilityCharacter3Move",
    "SyncCharacterAbilityCityAbilityCharacter4Move",
    "SyncCharacterAbilityCityAbilityCharacter5Move",
]

# OCR pipeline override — only_rec=True 让 OCR 返回 ROI 内所有文本，无需 expected
_OCR_OVERRIDE = {"recognition": {"param": {"only_rec": True}}}

_MAX_NO_CHANGE = 3  # NextCharacter 连续无变化，认为已到列表末尾
_MAX_ITERATIONS = 300  # 安全上限
_TM_THRESHOLD = 0.9  # TemplateMatch 置信度阈值，≥此值直接采用

# 恢复用 pipeline override — 1s 超时，忽略匹配结果
_RECOVERY_OVERRIDE = {"timeout": 1000}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _screencap(context: Context):
    """截取当前屏幕，返回 numpy 图像数组。"""
    return context.tasker.controller.post_screencap().wait().get()


def _ocr_on_image(context: Context, node_name: str, image) -> str | None:
    """对给定截图运行 OCR，返回命中的文本；失败返回 None。"""
    result = context.run_recognition(
        node_name,
        image,
        pipeline_override={node_name: _OCR_OVERRIDE},
    )
    if result is None or not result.hit:
        return None
    best = result.best_result
    if best is None:
        return None
    text: str = getattr(best, "text", "")
    return text.strip() or None


def _parse_skill_level(text: str | None, max_level: int) -> int:
    """解析技能 OCR 文本（如 ``"0/5"``）→ 当前等级整数。

    无法解析 / 技能不存在时返回 -1（即 N/A）。
    """
    if text is None:
        return -1
    # "当前/最大" 格式，如 "3/5"
    m = re.search(r"(\d+)\s*/\s*\d+", text)
    if m:
        level = int(m.group(1))
        if 0 <= level <= max_level:
            return level
    # 纯数字兜底
    m = re.search(r"(\d+)", text)
    if m:
        level = int(m.group(1))
        if 0 <= level <= max_level:
            return level
    return -1


def _ocr_skills(context: Context) -> list[int]:
    """OCR 技能等级，返回 ``[skill0, skill1]``。调用方负责确保已在技能页面。"""
    img = _screencap(context)
    skill0 = _parse_skill_level(
        _ocr_on_image(context, _NODE_OCR_SKILL0, img),
        max_level=5,
    )
    skill1 = _parse_skill_level(
        _ocr_on_image(context, _NODE_OCR_SKILL1, img),
        max_level=2,
    )
    return [skill0, skill1]


# ---------------------------------------------------------------------------
# 角色名识别 — OCR 为主，TemplateMatch 校验
# ---------------------------------------------------------------------------

# 模板文件名 → 角色名 映射。key 为文件名（含扩展名）；未命中时取 stem。
_TEMPLATE_TO_NAME: dict[str, str] = {
    "Chiz.png": "小吱",
    "Hotori.png": "浔",
    "Skia.png": "翳",
    "Zero.png": "零",
}


# TemplateMatch 兜底配置 — 与 JSON 中 SyncCharacterAbilityCityAbilityMatchCharacter 保持同步
_TM_CONFIG = {
    "templates": [
        "Character_UI/Character_Pic/Chiz.png",
        "Character_UI/Character_Pic/Hotori.png",
        "Character_UI/Character_Pic/Skia.png",
        "Character_UI/Character_Pic/Zero.png",
    ],
    "roi": (390, 80, 200, 210),
    "threshold": 0.8,
    "green_mask": True,
}


def _tm_get_name(context: Context) -> str | None:
    """逐模板运行 TemplateMatch，置信度 ≥ _TM_THRESHOLD 时返回角色名。

    优先从 pipeline JSON 动态读取配置，失败时回退到 ``_TM_CONFIG``。
    """
    img = _screencap(context)

    node_data = context.get_node_data(_NODE_MATCH_CHAR)
    if node_data:
        param = node_data.get("recognition", {}).get("param", {})
        templates = param.get("template", [])
        roi = param.get("roi", (0, 0, 0, 0))
        threshold = param.get("threshold", 0.8)
        green_mask = param.get("green_mask", False)
    else:
        logger.debug("SyncCityAbility: get_node_data 失败，使用硬编码 TM 配置")
        templates = _TM_CONFIG["templates"]
        roi = _TM_CONFIG["roi"]
        threshold = _TM_CONFIG["threshold"]
        green_mask = _TM_CONFIG["green_mask"]

    if not templates:
        return None

    for template_path in templates:
        result = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                template=[template_path],
                roi=tuple(roi) if roi else (0, 0, 0, 0),
                threshold=[threshold] if not isinstance(threshold, list) else threshold,
                green_mask=green_mask,
            ),
            img,
        )
        if result is None or not result.hit:
            continue
        best = result.best_result
        if best is None:
            continue
        score: float = getattr(best, "score", 0)
        if score < _TM_THRESHOLD:
            continue

        filename = Path(template_path).name
        stem = Path(template_path).stem
        name = _TEMPLATE_TO_NAME.get(filename, stem)
        logger.info("SyncCityAbility: TemplateMatch → %s (%.2f)", name, score)
        return name

    logger.debug("SyncCityAbility: TemplateMatch 无高置信度匹配")
    return None


def _ocr_name_with_retry(context: Context) -> str | None:
    """OCR 识别角色名字（重试），不含页面导航。"""
    for attempt in range(1, 4):
        img = _screencap(context)
        name = _ocr_on_image(context, _NODE_OCR_NAME, img)
        if name is not None:
            return name
        logger.debug("SyncCityAbility: OCR 名字第 %d 次失败", attempt)
        time.sleep(0.2)

    logger.info("SyncCityAbility: OCR 名字 3 次失败")
    return None


def _get_character_name(context: Context) -> str | None:
    """获取当前角色名：OCR → OpenAbilityPage → TemplateMatch 校验。

    调用前需在信息页面；调用后停留在技能页面。
    TM 置信度 ≥ 0.9 时覆盖 OCR 结果。
    """
    # 1. OCR 名字（信息页）
    name = _ocr_name_with_retry(context)
    if name is not None:
        logger.info("SyncCityAbility: OCR → %s", name)

    # 2. 进入技能页面
    context.run_task(_NODE_OPEN_ABILITY)

    # 3. TemplateMatch 校验 — 置信度 ≥ 0.9 则覆盖 OCR 结果
    tm_name = _tm_get_name(context)
    if tm_name is not None:
        return tm_name

    return name


# ---------------------------------------------------------------------------
# Custom Action
# ---------------------------------------------------------------------------


@AgentServer.custom_action("SyncCharacterAbilityCityAbilityMainAction")
class SyncCharacterAbilityCityAbilityMainAction(CustomAction):
    """角色都市技能同步主流程。

    遍历角色列表，OCR 识别每个角色的名字和两个都市技能等级，
    通过 ``CharacterAbility_CityAbility`` 管理器持久化数据。

    custom_action_param:
        {"fresh_record": true}  — 全新记录，扫描完一次性清空+存入
        不传或 {}               — 逐角色保存
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        # ---- 0. 解析参数 ----
        fresh_record = False
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                if isinstance(params, dict):
                    fresh_record = bool(params.get("fresh_record", False))
            except (json.JSONDecodeError, TypeError):
                pass

        logger.info(
            "SyncCityAbility: 开始遍历  fresh_record=%s",
            fresh_record,
        )

        # ---- 1. 清除锚点，防止内部 run_task(NextCharacter→Character1Click) 重新触发 Main ----
        context.set_anchor("CityAbilityAfterClick", "")

        # ---- 2. 尝试一次 OpenInfoPage，规范化页面状态（忽略结果） ----
        context.run_task(
            _NODE_OPEN_INFO,
            pipeline_override={
                _NODE_OPEN_INFO: _RECOVERY_OVERRIDE,
            },
        )

        last_name: str | None = None
        no_change = 0
        results: dict[str, list[int]] = {}  # 全新记录模式用

        for iteration in range(_MAX_ITERATIONS):
            if context.tasker.stopping:
                logger.info("SyncCityAbility: 收到停止信号，退出")
                break

            # ---- 3. 识别角色名（信息页 OCR → 技能页 TemplateMatch 校验） ----
            name = _get_character_name(context)
            # 调用后停留在技能页面

            if name is None:
                logger.debug("SyncCityAbility[%d]: 识别失败，跳过", iteration)
                no_change += 1
            else:
                # ---- 4. OCR 技能（已在技能页） ----
                levels = _ocr_skills(context)
                logger.info(
                    "SyncCityAbility[%d]: %s 技能0=%d 技能1=%d",
                    iteration,
                    name,
                    levels[0],
                    levels[1],
                )
                if fresh_record:
                    results[name] = levels
                else:
                    set_character_abilities(name, levels)

                if name == last_name:
                    no_change += 1
                else:
                    no_change = 0
                    last_name = name

            # ---- 5. 列表末尾检测：NextCharacter 无新角色 → 扫描剩余可见位置后结束 ----
            if no_change >= _MAX_NO_CHANGE:
                logger.info(
                    "SyncCityAbility: %d 次角色名未变化，扫描列表剩余角色",
                    no_change,
                )
                # 确保回到信息页（_get_character_name 结束时停在技能页）
                context.run_task(_NODE_OPEN_INFO)
                if fresh_record:
                    self._scan_remaining_characters(context, results)
                else:
                    self._scan_remaining_characters(context)
                logger.info("SyncCityAbility: 剩余角色扫描完成，遍历结束")
                break

            # ---- 6. 回到信息页 → 切换下一个角色 ----
            context.run_task(_NODE_OPEN_INFO)
            context.run_task(_NODE_NEXT)

        # ---- 7. 全新记录模式：清空旧数据 + 批量存入 ----
        if fresh_record and results:
            clear_all()
            for char_name, levels in results.items():
                set_character_abilities(char_name, levels)
            logger.info("SyncCityAbility: 批量存入 %d 个角色", len(results))

        logger.info("SyncCityAbility: 遍历完成，共 %d 轮", iteration + 1)
        return CustomAction.RunResult(success=True)

    # ------------------------------------------------------------------
    # 列表剩余可见角色
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_remaining_characters(
        context: Context,
        results: dict[str, list[int]] | None = None,
    ) -> None:
        """依次点击列表位置 2→3→4→5，扫描 NextCharacter 无法到达的末尾角色。

        每个位置：回到信息页 → 标准角色流程（OCR → 技能页 → TM → OCRSkill）。
        """
        for click_node in _CLICK_NODES:
            if context.tasker.stopping:
                return

            logger.info("SyncCityAbility tail: %s", click_node)
            context.run_task(click_node)
            time.sleep(0.3)

            # 点击后已在信息页，走标准角色流程
            name = _get_character_name(context)
            if name is None:
                logger.debug("SyncCityAbility tail: %s 识别失败，跳过", click_node)
                # _get_character_name 结束时在技能页，回到信息页以便下一个 click
                context.run_task(_NODE_OPEN_INFO)
                continue

            # OCR 技能（已在技能页）
            levels = _ocr_skills(context)
            logger.info(
                "SyncCityAbility tail: %s 技能0=%d 技能1=%d",
                name,
                levels[0],
                levels[1],
            )
            if results is not None:
                results[name] = levels
            else:
                set_character_abilities(name, levels)

            # 回到信息页，准备下一个 click
            context.run_task(_NODE_OPEN_INFO)
