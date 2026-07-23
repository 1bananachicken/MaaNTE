# -*- coding: utf-8 -*-
"""
角色都市技能等级管理器

每个角色固定有 2 个都市技能，记为 0 和 1。
- 技能0: 必定存在，等级范围 0~5，默认操作对象
- 技能1: 部分角色可能没有，等级范围 0~2，不存在时用 -1 表示

JSON 文件路径: <安装目录>/config/CharacterAbility_CityAbility.json
数据结构: {"角色名": [技能0等级, 技能1等级]}，技能1为 -1 表示不存在

用法:
    from custom.action.Common.CharacterAbility_CityAbility import (
        get_ability_level,
        set_ability_level,
        get_character_abilities,
        set_character_abilities,
        list_characters,
        remove_character,
        reload,
    )

    # 查询技能0（默认）
    level = get_ability_level("薄荷")
    # 查询技能1
    level = get_ability_level("薄荷", 1)

    # 写入技能0（默认）
    set_ability_level("薄荷", 3)
    # 写入技能1
    set_ability_level("薄荷", 1, 1)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "CharacterAbility_CityAbility.json"

# 技能等级范围。技能1允许 -1 表示"不存在"
_MAX_LEVEL = {0: 5, 1: 2}
_NOT_AVAILABLE = -1


def _find_project_root() -> Path:
    """从当前文件位置向上查找项目根目录（包含 config/ 目录的路径）。"""
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    # 回退：假设标准结构 agent/custom/action/Common/ → 向上 5 级到项目根
    return here.parents[4]


def _get_config_path() -> Path:
    """获取 JSON 配置文件的绝对路径。"""
    config_dir = _find_project_root() / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / _CONFIG_FILENAME


def _validate_level(ability_index: int, level: int) -> None:
    """校验技能等级是否在合法范围内。

    技能0: 0~5
    技能1: -1（不存在）或 0~2
    """
    if ability_index == 0:
        if not isinstance(level, int) or level < 0 or level > _MAX_LEVEL[0]:
            raise ValueError(
                f"技能0等级必须在 0~{_MAX_LEVEL[0]} 之间，收到: {level}"
            )
    else:
        if level == _NOT_AVAILABLE:
            return
        if not isinstance(level, int) or level < 0 or level > _MAX_LEVEL[1]:
            raise ValueError(
                f"技能1等级必须在 0~{_MAX_LEVEL[1]} 之间（或 {_NOT_AVAILABLE} 表示不存在），收到: {level}"
            )


class CharacterCityAbilityManager:
    """角色都市技能等级管理器。

    线程安全，使用可重入锁保护 JSON 数据。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config_path = _get_config_path()
        self._data: dict[str, list[int]] = {}
        self._load()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """从 JSON 文件加载数据到内存。"""
        if not self._config_path.exists():
            logger.debug("配置文件不存在，使用空数据: %s", self._config_path)
            self._data = {}
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            logger.exception("配置文件 JSON 解析失败，将重建: %s", self._config_path)
            self._data = {}
            self._save()
            return
        except OSError:
            logger.exception("读取配置文件失败: %s", self._config_path)
            self._data = {}
            return

        if not isinstance(raw, dict):
            logger.warning("配置文件顶层不是 dict，将重建: %s", self._config_path)
            self._data = {}
            self._save()
            return

        # 规范化：确保值是 [int, int]，技能1可为 -1
        cleaned: dict[str, list[int]] = {}
        for char_name, levels in raw.items():
            if not isinstance(char_name, str):
                continue
            if not isinstance(levels, list) or len(levels) != 2:
                continue
            try:
                lv0 = max(0, min(int(levels[0]), _MAX_LEVEL[0]))
                lv1_raw = int(levels[1])
                if lv1_raw == _NOT_AVAILABLE:
                    lv1 = _NOT_AVAILABLE
                else:
                    lv1 = max(0, min(lv1_raw, _MAX_LEVEL[1]))
            except (TypeError, ValueError):
                continue
            cleaned[char_name] = [lv0, lv1]

        self._data = cleaned
        logger.info("已加载角色技能配置: %d 个角色", len(self._data))

    def _save(self) -> None:
        """将内存数据持久化到 JSON 文件。"""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)
            logger.debug("角色技能配置已保存")
        except OSError:
            logger.exception("写入配置文件失败: %s", self._config_path)

    def _ensure_character(self, character_name: str) -> None:
        """确保角色存在于数据中，不存在则初始化为 [0, -1]（技能1默认不存在）。"""
        if character_name not in self._data:
            self._data[character_name] = [0, _NOT_AVAILABLE]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_ability_level(
        self, character_name: str, ability_index: int = 0
    ) -> int:
        """查询某角色的技能等级。

        Args:
            character_name: 角色名称
            ability_index:  技能编号，0 或 1，默认 0

        Returns:
            技能等级（int），未记录时技能0返回0、技能1返回-1。
        """
        if ability_index not in (0, 1):
            raise ValueError(f"技能编号只能为 0 或 1，收到: {ability_index}")

        with self._lock:
            levels = self._data.get(character_name, [0, _NOT_AVAILABLE])
            return levels[ability_index]

    def set_ability_level(
        self, character_name: str, level: int, ability_index: int = 0
    ) -> None:
        """写入 / 更新某角色的技能等级。

        Args:
            character_name: 角色名称
            level:          技能等级（技能0: 0~5，技能1: 0~2 或 -1 表示不存在）
            ability_index:  技能编号，0 或 1，默认 0
        """
        if ability_index not in (0, 1):
            raise ValueError(f"技能编号只能为 0 或 1，收到: {ability_index}")
        _validate_level(ability_index, level)

        with self._lock:
            self._ensure_character(character_name)
            self._data[character_name][ability_index] = level
            self._save()
            logger.info(
                "更新技能等级: 角色=%s 技能%d 等级=%d",
                character_name,
                ability_index,
                level,
            )

    def get_character_abilities(self, character_name: str) -> list[int]:
        """查询某角色的两个技能等级。

        Args:
            character_name: 角色名称

        Returns:
            [技能0等级, 技能1等级]，未记录时返回 [0, -1]。
        """
        with self._lock:
            return list(self._data.get(character_name, [0, _NOT_AVAILABLE]))

    def set_character_abilities(
        self, character_name: str, levels: list[int]
    ) -> None:
        """批量写入某角色的两个技能等级。

        Args:
            character_name: 角色名称
            levels:         [技能0等级, 技能1等级]
        """
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError("levels 必须是包含 2 个整数的列表，例如 [3, 1] 或 [3, -1]")
        lv0, lv1 = levels
        _validate_level(0, lv0)
        _validate_level(1, lv1)

        with self._lock:
            self._data[character_name] = [lv0, lv1]
            self._save()
            logger.info(
                "批量更新技能: 角色=%s 技能0=%d 技能1=%d",
                character_name,
                lv0,
                lv1,
            )

    def remove_character(self, character_name: str) -> bool:
        """删除某角色的全部技能记录。

        Args:
            character_name: 角色名称

        Returns:
            是否成功删除（角色不存在时返回 False）。
        """
        with self._lock:
            if character_name not in self._data:
                logger.debug("删除角色失败，不存在: %s", character_name)
                return False
            del self._data[character_name]
            self._save()
            logger.info("已删除角色: %s", character_name)
            return True

    def list_characters(self) -> list[str]:
        """列出所有已记录的角色名称。

        Returns:
            角色名称列表。
        """
        with self._lock:
            return list(self._data.keys())

    def reload(self) -> None:
        """从 JSON 文件重新加载数据，丢弃内存中的未保存修改。"""
        with self._lock:
            self._load()
            logger.info("已重新加载角色技能配置")

    def clear_all(self) -> None:
        """清空所有角色技能记录。"""
        with self._lock:
            self._data.clear()
            self._save()
            logger.info("已清空全部角色技能记录")


# ======================================================================
# 模块级单例 + 便捷函数
# ======================================================================

_manager: CharacterCityAbilityManager | None = None
_singleton_lock = threading.Lock()


def get_manager() -> CharacterCityAbilityManager:
    """获取全局 ``CharacterCityAbilityManager`` 单例。"""
    global _manager
    if _manager is None:
        with _singleton_lock:
            if _manager is None:
                _manager = CharacterCityAbilityManager()
    return _manager


def get_ability_level(character_name: str, ability_index: int = 0) -> int:
    """查询某角色的技能等级（模块级快捷函数）。

    Args:
        character_name: 角色名称
        ability_index:  技能编号，0 或 1，默认 0（技能0）
    """
    return get_manager().get_ability_level(character_name, ability_index)


def set_ability_level(
    character_name: str, level: int, ability_index: int = 0
) -> None:
    """写入 / 更新某角色的技能等级（模块级快捷函数）。

    Args:
        character_name: 角色名称
        level:          技能等级（技能0: 0~5，技能1: 0~2 或 -1 表示不存在）
        ability_index:  技能编号，0 或 1，默认 0（技能0）
    """
    get_manager().set_ability_level(character_name, level, ability_index)


def get_character_abilities(character_name: str) -> list[int]:
    """查询某角色的两个技能等级（模块级快捷函数）。

    Returns:
        [技能0等级, 技能1等级]
    """
    return get_manager().get_character_abilities(character_name)


def set_character_abilities(
    character_name: str, levels: list[int]
) -> None:
    """批量写入某角色的两个技能等级（模块级快捷函数）。

    Args:
        character_name: 角色名称
        levels:         [技能0等级, 技能1等级]
    """
    get_manager().set_character_abilities(character_name, levels)


def remove_character(character_name: str) -> bool:
    """删除某角色的全部技能记录（模块级快捷函数）。"""
    return get_manager().remove_character(character_name)


def list_characters() -> list[str]:
    """列出所有已记录的角色名称（模块级快捷函数）。"""
    return get_manager().list_characters()


def reload() -> None:
    """从 JSON 文件重新加载数据（模块级快捷函数）。"""
    get_manager().reload()


def clear_all() -> None:
    """清空所有角色技能记录（模块级快捷函数）。"""
    get_manager().clear_all()
