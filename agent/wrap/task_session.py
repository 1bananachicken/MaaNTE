# -*- coding: utf-8 -*-
"""
任务级状态容器 — 按状态类型自动隔离，替代模块级全局变量。

隔离机制:
    key = (id(tasker), state 类型)
    不同 action 使用不同的 dataclass → 自动隔离，不会串数据。

用法:
    from dataclasses import dataclass
    from wrap import TaskSession

    @dataclass
    class TetrisState:
        round: int = 0
        target: int = 0
        done: bool = False

    class AutoTetris(CustomAction):
        def run(self, context, cfg):
            s = TaskSession.of(context, TetrisState)
            s.round += 1   # 类型安全，IDE 补全
            ...

    class AutoFish(CustomAction):
        def run(self, context, cfg):
            s = TaskSession.of(context, FishState)  # TetrisState 不会污染这里
            ...

生命周期:
    创建 — 首次 TaskSession.of(context, SomeState) 时自动创建
    回收 — Tasker 被 GC 时自动回收（或手动 cleanup）
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from maa.context import Context

_T = TypeVar("_T")


class TaskSession:
    """按状态类型隔离的会话容器，线程安全。"""

    # key = (tasker_id, state_class) → state_instance
    _store: dict[tuple[int, type], object] = {}
    _lock = threading.Lock()

    @classmethod
    def of(cls, context: "Context", state_cls: type[_T]) -> _T:
        """获取或创建当前 tasker + state 类型的实例。

        Args:
            context: MaaFramework Context（必须有 .tasker 属性）
            state_cls: 任意 dataclass，不同类自动隔离

        Returns:
            state_cls 的实例，跨同类型 action 调用共享

        Example:
            s = TaskSession.of(context, TetrisState)
            s.round += 1
        """
        tid = id(context.tasker)  # type: ignore[union-attr]
        key = (tid, state_cls)
        with cls._lock:
            if key not in cls._store:
                cls._store[key] = state_cls()
            return cls._store[key]  # type: ignore[return-value]

    @classmethod
    def cleanup(cls, context: "Context", state_cls: type | None = None) -> None:
        """清理 session。

        Args:
            context: 要清理的 Context
            state_cls: 指定类型则只清理该类型，None 则清空该 tasker 所有
        """
        tid = id(context.tasker)  # type: ignore[union-attr]
        with cls._lock:
            if state_cls is not None:
                cls._store.pop((tid, state_cls), None)
            else:
                keys = [k for k in cls._store if k[0] == tid]
                for k in keys:
                    del cls._store[k]

    @classmethod
    def _size(cls) -> int:
        """调试用。"""
        with cls._lock:
            return len(cls._store)
