# -*- coding: utf-8 -*-
"""
Wrap 层 — CustomAction 增强工具箱。

提供:
    task_action   — 增强装饰器（参数解析 + bool→RunResult 归一化）
"""

try:
    from .task_action import task_action
except ImportError:
    task_action = None  # maa 包在运行时由 MaaFramework 提供

__all__ = ["task_action"]
