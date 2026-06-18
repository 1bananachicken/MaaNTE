# -*- coding: utf-8 -*-
"""
CustomAction 增强装饰器。

功能:
    1. 自动解析 custom_action_param → dataclass（类型安全）
    2. 自动重试（可配置次数）
    3. 自动将 bool 返回值转为 RunResult

用法:
    from dataclasses import dataclass
    from wrap import task_action

    @dataclass
    class MyConfig:
        count: int = 10
        mode: str = "fast"

    @task_action("my_action", MyConfig, retries=2)
    class MyAction(CustomAction):
        def run(self, context, cfg: MyConfig):   # cfg 已解析+校验
            ...
            return True   # bool 自动转 RunResult(success=True)

兼容性:
    不改变 pipeline JSON 格式，原有 custom_action_param 照常传入。
    不改变 CustomAction 继承关系，不破坏现有注册机制。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

logger = logging.getLogger("maante.wrap")

_T = TypeVar("_T")


# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------

def _parse_config(raw: Any, config_cls: type[_T] | None) -> Any:
    """将 custom_action_param 解析为 config_cls 实例或原始 dict。"""
    # 统一成 dict
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("task_action: invalid JSON param: %s", raw[:200])
            raw = {}

    if config_cls is None:
        return raw

    # 提取 dataclass 字段，缺失的用默认值
    field_map = {f.name: f for f in dataclasses.fields(config_cls)}
    kwargs = {}
    for name, field in field_map.items():
        value = raw.get(name)
        if value is not None:
            kwargs[name] = value
        elif field.default is not dataclasses.MISSING:
            kwargs[name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs[name] = field.default_factory()
        # 无默认值 → 跳过，dataclass 构造时会报错（符合预期）
    return config_cls(**kwargs)


# ------------------------------------------------------------------
# 核心装饰器
# ------------------------------------------------------------------

def task_action(
    name: str,
    config_cls: type | None = None,
    retries: int = 0,
) -> Callable[[type], type]:
    """
    CustomAction 增强装饰器。

    Args:
        name: 注册到 AgentServer 的 action 名称（同 @AgentServer.custom_action）
        config_cls: 可选 dataclass，用于自动解析 custom_action_param
        retries: 失败重试次数（只对未捕获异常生效，对返回 False 不重试）

    Example:
        @dataclass
        class FishConfig:
            count: int = 99

        @task_action("auto_fish", FishConfig, retries=2)
        class AutoFish(CustomAction):
            def run(self, context, cfg: FishConfig):
                ...
    """

    def decorator(action_cls: type) -> type:
        original = action_cls.run

        @wraps(original)
        def run(
            self: CustomAction,
            context: Context,
            argv: CustomAction.RunArg,
        ) -> CustomAction.RunResult:
            config = _parse_config(argv.custom_action_param, config_cls)

            # 重试循环
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    result = original(self, context, config)
                    return _normalize_result(result)
                except Exception as exc:
                    last_exc = exc
                    if attempt < retries:
                        delay = 0.5 * (attempt + 1)
                        logger.warning(
                            "task_action %s attempt %d/%d failed: %s, retrying in %.1fs",
                            name, attempt + 1, retries + 1, exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "task_action %s all %d attempts failed: %s",
                            name, retries + 1, exc,
                        )

            # 所有重试耗尽
            return CustomAction.RunResult(success=False)

        action_cls.run = run
        return AgentServer.custom_action(name)(action_cls)

    return decorator


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------

def _normalize_result(result: Any) -> CustomAction.RunResult:
    """统一返回值：bool → RunResult，直接透传 RunResult。"""
    if isinstance(result, CustomAction.RunResult):
        return result
    if isinstance(result, bool):
        return CustomAction.RunResult(success=result)
    # None 或其他视为失败
    logger.warning("task_action: unexpected return type %s, treating as failure", type(result))
    return CustomAction.RunResult(success=False)
