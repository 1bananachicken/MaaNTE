---
name: python-action-guide
description: MaaNTE Python 自定义动作（CustomAction）编写指南。覆盖 agent/custom/action/ 下的 Python 代码的架构、注册、命名、日志、Controller API、Pipeline 集成、错误处理等编码规范和模式参考。在编写、修改或审查 Python 自定义动作，或需要了解 agent 项目结构与 MaaFramework Python 绑定集成方式时使用。
---

# MaaNTE Python CustomAction 编写指南

## 架构定位

Python CustomAction 处理 Pipeline JSON 无法覆盖的复杂逻辑（图像算法、状态机、音频检测、MIDI 播放等）。禁止在 Python 中编写大规模业务流程——流程控制由 Pipeline JSON 负责。

所有坐标与图像以 **720p (1280×720)** 为基准。

## 目录结构

```
agent/
├── main.py                         # 入口：venv、依赖安装、AgentServer 启动
├── custom/
│   ├── __init__.py                 # from .action import *
│   └── action/
│       ├── __init__.py             # 所有 CustomAction 的 import 与 __all__ 注册
│       ├── Common/                 # 通用工具（click、alt_click、logger、utils）
│       │   ├── __init__.py
│       │   ├── click.py
│       │   ├── alt_click.py
│       │   ├── logger.py
│       │   └── utils.py            # get_image、click_rect、match_template_in_region
│       ├── AutoFish/               # 钓鱼业务（按功能聚合）
│       ├── Tetris/                 # 俄罗斯方块业务
│       │   ├── feats/              # 功能实现
│       │   └── utils/              # 工具类（棋盘、场景检测）
│       ├── rhythm/                 # 节奏任务
│       │   ├── feats/
│       │   └── utils/
│       ├── SoundTrigger/           # 音频触发业务
│       ├── Movement/               # 移动控制
│       ├── auto_piano/             # 自动钢琴
│       ├── auto_make_coffee.py     # 单文件简单动作
│       ├── auto_tetris.py          # 俄罗斯方块入口
│       ├── furniture_claim.py      # 家具收取
│       └── realtime_task.py        # 实时任务调度
└── utils/
    ├── logger.py                   # 日志系统（loguru/logging）
    ├── pienv.py                    # PI 环境变量
    ├── time.py                     # 时间工具
    └── win32_process.py            # Win32 窗口查找
```

## 注册机制

### 装饰器注册

每个 CustomAction 类使用 `@AgentServer.custom_action("name")` 注册，名称必须与 Pipeline JSON 中 `custom_action` 的值一致：

```python
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

@AgentServer.custom_action("auto_make_coffee")
class AutoMakeCoffee(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        ...
```

Pipeline JSON 中对应：

```jsonc
{
    "action": {
        "type": "Custom",
        "param": {
            "custom_action": "auto_make_coffee",
            "custom_action_param": { "count": 10 }
        }
    }
}
```

### `__init__.py` 注册

所有 CustomAction 必须在 `agent/custom/action/__init__.py` 中导入并加入 `__all__`：

```python
from .auto_make_coffee import *
from .Tetris import *

__all__ = [
    "AutoMakeCoffee",
    "AutoTetris",
    "TetrisResetContext",
    "TetrisCheckVitalityAction",
]
```

遗漏导入 = 动作不生效。

## 命名

- **类名**：PascalCase，语义化。如 `AutoMakeCoffee`、`TetrisGamePlayer`、`SoundDodgeAction`。
- **注册名**：snake_case，与 Pipeline JSON 一致。如 `"auto_make_coffee"`、`"auto_tetris"`。
- **文件名**：snake_case。单文件简单动作直接用 `.py` 文件；复杂模块用目录 + `__init__.py`。
- **内部函数/变量**：snake_case，模块内部使用的加前导下划线 `_`。

## CustomAction 模板

### 简单动作

```python
import json
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger

@AgentServer.custom_action("my_action")
class MyAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        logger.info("MyAction 开始")

        params = {}
        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
            except (json.JSONDecodeError, TypeError):
                pass

        controller = context.tasker.controller

        # 截图
        controller.post_screencap().wait()
        img = controller.cached_image

        # 检查停止信号
        if context.tasker.stopping:
            return CustomAction.RunResult(success=False)

        # 业务逻辑 ...

        return CustomAction.RunResult(success=True)
```

### 带状态管理的动作

对于需要跨多次调用保持状态的场景（如连打计数），使用模块级全局变量，并通过独立的 reset action 重置：

```python
_round_count = 0
_task_config = {}

@AgentServer.custom_action("my_reset")
class MyResetContext(CustomAction):
    def run(self, context, argv) -> CustomAction.RunResult:
        global _round_count, _task_config
        _round_count = 0
        _task_config = {}
        return CustomAction.RunResult(success=True)

@AgentServer.custom_action("my_action")
class MyAction(CustomAction):
    def run(self, context, argv) -> CustomAction.RunResult:
        global _round_count
        _round_count += 1
        ...
```

## Controller API

通过 `context.tasker.controller` 访问，常用操作：

```python
controller = context.tasker.controller

# 截图
controller.post_screencap().wait()
img = controller.cached_image  # numpy array (BGR or BGRA)

# 点击（绝对坐标，720p 基准）
controller.post_touch_down(x, y).wait()
time.sleep(0.005)  # 短暂按下
controller.post_touch_up().wait()

# 按键（虚拟键码）
controller.post_key_down(0x46).wait()  # F 键
time.sleep(0.1)
controller.post_key_up(0x46).wait()

# 组合键：先 Alt 再点
controller.post_key_down(0x12).wait()  # VK_MENU (Alt)
# ... 执行点击 ...
controller.post_key_up(0x12).wait()
```

常用虚拟键码：

| 键 | 码 |
|----|----|
| Esc | 27 |
| F | 70 |
| E | 69 |
| Q | 81 |
| D | 0x44 |
| F | 0x46 |
| J | 0x4A |
| K | 0x4B |

## Pipeline 集成

### context.run_recognition

调用已定义的 Pipeline 识别节点：

```python
# 基本调用
result = context.run_recognition("FurnitureOcrRec", image)

# 带参数覆盖（动态 expected 文本、自定义 roi 等）
result = context.run_recognition(
    "FurnitureOcrRec",
    image,
    pipeline_override={
        "FurnitureOcrRec": {
            "recognition": {"param": {"expected": "仓鼠球"}}
        }
    },
)
if result and result.box:
    x, y, w, h = result.box.x, result.box.y, result.box.w, result.box.h
```

### context.run_recognition_direct

在 Python 中直接调用 MaaFramework 识别算法（无需 Pipeline 节点定义）：

```python
from maa.pipeline import JRecognitionType, JOCR

detail = context.run_recognition_direct(
    JRecognitionType.OCR, JOCR(roi=[x, y, w, h]), frame
)
if detail and detail.hit and detail.best_result:
    text = detail.best_result.text
```

### context.run_action

调用 Pipeline 中定义的动作节点：

```python
context.run_action(
    "_MAANTE_FOCUS_",
    pipeline_override={
        "_MAANTE_FOCUS_": {
            "focus": {"Node.Action.Starting": "消息内容"},
            "action": "DoNothing",
            "pre_delay": 0,
            "post_delay": 0,
        }
    },
)
```

### context.run_task

启动子任务链：

```python
context.run_task("FurnitureClaim", pipeline_override={...})
```

### context.override_next

动态修改节点的 `next` 列表，用于运行时流程控制：

```python
context.override_next("RhythmRepeatCheck", ["RhythmExit"])
```

## 错误处理

- `run()` 返回 `CustomAction.RunResult(success=True/False)` 表示动作成功/失败。
- 失败时 MaaFramework 会走 Pipeline 中该节点的 `on_error` 分支。
- 异常应捕获并记录，避免静默吞掉：

```python
try:
    params = json.loads(argv.custom_action_param)
except (json.JSONDecodeError, ValueError, TypeError) as e:
    logger.warning("参数解析失败: %r, error: %s", argv.custom_action_param, e)
    return CustomAction.RunResult(success=False)
```

- 停止信号检查：在长循环中定期检查 `context.tasker.stopping`，及时退出：

```python
while True:
    if context.tasker.stopping:
        return CustomAction.RunResult(success=False)
    # ... 循环体 ...
```

## 日志

使用 `utils.logger`，禁止 `print()`。详见 [maa-logging](../maa-logging/SKILL.md)。

```python
from utils.logger import logger

logger.info("任务开始 | 参数=%s", params)
logger.debug("识别结果: %s", result)
logger.warning("模板缺失，功能降级")
logger.error("不可恢复错误: %s", e)
```

## 资源路径

模板图片等资源文件放在 `assets/resource/base/image/` 下。在代码中引用：

```python
from pathlib import Path

base = Path(__file__).parents[4] / "assets" / "resource" / "base"
if not base.exists():
    base = Path(__file__).parents[4] / "resource" / "base"

image_dir = base / "sounds" / "dodge.wav"
```

开发模式下（`interface.json` 版本为 `DEBUG`），cwd 切换到 `assets/`，使用 `resource/base/` 路径。

## 坐标系统

- 所有坐标以 **1280×720** 为基准。
- `controller.post_screencap()` 返回的图片可能不是 720p（取决于实际窗口分辨率），但坐标逻辑仍按 720p 编写。
- Pipeline 中 `roi`、`target` 等同样基于 720p。

## 审查清单

- [ ] 注册名与 Pipeline `custom_action` 值一致
- [ ] 已在 `agent/custom/action/__init__.py` 中导入并加入 `__all__`
- [ ] 使用 `utils.logger`，无 `print()` 调用
- [ ] 日志使用 `%` 风格占位符，不拼接字符串
- [ ] 无大规模流程代码——流程由 Pipeline 驱动
- [ ] 坐标/图像基于 720p
- [ ] 长循环中有 `context.tasker.stopping` 检查
- [ ] 异常有合理的错误处理和返回值
- [ ] 模块级状态变量有对应的 reset action 或清理逻辑
- [ ] 重复逻辑已考虑抽取为 `Common/utils.py` 中的共用函数

## 参考

- Pipeline 协议规范：[MaaFramework PipelineProtocol](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/3.1-PipelineProtocol.md)
- 日志规范：[maa-logging skill](../maa-logging/SKILL.md)
- Pipeline 编写：[pipeline-guide skill](../pipeline-guide/SKILL.md)
- 项目注册示例：`agent/custom/action/__init__.py`
- Python binding：`maa.custom_action`、`maa.context` 的 MaaFramework Python 绑定
