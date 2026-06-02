---
name: python-action-guide
description: MaaNTE Python 自定义动作（CustomAction）编写指南。覆盖 agent/custom/action/ 下的 Python 代码的架构、注册、命名、maafocus 用户消息、日志、Controller API、Pipeline 集成、错误处理等编码规范和模式参考。在编写、修改或审查 Python 自定义动作，或需要了解 agent 项目结构与 MaaFramework Python 绑定集成方式时使用。
---

# MaaNTE Python CustomAction 编写指南

## 架构定位

Python CustomAction 处理 Pipeline JSON 无法覆盖的复杂逻辑（图像算法、状态机、音频检测、MIDI 播放等）。**流程控制由 Pipeline 负责，Python 只处理难点。** 一句话：**Pipeline 管流程，Python 管难点。**

没有必要的 Python 逻辑会大大增加代码复杂度，造成下一位开发者开发调试极其困难。

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
│       ├── pinkpaw/                # 粉爪大劫案
│       ├── DatasetCollection/      # 数据集采集
│       ├── auto_make_coffee.py     # 单文件简单动作
│       ├── auto_tetris.py          # 俄罗斯方块入口
│       ├── auto_f_scroll.py        # 自动滚屏
│       ├── furniture_claim.py      # 家具收取
│       ├── realtime_task.py        # 实时任务调度
│       ├── withdraw_money_choose_item.py  # 补货取钱物品选择
│       └── __init__.py             # 所有 CustomAction 的导入与注册
└── utils/
    ├── __init__.py                 # 子模块聚合导出
    ├── logger.py                   # 日志系统（loguru/logging）
    ├── pienv.py                    # PI 环境变量
    ├── maafocus.py                 # 用户可见消息（Pipeline focus 协议）
    ├── i18n.py                     # 多语言/翻译
    ├── screen.py                   # 屏幕缩放/分辨率
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

## Common 内置动作

以下 Custom Action 位于 `agent/custom/action/Common/`，可在 Pipeline 中直接使用，无需额外编写 Python 代码。

### click_override

自定义点击。通过 `custom_action_param` 指定目标 rect，或使用当前识别结果的 box。

- 注册名：`click_override`
- 参数 `custom_action_param`：`{ "target": [x, y, w, h] }`
- 若未提供 `custom_action_param`，则使用 `argv.box`（识别结果 box）

```jsonc
{
    "action": {
        "type": "Custom",
        "param": {
            "custom_action": "click_override",
            "custom_action_param": { "target": [100, 200, 50, 50] }
        }
    }
}
```

### alt_click

Alt + 点击。先按下 Alt 键，再点击识别结果 box 位置，最后松开 Alt。

- 注册名：`alt_click`
- 无需额外参数，点击位置由识别结果的 `box` 决定

```jsonc
{
    "recognition": { "type": "TemplateMatch", "param": { "template": "xxx.png" } },
    "action": {
        "type": "Custom",
        "param": { "custom_action": "alt_click" }
    }
}
```

## Common 工具函数

`agent/custom/action/Common/utils.py` 提供常用辅助函数：

| 函数 | 说明 |
|------|------|
| `get_image(controller)` | 截图，返回 numpy array |
| `click_rect(controller, rect, delay)` | 点击指定 rect 的中心 |
| `match_template_in_region(img, region, template, min_similarity, green_mask)` | 在区域内做模板匹配，返回 `(hit, score, x, y)` |

```python
from Common.utils import get_image, click_rect, match_template_in_region

img = get_image(controller)
hit, score, x, y = match_template_in_region(img, [0, 0, 1280, 720], template, 0.8)
if hit:
    click_rect(controller, [x, y, 50, 50])
```

## CustomAction 模板

### 简单动作

```python
import json
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger
from utils.maafocus import PrintT

@AgentServer.custom_action("my_action")
class MyAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        PrintT(context, "my_action.started")

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
        logger.debug("识别结果: score=%.2f", score)

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

### RecognitionDetail 结构

`context.run_recognition()` 返回 `RecognitionDetail`（`maa.define.RecognitionDetail`），结构如下：

```python
@dataclass
class RecognitionDetail:
    reco_id: int
    name: str
    algorithm: Union[AlgorithmEnum, str]
    hit: bool                              # 是否有命中
    box: Optional[Rect]                    # 最佳命中的 box
    all_results: List[RecognitionResult]   # 全部原始结果
    filtered_results: List[RecognitionResult]  # 过滤后的结果（常用）
    best_result: Optional[RecognitionResult]   # 最佳结果
    raw_detail: Dict
    raw_image: numpy.ndarray
    draw_images: List[numpy.ndarray]
```

关键方法/属性：

```python
result = context.run_recognition("MyNode", image)

# 是否存在命中
if result and result.hit:
    # 遍历所有命中（正确的多结果方式，不是 .all）
    for r in result.filtered_results:
        # r.box  — Rect(x, y, w, h)
        # 根据算法类型，r 还有不同属性：
        #   OCRResult:        r.text (str), r.score (float)
        #   ColorMatchResult: r.count (int)
        #   TemplateMatchResult: r.score (float)

# result.box — 最佳命中的 Rect（快捷方式）
# result.box.x, result.box.y, result.box.w, result.box.h 都是 int
```

### 多结果模式

**点击所有命中 rect（如 ColorMatch connected 组件、OCR 多匹配）：**

```python
def _filtered_boxes(result):
    """返回 filtered_results 中所有命中的 Rect 列表"""
    if result is None or not result.hit:
        return []
    return [r.box for r in result.filtered_results if r.box is not None]

result = context.run_recognition("SomeColorMatch", image)
for box in _filtered_boxes(result):
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    controller.post_touch_down(cx, cy).wait()
    time.sleep(0.001)
    controller.post_touch_up().wait()
```

**遍历 OCR 多结果并提取文本：**

```python
result = context.run_recognition("SomeOCRNode", image)
for r in result.filtered_results if result else []:
    text = getattr(r, "text", "")           # OCRResult 有 .text，其他类型无
    box = r.box                              # Rect
    # 处理 text 和 box ...
```

**常见错误：**
- ~~`result.all`~~ → 不存在，应该用 `result.filtered_results`
- ~~`box.text`~~ → `Rect` 没有 `.text`，应该用 `getattr(r, "text", "")` 在结果对象上取
- ~~`ColorMatchResult.box.text`~~ → ColorMatchResult 只有 `.box`（Rect）和 `.count`（int），无 `.text`

**`box` 类型注意**：MaaFramework Python 绑定返回的 `.box` 实际是 `list [x,y,w,h]` 而非 `Rect` 对象（虽然类型标注为 `Rect`），转换时需兼容：

```python
def _box_to_rect(box):
    if isinstance(box, (list, tuple)):
        return list(box)
    return [box.x, box.y, box.w, box.h]
```

**真实 `raw_detail` 结构参考（ColorMatch）：**

```json
{
  "reco_id": 400000008,
  "algorithm": "ColorMatch",
  "box": [256, 384, 19, 18],
  "detail": {
    "all": [
      {"box": [256, 384, 19, 18], "count": 185},
      {"box": [256, 531, 19, 17], "count": 182}
    ],
    "best": {"box": [256, 384, 19, 18], "count": 185},
    "filtered": [
      {"box": [256, 384, 19, 18], "count": 185},
      {"box": [256, 531, 19, 17], "count": 182}
    ]
  }
}
```

- `result.filtered_results[i].box` → `list [x,y,w,h]`
- `result.filtered_results[i].count` → `int`（ColorMatch）/ `result.filtered_results[i].text` → `str`（OCR）
- `result.raw_detail` 包含完整 JSON-serializable dict，可作为兜底访问

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

## 用户可见消息（maafocus）

**所有面向用户的进度、状态、错误提示必须使用 `maafocus.Print()` / `PrintT()`，禁止 `print()` 和 `logger.info()`。**

两个通道职责分离：

| | logger | maafocus |
|---|---|---|
| 目标 | 开发者（调试） | 终端用户 |
| 传输 | stderr / 文件 | Pipeline `focus` 协议 → MXU |
| i18n | 不支持 | `PrintT()` 自动调 `T()` |
| 阈值 | MXU 模式下仅 WARNING+ | 全部 |


### PrintT — i18n 用户消息（推荐）

```python
from utils.maafocus import PrintT

PrintT(context, "tetris.task_started", " | ".join(parts))
PrintT(context, "coffee.making", count + 1, make_count)
PrintT(context, "rhythm.playing_started", fps, "ON", scene_lock)
```

首个参数是 i18n key（定义在 `assets/resource/locales/agent/zh_cn.json`），后续参数为 `%`-格式化值。

### Print — 原样文本

```python
from utils.maafocus import Print

Print(context, "纯文本消息，不做翻译")
```

### 模块依赖

`maafocus.py` 内部导入 `utils.i18n`，因此需确保 `i18n_init()` 已先执行（`main.py` 中已有）。

### 辅助类中无 context 的情况

辅助类（如 `_KeyScheduler`、`MaaKeyboardBridge`）不持有 `context`，内部只用 `logger.debug()` 记录调试信息。用户消息由调用方（CustomAction.run()）发送。

### 管道 focus 消息（JSON 侧）

Pipeline JSON 中的 `"focus"` 字段是纯 pipeline 层消息，不走 Python i18n：

```jsonc
"focus": { "Node.Action.Succeeded": "钓到鱼了！" }
```

其 i18n 依赖 MXU 侧支持，与 agent 的 `T()` 系统不互通。

### context.run_action

调用 Pipeline 中定义的动作节点：

```python
context.run_action("SomeClickNode")
```

带参数覆盖：

```python
context.run_action("SomeClickNode", pipeline_override={
    "SomeClickNode": {"target": [100, 200]},
})
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

**logger 用于开发者调试，用户消息走 maafocus。** 详见 [maa-logging](../maa-logging/SKILL.md)。

```python
from utils.logger import logger

logger.debug("识别结果: %s", result)        # 内部细节
logger.warning("模板缺失，功能降级")         # 开发者关注的异常
logger.error("不可恢复错误: %s", e)          # 严重错误
```

禁止 `print()`。MXU 模式下 logger 的 console_level 已降为 `"WARNING"`，因此 `logger.info()` 不再出现在用户面前。

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
- [ ] 配套文件齐全：Python 文件在 `agent/custom/action/` 下、已在 `__init__.py` 中导入并加入 `__all__`

## 参考

- Pipeline 协议规范：[MaaFramework PipelineProtocol](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/3.1-PipelineProtocol.md)
- 日志规范：[maa-logging skill](../maa-logging/SKILL.md)
- Pipeline 编写：[pipeline-guide skill](../pipeline-guide/SKILL.md)
- 项目注册示例：`agent/custom/action/__init__.py`
- Python binding：`maa.custom_action`、`maa.context` 的 MaaFramework Python 绑定
