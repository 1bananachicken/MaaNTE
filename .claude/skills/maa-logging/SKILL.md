---
name: maa-logging
description: MaaNTE Python 日志规范。覆盖 agent/utils/logger.py 的 loguru/标准 logging 使用方式、日志级别选择、格式化与最佳实践。在编写或修改 agent/ 下的 Python 代码、添加日志输出、或审查日志质量时使用。
---

# MaaNTE Python 日志规范

## 架构

MaaNTE 的日志系统位于 `agent/utils/logger.py`，提供统一的 `logger` 对象：

- **优先 loguru**：若已安装 `loguru`，则使用 loguru 的彩色输出与结构化日志。
- **回退标准 logging**：若未安装 loguru，使用标准库 `logging` + `TimedRotatingFileHandler`。
- **控制台**：根据客户端类型（MXU / MFAAvalonia / 其他）自动适配输出格式（HTML 彩色 / 短标签 / ANSI 颜色）。
- **MXU 模式**：console_level 自动降为 `"WARNING"`，用户只看到 WARNING/ERROR；INFO/DEBUG 仅入文件日志。
- **文件**：按天滚动，保留 2 周，压缩归档；文件日志固定带时间戳和调用位置。

## logger 与 maafocus 的职责分离

```
logger          → 开发者日志（内部状态、算法细节、调试信息）
maafocus        → 用户可见消息（任务进度、状态变更、错误提示）
```

- `logger.debug/info` 在 MXU 模式下**不会**出现在用户面前（console_level = WARNING）。
- 所有面向用户的消息必须使用 `maafocus.Print(context, msg)` 或 `maafocus.PrintT(context, key, *args)`。
- `maafocus.PrintT()` 自动通过 `utils.i18n.T()` 查翻译，支持多语言。
- 详见 [python-action-guide](../python-action-guide/SKILL.md#用户可见消息maafocus)。

## 导入

```python
from utils.logger import logger
```

如果模块在 `custom/action/` 下且 `utils` 不在直接路径中，参考已有代码的相对导入方式。少数子模块（如 `SoundDodgeAction.py`）通过 `custom.action.Common.logger.get_logger(__name__)` 获取独立 logger，这是为了利用 loguru 的 `{name}` 字段；若不关心区分 logger 名，直接用 `from utils.logger import logger` 即可。

## 日志级别

| 级别 | 用途 | 示例 |
|------|------|------|
| `logger.info` | 内部关键步骤、启动初始化 | `logger.info("已加载演奏配置文件: %s", path)` |
| `logger.debug` | 识别细节、中间计算结果 | `logger.debug("未识别到 %s", name)` |
| `logger.warning` | 可恢复异常、降级、配置缺失 | `logger.warning("鼓面模板缺失，演奏检测不可用")` |
| `logger.error` | 不可恢复错误 | `logger.error("Error: %s", e)` |
| `logger.success` | 操作成功确认（loguru） | `logger.success("初始化完成")` |
| `logger.trace` | 逐帧/高频细节（loguru） | `logger.trace("候选入队: ...")` |

**注意**：MXU 模式下 console_level 为 `"WARNING"`，`logger.info` 不会出现在用户面前。用户可见的消息请用 `maafocus.PrintT()`。

## 格式化

使用 `%s` / `%d` 等 `%` 风格占位符（loguru 风格），不要手动拼接字符串：

```python
# Good
logger.info("演奏开始 | FPS=%d | 鼓面检测=%s", target_fps, drum_available)
logger.debug("未识别到 %s", name)
logger.warning("候选丢弃: lane=%s target=%.3f reason=min_interval", lane, target_time)

# Bad
logger.info("演奏开始 | FPS=" + str(target_fps))
logger.info(f"演奏开始 | FPS={target_fps}")
```

## 禁止 print / logger.info 用户消息

**用户可见的消息必须通过 `maafocus.Print()` / `PrintT()` 发送，禁止 `print()` 和 `logger.info()`。**

```python
# Bad — 用户看不到（MXU 模式下 logger.info 被过滤）
logger.info("冲咖啡任务开始")
print("=== Auto Make Coffee Action Started ===")

# Good — 用户能看的用 maafocus
PrintT(context, "coffee.started")
PrintT(context, "coffee.making", count + 1, make_count)

# Good — 开发者调试用 logger
logger.debug("识别分数: prob=%.2f", prob)
logger.warning("模板缺失，功能降级")
logger.error("不可恢复错误: %s", e)
```

## 敏感信息

不要在日志中输出密钥、完整文件路径中包含敏感信息的用户名等。

## 审查清单

- [ ] 导入使用 `from utils.logger import logger`（或合理的相对导入）
- [ ] 日志级别合理：info 用于关键节点，debug 用于识别细节
- [ ] 使用 `%` 风格占位符，不拼接字符串、不用 f-string
- [ ] 无 `print()` 调用（应使用 logger 或 pipeline focus）
- [ ] 无高频大量日志（循环内使用 debug/trace）
- [ ] 异常信息包含足够的上下文（参数值、当前步骤）
