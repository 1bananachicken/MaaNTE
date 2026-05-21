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
- **文件**：按天滚动，保留 2 周，压缩归档；文件日志固定带时间戳和调用位置。

## 导入

```python
from utils.logger import logger
```

如果模块在 `custom/action/` 下且 `utils` 不在直接路径中，参考已有代码的相对导入方式。少数子模块（如 `SoundDodgeAction.py`）通过 `custom.action.Common.logger.get_logger(__name__)` 获取独立 logger，这是为了利用 loguru 的 `{name}` 字段；若不关心区分 logger 名，直接用 `from utils.logger import logger` 即可。

## 日志级别

| 级别 | 用途 | 示例 |
|------|------|------|
| `logger.info` | 关键步骤、状态变更、任务开始/结束 | `logger.info("演奏开始")` |
| `logger.debug` | 识别细节、中间计算结果 | `logger.debug("未识别到 %s", name)` |
| `logger.warning` | 可恢复异常、降级、配置缺失 | `logger.warning("鼓面模板缺失，演奏检测不可用")` |
| `logger.error` | 不可恢复错误 | `logger.error("Error: %s", e)` |
| `logger.success` | 操作成功确认（loguru） | `logger.success("任务完成")` |
| `logger.trace` | 逐帧/高频细节（loguru） | `logger.trace("候选入队: ...")` |

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

## 禁止 print

**所有用户可见的输出应使用 logger，禁止 `print()`**。Pipeline 的 focus 消息用于向用户展示任务进度，Python 代码中的 print 无法正确路由到 MXU/MFA 的日志面板。

```python
# Bad
print("=== Auto Make Coffee Action Started ===")
print(f"=== Making Coffee {count + 1}/{make_count} ===")

# Good
logger.info("冲咖啡任务开始")
logger.info("制作进度: %d/%d", count + 1, make_count)
```

特例：`auto_tetris.py` 中 `print()` 的报告消息（如 `[泯除方块] 任务开始 | 模式: 单人`）应迁移为 `context.run_action("_MAANTE_FOCUS_", pipeline_override=...)` 或 logger。

## 敏感信息

不要在日志中输出密钥、完整文件路径中包含敏感信息的用户名等。

## 审查清单

- [ ] 导入使用 `from utils.logger import logger`（或合理的相对导入）
- [ ] 日志级别合理：info 用于关键节点，debug 用于识别细节
- [ ] 使用 `%` 风格占位符，不拼接字符串、不用 f-string
- [ ] 无 `print()` 调用（应使用 logger 或 pipeline focus）
- [ ] 无高频大量日志（循环内使用 debug/trace）
- [ ] 异常信息包含足够的上下文（参数值、当前步骤）
