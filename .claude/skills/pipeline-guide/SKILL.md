---
name: pipeline-guide
description: MaaNTE Pipeline JSON 编写指南。基于 MaaFramework Pipeline 协议，提供节点命名、识别算法、动作类型、流程控制、可复用节点等编码规范与模式参考。在编写、修改或审查 Pipeline JSON、设计节点流程、使用 TemplateMatch/OCR/Custom 识别或 Click/Swipe 动作时使用。
---

# MaaNTE Pipeline 编写指南

## 核心原则

1. **状态驱动**：遵循"识别 → 操作 → 识别"循环。每次操作必须基于识别结果，禁止假设操作后画面状态。
2. **高命中率**：扩充 `next` 列表，覆盖当前操作后所有可能画面，力争一次截图命中。
3. **避免硬延迟**：尽量不用 `pre_delay` / `post_delay` / `timeout`，用中间识别节点或 `pre_wait_freezes` / `post_wait_freezes` 替代；当确实不需要延迟时，要在节点上显式将 `rate_limit` / `pre_delay` / `post_delay` 设为 0（协议默认 `rate_limit=1000ms`、`pre_delay/post_delay=200ms`，省略字段会引入隐式等待）。
4. **720p 基准**：所有坐标、ROI、图片必须基于 **1280×720**。

## 目录结构

```
assets/resource/base/pipeline/
├── Common/            # 通用可复用节点（按钮、场景跳转等）
├── Fish/              # 钓鱼任务流水线
├── Furniture/         # 家具收取流水线
├── Interface/         # 场景管理器接口节点
├── Rhythm/            # 节奏任务流水线
├── SceneManager/      # 场景管理内部节点
├── SoundDodge/        # 闪避反击流水线
├── Tetris/            # 俄罗斯方块流水线
├── MakeCoffee.json    # 冲咖啡任务
├── ClaimRewards.json  # 领取奖励
├── WithdrawMoney.json # 补货取钱
└── ...
```

## 节点命名

- 使用 **PascalCase**，同一任务内节点以任务名/模块名为前缀。
- 示例：`FishNewEntrance`、`TetrisEntrance`、`MakeCoffeeStart`。

## Pipeline v2 格式（推荐）

MaaNTE 使用 v2 格式，recognition 和 action 放入二级字典：

```jsonc
{
    "MyNode": {
        "recognition": {
            "type": "TemplateMatch",
            "param": {
                "template": "MyTask/button.png",
                "roi": [100, 200, 300, 100],
                "threshold": 0.7
            }
        },
        "action": {
            "type": "Click"
        },
        "next": ["NextNode"]
    }
}
```

## 常用识别算法

### TemplateMatch（找图）

```jsonc
"recognition": {
    "type": "TemplateMatch",
    "param": {
        "template": "path/to/image.png",  // 相对 resource/base/image 文件夹
        "roi": [x, y, w, h],              // 720p 坐标，缩小搜索范围
        "threshold": 0.7                   // 默认 0.7，按需调整
    }
}
```

- 图片必须从无损原图裁剪并缩放到 720p。
- `green_mask: true` 可遮蔽不参与匹配的区域（用 RGB(0,255,0) 涂色）。

### OCR（文字识别）

```jsonc
"recognition": {
    "type": "OCR",
    "param": {
        "roi": [x, y, w, h],
        "expected": ["完整文本"]
    }
}
```

- `expected` 写完整文本，不要写片段。

### And / Or（组合识别）

```jsonc
// And：全部子识别都成功才算命中
"recognition": {
    "type": "And",
    "param": {
        "all_of": ["NodeA", "NodeB"],
        "box_index": 0
    }
}

// Or：任一子识别成功即命中
"recognition": {
    "type": "Or",
    "param": {
        "any_of": ["NodeA", "NodeB"]
    }
}
```

### Custom（自定义识别/动作）

调用 Python Agent 注册的自定义识别器或动作：

```jsonc
// 自定义动作
"action": {
    "type": "Custom",
    "param": {
        "custom_action": "auto_make_coffee",
        "custom_action_param": {
            "count": 10
        }
    }
}

// 自定义识别
"recognition": {
    "type": "Custom",
    "param": {
        "custom_recognition": "MyRecognition",
        "custom_recognition_param": {}
    }
}
```

## 常用动作类型

| 动作       | 用途       | 关键字段                      |
| ---------- | ---------- | ----------------------------- |
| `Click`    | 点击       | `target`, `target_offset`     |
| `Swipe`    | 滑动       | `begin`, `end`, `duration`    |
| `ClickKey` | 按键       | `key`（虚拟键码）              |
| `InputText`| 输入文本   | `input_text`                  |
| `StopTask` | 停止当前任务 | 无                          |
| `Custom`   | 自定义动作 | `custom_action`, `custom_action_param` |
| `DoNothing`| 不执行     | 无                            |

`target` 支持：`true`（当前识别结果）、节点名字符串、`[x, y]`、`[x, y, w, h]`。

## 流程控制

### next 列表

按序识别，首个命中的节点执行其 action 后成为当前节点。`next` 为空或全部超时则任务结束。

### Node Attributes（节点属性）

**`[JumpBack]`**：命中后执行完该节点链，自动返回父节点继续识别 next。适用于处理弹窗、加载等中断场景。

```jsonc
"next": [
    "BusinessNode",
    "[JumpBack]HandlePopup",
    "[JumpBack]WaitLoading"
]
```

**`[Anchor]`**：动态引用锚点，运行时解析为最后设置该锚点的节点。

```jsonc
"FishNewEntrance": {
    "anchor": "FishNewRestart"
}
// ...
"next": ["[Anchor]FishNewRestart"]
```

### 等待画面稳定

用 `pre_wait_freezes` / `post_wait_freezes` 等待画面静止：

```jsonc
"post_wait_freezes": {
    "time": 200,
    "target": [0, 0, 0, 0]  // 全屏
}
```

### max_hit

限制节点最大命中次数：

```jsonc
"max_hit": 3
```

### focus（用户提示消息）

用于向用户展示任务进度或状态。

**Pipeline JSON `focus`**（纯 pipeline 层消息，不走 Python i18n）：

```jsonc
"focus": {
    "Node.Action.Succeeded": "钓到鱼了！"
}
```

**Python 侧用户消息**应使用 `maafocus.Print()` / `PrintT()`（支持 i18n），详见 [python-action-guide](../python-action-guide/SKILL.md#用户可见消息maafocus)：

```python
from utils.maafocus import PrintT
PrintT(context, "tetris.task_done")
```

两者通过同一条 MaaFramework focus 协议到达 MXU，但来源不同：
- Pipeline JSON `focus` — 静态字符串，适合简单状态通知
- `maafocus.Print()` — 支持 i18n 和动态参数，适合 Python 侧的用户消息

### on_error

识别超时或动作失败时执行的节点列表：

```jsonc
"on_error": ["TetrisExit"]
```

## 典型模式

### 带中断处理的任务入口

```jsonc
{
    "MyTaskEntry": {
        "next": [
            "MyTaskMainStep",
            "[JumpBack]HandlePopup",
            "[JumpBack]WaitLoading"
        ]
    }
}
```

### 确认后验证画面变化

```jsonc
{
    "ClickConfirm": {
        "recognition": { "type": "TemplateMatch", "param": { "template": "confirm.png" } },
        "action": { "type": "Click" },
        "post_wait_freezes": { "time": 200, "target": [0, 0, 0, 0] },
        "next": ["VerifyNextScreen", "[JumpBack]ClickConfirm"]
    }
}
```

## 审查清单

- [ ] 字段名拼写正确、类型合法（核对 Pipeline 协议）
- [ ] 无不必要的 `pre_delay` / `post_delay` / `timeout`
- [ ] `next` 列表覆盖所有可能画面，含弹窗/加载/异常
- [ ] 每次点击后有识别验证，不假设操作后状态
- [ ] ROI / target 坐标基于 1280×720
- [ ] 自定义动作名与 Python `@AgentServer.custom_action("name")` 一致
- [ ] 自定义识别/动作参数与 Python 代码中解析的参数名一致
- [ ] 用户消息优先用 `maafocus.PrintT()`（Python 侧），简单通知用 JSON `focus`（pipeline 侧）
- [ ] OCR `expected` 写完整文本
- [ ] 使用 `post_wait_freezes` 或中间节点避免重复点击

## 参考

- Pipeline 协议完整规范：[MaaFramework PipelineProtocol](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/3.1-PipelineProtocol.md)
- Python 自定义动作开发：`agent/custom/action/` 目录下的实现
- 节点测试：`docs/zh_cn/developers/node-testing.md`
