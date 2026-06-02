# Pipeline JSON 编写指南

## 核心原则

1. **状态驱动**：遵循"识别 → 操作 → 识别"循环。每次操作必须基于识别结果，禁止假设操作后画面状态。
2. **高命中率**：扩充 `next` 列表，覆盖当前操作后所有可能画面，力争一次截图命中。
3. **避免硬延迟**：尽量不用 `pre_delay` / `post_delay` / `timeout`，用中间识别节点或 `pre_wait_freezes` / `post_wait_freezes` 替代。当确实不需要延迟时，显式将 `rate_limit` / `pre_delay` / `post_delay` 设为 0（协议默认 `rate_limit=1000ms`、`pre_delay/post_delay=200ms`，省略字段会引入隐式等待）。
4. **720p 基准**：所有坐标、ROI、图片必须基于 **1280×720**。

## 节点命名

使用 **PascalCase**，同一任务内以任务名/模块名为前缀。示例：`FishNewEntrance`、`TetrisEntrance`、`TouchDetect`。私有节点以 `__` 开头：`__ScenePrivateWorldEnterBag`。

## Pipeline v2 格式

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
        "action": { "type": "Click" },
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
        "template": "path/to/image.png",  // 相对 resource/base/image
        "roi": [x, y, w, h],
        "threshold": 0.7,
        "green_mask": true                // 可选：跳过 RGB(0,255,0) 区域
    }
}
```

- 图片必须从无损原图裁剪并缩放到 720p
- `order_by: "Score"` 可按置信度排序

### OCR（文字识别）

```jsonc
"recognition": {
    "type": "OCR",
    "param": {
        "roi": [x, y, w, h],
        "expected": ["完整文本"],
        "threshold": 0.5,                // 默认 0.3
        "only_rec": false                // true 时只识别不动作
    }
}
```

- `expected` 写完整文本。需要跳过 i18n 同步时加 `// @i18n-skip` 标记
- `only_rec: true` 用于 Python 侧处理多结果

### ColorMatch（颜色匹配）

```jsonc
"recognition": {
    "type": "ColorMatch",
    "param": {
        "roi": [x, y, w, h],
        "method": 40,                    // RGB 距离
        "lower": [r, g, b],
        "upper": [r, g, b],
        "count": 20,                     // 最小像素数
        "connected": true                // 返回连通组件
    }
}
```

### DirectHit（直接命中）

不做识别，直接执行动作。适用于前一节点已确认状态、不需要重复识别的场景：

```jsonc
"recognition": { "type": "DirectHit" },
"action": { "type": "ClickKey", "param": { "key": 70 } }
```

### And / Or（组合识别）

```jsonc
// And：全部子识别都成功才算命中
"recognition": { "type": "And", "param": { "all_of": ["NodeA", "NodeB"] } }

// Or：任一子识别成功即命中
"recognition": { "type": "Or", "param": { "any_of": ["NodeA", "NodeB"] } }
```

### Custom（自定义动作）

```jsonc
"action": {
    "type": "Custom",
    "param": {
        "custom_action": "auto_make_coffee",
        "custom_action_param": { "count": 10 }
    }
}
```

`custom_action` 的值必须与 Python 中 `@AgentServer.custom_action("name")` 一致。

## 常用动作类型

| 动作 | 关键字段 |
|------|----------|
| `Click` | `target`（`true`/节点名/`[x,y]`/`[x,y,w,h]`） |
| `LongPress` | `target`, `duration` |
| `Swipe` | `begin`, `end`, `duration`, `end_hold` |
| `ClickKey` | `key`（虚拟键码） |
| `Custom` | `custom_action`, `custom_action_param` |
| `DoNothing` | 不执行动作 |
| `StopTask` | 停止当前任务 |

## 流程控制

### next 列表

按序识别，首个命中的节点执行后成为当前节点。全部超时则任务结束（默认超时 20 秒）。

### 节点属性

**`[JumpBack]`** — 命中后执行完该节点链，自动返回父节点继续识别 next。适用于处理弹窗、加载等中断场景。

```jsonc
"next": [
    "BusinessNode",
    "[JumpBack]SceneAnyEnterWorld",
    "[JumpBack]SceneClickBlankToExit"
]
```

**`[Anchor]`** — 动态引用锚点，运行时解析为最后设置该锚点的节点。

```jsonc
"FishNewEntrance": { "anchor": "FishNewRestart" },
// ...
"next": ["[Anchor]FishNewRestart"]
```

### max_hit

限制节点最大命中次数：

```jsonc
"max_hit": 100   // 实现可控循环
```

### focus（用户消息）

```jsonc
"focus": {
    "Node.Action.Succeeded": "$task_fish_new_focus_got_fish"
}
```

推荐使用 `$i18n_key` 格式，五种语言文件中定义翻译。详见 [i18n](./i18n.md)。

### on_error

```jsonc
"on_error": ["TetrisExit"]
```

### 等待画面稳定

```jsonc
"post_wait_freezes": {
    "time": 200,
    "target": [0, 0, 0, 0]，
    "threshold": 0.95,
    "timeout": 20000
}
```

## 典型模式

### 带中断处理的任务入口

```jsonc
{
    "MyTaskEntry": {
        "next": [
            "MyTaskMainStep",
            "[JumpBack]HandlePopup",
            "[JumpBack]SceneAnyEnterWorld"
        ]
    }
}
```

### switch 控制节点启停

```jsonc
// 任务配置
"MyFeature": {
    "type": "switch",
    "cases": {
        "Yes": { "pipeline_override": { "MyNode": { "enabled": true } } },
        "No":  { "pipeline_override": { "MyNode": { "enabled": false } } }
    }
}
```

### 子模块目录

复杂任务建议拆分为独立目录：

```
pipeline/WithdrawMoney/
├── WithdrawMoney.json       # 主流程节点
└── WithdrawMoneyStatus.json # 辅助识别/动作节点
```
