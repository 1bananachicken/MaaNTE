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
├── Common/              # 通用可复用节点（按钮、场景跳转等）
├── DatasetCollection/   # 自动驾驶数据集采集
├── Fish/                # 钓鱼任务流水线
├── Furniture/           # 家具收取流水线
├── Interface/           # 场景管理器接口节点
├── Movement/            # 移动测试
├── PinkPawHeist/        # 粉爪大劫案核心模块
├── Rhythm/              # 节奏任务流水线
├── SceneManager/        # 场景管理内部节点
├── SoundDodge/          # 闪避反击流水线
├── Tetris/              # 俄罗斯方块流水线
├── WithdrawMoney/       # 补货取钱子模块
├── realtime_assist/     # 实时辅助子模块
├── AutoFScroll.json     # 自动滚屏
├── auto_piano.json      # 自动钢琴
├── ClaimRewards.json    # 领取奖励
├── FountainCheckin.json # 喷泉签到
├── MakeCoffee.json      # 冲咖啡任务
├── PinkPawHeist.json    # 粉爪大劫案入口
├── realtime_assist.json # 实时辅助入口
├── TestMovement.json    # 移动测试
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

## SceneManager 与场景跳转

MaaNTE 的场景管理器提供"从任意界面自动导航到目标场景"的能力，分两层架构：

- **公共接口**（`Interface/Scene/`）— Pipeline 任务使用的节点，名称不含 `__ScenePrivate`
- **私有实现**（`SceneManager/`）— 内部节点，以 `__ScenePrivate` 开头，**禁止在 Pipeline 中直接引用**

### 常用场景跳转接口

| 接口 | 说明 |
|------|------|
| `SceneAnyEnterWorld` | 从任意界面返回大世界 |
| `SceneLoading` | 等待加载界面结束 |
| `SceneClickBlankToExit` | 点击空白区域关闭弹窗 |
| `SceneAnyEnterEscMenu` | 进入 Esc 菜单 |
| `SceneAnyEnterBagMenu` | 进入背包 |
| `SceneAnyEnterBattlePassMenu` | 进入环期赏令 |
| `SceneAnyEnterCharactersMenu` | 进入角色界面 |
| `SceneAnyEnterCityTycoonsMenu` | 进入都市大亨 |
| `SceneAnyEnterEventsMenu` | 进入活动菜单 |
| `SceneAnyEnterExplorationGuideMenu` | 进入探索指南 |
| `SceneAnyEnterHethereauHobbiesMenu` | 进入都市闲趣 |

### InScene 场景识别

InScene 节点（`Interface/Scene/Status.json`）用于判断当前画面所在场景，配合 SceneManager 实现自动导航：

```jsonc
{
    "MyTaskEntry": {
        "next": [
            "MyTaskCheckInWorld",
            "[JumpBack]SceneAnyEnterWorld"   // 不在大世界时自动跳转
        ]
    },
    "MyTaskCheckInWorld": {
        "recognition": { "type": "And", "param": { "all_of": ["InWorld"] } },
        "next": ["MyTaskNextStep"]
    }
}
```

> **重要**：跳转后必须有 InScene 检查节点确认已在目标场景，避免反复跳转导致死循环。

常用 InScene 节点：`InWorld`、`InEscMenu`、`InBagMenu`、`InCityTycoonMenu`、`InExplorationGuideMenu`、`InBattlePassMenu`、`InCharactersMenu`。完整列表见 `Interface/Scene/Status.json`。

### 新增场景接口

需要新增场景跳转时：
1. 在 `SceneManager/` 下添加 `__ScenePrivate*` 私有节点处理实际导航
2. 在 `Interface/Scene/` 中添加公共接口节点
3. 在 `Status.json` 中添加状态检测节点（如 `InNewMenu`），使用 OCR 识别页面标题文字
4. 在 SceneManager 的 `__ScenePrivateAnyEnterXxxSuccess` 中引用新节点，接入万能跳转链

## 关键编码规范

### 禁止硬延迟

尽量不用 `pre_delay` / `post_delay` / `timeout`，用中间识别节点或 `pre_wait_freezes` / `post_wait_freezes` 替代。当确实不需要延迟时，显式将 `rate_limit` / `pre_delay` / `post_delay` 设为 0（协议默认 `rate_limit=1000ms`、`pre_delay/post_delay=200ms`）。

**不要为了执行稳定而使用延迟，而是通过增加中间节点判断，因为延迟实际上是在掩盖问题，在用户设备存在高延迟时仍然不会稳定。**

### 第一轮即命中

尽可能扩充 `next` 列表，保证任何游戏画面都处于预期中，实现一次截图就命中目标节点。项目一般拒绝一切形式的重试机制，一定要保证在一次流程中完成所有任务。

### 识别 → 操作 → 再识别

每一步操作后必须重新识别确认画面变化，禁止假设操作后状态：

- **推荐**：识别 A → 点击 A → 识别 B → 点击 B
- **禁止**：整体识别一次 → 点击 A → 点击 B → 点击 C

例如：点击提交按钮后必须识别确认提交成功（用户网络可能延迟，界面可能卡死）。

### 禁止盲目重试

遇到 bug 时找根因，详细到具体哪个节点失败、哪个识别不符合预期，去修补对应节点的识别/操作问题。**禁止**同样的操作再试一次、盲目添加 `max_hit`。

### 处理弹窗和加载

好的流程是正常主线能跑、弹窗能处理、加载能等过去、不在目标场景时能自动跳过去。常见做法是在 `next` 里挂：

- `[JumpBack]SceneAnyEnterWorld`
- `[JumpBack]SceneClickBlankToExit`
- `[JumpBack]SceneLoading`

### OCR 写完整文本

`expected` 写完整文本，不写半截。多语言处理由 CI 工作流自动完成。需要片段或手写正则时添加 `// @i18n-skip` 标记。

### 先复用，再新增

写新节点前先查已有 Pipeline 是否已有现成能力。优先使用 SceneManager 公共接口，**禁止直接引用 `__ScenePrivate*` 内部节点**。

### 配套文件

新增或修改任务时，改动通常涉及多个文件：
- `assets/resource/tasks/*.json` — 任务配置
- `assets/resource/base/pipeline/**/*.json` — Pipeline 节点
- `assets/resource/locales/interface/` 下五种语言文件 — i18n 文案
- `assets/interface.json` — 注册 task 文件

### switch 控制节点启停

```jsonc
// 任务配置中通过 switch option 控制节点 enabled
"MyFeature": {
    "type": "switch",
    "cases": {
        "Yes": { "pipeline_override": { "MyNode": { "enabled": true } } },
        "No":  { "pipeline_override": { "MyNode": { "enabled": false } } }
    }
}
```

## 子模块目录

复杂任务建议拆分为独立目录，主流程与辅助节点分离：

```
pipeline/WithdrawMoney/
├── WithdrawMoney.json       # 主流程节点
└── WithdrawMoneyStatus.json # 辅助识别/动作节点
```

## 审查清单

- [ ] 字段名拼写正确、类型合法（核对 Pipeline 协议）
- [ ] 无不必要的 `pre_delay` / `post_delay` / `timeout`，不需要时显式设为 0
- [ ] `next` 列表覆盖所有可能画面，含弹窗/加载/异常，力争一次截图命中
- [ ] 每次操作后有识别验证，不假设操作后状态（识别→操作→再识别）
- [ ] 无盲目重试逻辑（同样操作再试一次、随意加 `max_hit`）
- [ ] 弹窗和加载有对应的 `[JumpBack]` 处理节点
- [ ] 先查已有 Pipeline 是否有现成能力，优先复用
- [ ] 未直接引用 `__ScenePrivate*` 内部节点
- [ ] 场景跳转后有 InScene 检查节点确认到达目标场景
- [ ] ROI / target 坐标基于 1280×720
- [ ] 自定义动作名与 Python `@AgentServer.custom_action("name")` 一致
- [ ] 自定义识别/动作参数与 Python 代码中解析的参数名一致
- [ ] 用户消息优先用 `maafocus.PrintT()`（Python 侧），简单通知用 JSON `focus`（pipeline 侧）
- [ ] OCR `expected` 写完整文本
- [ ] 使用 `post_wait_freezes` 或中间节点避免重复点击
- [ ] 配套文件齐全（task JSON、pipeline JSON、i18n、interface.json 注册）

## 参考

- Pipeline 协议完整规范：[MaaFramework PipelineProtocol](https://github.com/MaaXYZ/MaaFramework/blob/main/docs/en_us/3.1-PipelineProtocol.md)
- Python 自定义动作开发：`agent/custom/action/` 目录下的实现
- 节点测试：`docs/zh_cn/develop/`
