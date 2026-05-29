# InScene 场景识别

## 概述

InScene 是 MaaNTE 的场景识别模块，用于判断当前画面是否处于某个特定场景（大世界、背包、菜单等）。

所有识别节点定义在 `assets/resource/base/pipeline/Interface/Scene/Status.json` 中，可在多个 Pipeline 任务中统一引用和复用。

## 核心概念

**InScene 只做一件事：告诉你"现在在哪个界面"。** 它只负责识别、不做任何操作。

配合 SceneManager（`[JumpBack]` 场景跳转），可以实现：识别当前不在目标场景 → 自动跳转到目标场景 → 确认已到达 → 继续业务。

## 使用方式

在 Pipeline 中通过 `And` / `Or` 引用 InScene 节点：

```jsonc
{
    "MyTaskCheckScene": {
        "recognition": {
            "type": "And",
            "param": {
                "all_of": ["InWorld"]   // 确认在大世界界面
            }
        },
        "next": ["MyTaskNextStep"]
    }
}
```

配合 `[JumpBack]` 实现自动跳转：

```jsonc
{
    "MyTaskEntry": {
        "next": [
            "MyTaskCheckInWorld",                     // 先检查是否已在目标场景
            "[JumpBack]SceneAnyEnterWorld"            // 不在则自动跳转到大世界
        ]
    },
    "MyTaskCheckInWorld": {
        "recognition": { "type": "And", "param": { "all_of": ["InWorld"] } },
        "next": ["MyTaskNextStep"]
    }
}
```

> **重要**：跳转后必须有 InScene 检查节点，确认已在目标场景再继续，避免反复跳转导致死循环。

## 可用节点

所有节点定义在 `Interface/Scene/Status.json`，文件名和节点描述已标注用途。

### 场景识别

| 节点 | 说明 | 识别方式 |
|------|------|----------|
| `InWorld` | 在大世界界面 | And（同时存在 EscMenu 按钮和任务按钮） |
| `InEscMenu` | 在 Esc 菜单内 | OCR 识别 "猎人等级" |
| `InBagMenu` | 在背包菜单内 | OCR 识别 "背包" |
| `InCityTycoonMenu` | 在都市大亨菜单内 | OCR 识别 "都市大亨" |
| `InExplorationGuideMenu` | 在探索指南菜单内 | OCR 识别 "探索指南" |
| `InBattlePassMenu` | 在环期赏令菜单内 | OCR 识别 "历练奖赏" |
| `InScarboroughFairMenu` | 在斯卡布罗集市内 | OCR 识别 "弧盘研募" |
| `InEventsMenu` | 在活动菜单内 | OCR 识别 "活动" |
| `InCharactersMenu` | 在角色菜单内 | OCR 识别 "个人信息" |

### 流程控制

| 节点 | 说明 |
|------|------|
| `EmptyNode` | 空节点，无识别无动作，仅用于流程跳转 |
| `StopNode` | 流程结束节点，执行 `StopTask` 终止当前任务 |

## 新增场景识别节点

当需要新的场景判断时，在 `Status.json` 中添加节点：

```jsonc
"InNewMenu": {
    "desc": "在新菜单内",
    "recognition": {
        "type": "OCR",
        "param": {
            "roi": [60, 20, 110, 40],
            "expected": ["新菜单名称"],
            "threshold": 0.8
        }
    }
}
```

- 命名规范：`InXxxMenu` 或 `InXxx`
- 建议使用 OCR 识别页面顶部标题文字（文字类识别用 OCR，不要用 TemplateMatch）
- `expected` 写完整文本，多语言由 CI 自动同步
- 添加后在 SceneManager 的 `__ScenePrivateAnyEnterXxxSuccess` 节点中引用，即可接入万能跳转链
