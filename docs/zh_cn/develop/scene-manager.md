# 场景管理器

## 概述

SceneManager 是 MaaNTE 的场景导航模块，提供"从任意界面自动导航到目标场景"的能力。
> [!NOTE]
> ⚠ 绝赞施工中，如有遗漏欢迎补充或者@EeeMaoY

### 设计原理

SceneManager 分为两层：

- **公共接口**（`Interface/Scene/`）— Pipeline 任务使用的节点，**名称不包含 `__ScenePrivate`**
- **私有实现**（`SceneManager/`）— 内部节点，以 `__ScenePrivate` 开头，**禁止在 Pipeline 中直接引用**

公共接口通过 `[JumpBack]` 机制组织成有层级的跳转链：当目标界面无法直接进入时，自动回退到更基础的场景再重试。

## 使用方式

```jsonc
{
    "MyTaskEntry": {
        "next": [
            "MyTaskMainStep",
            "[JumpBack]SceneAnyEnterCityTycoonsMenu"  // 不在目标场景时自动跳转
        ]
    }
}
```

**典型模式**（参考 `Interface/Example/SceneExample.json`）：

```jsonc
{
    "MyTaskStart": {
        "next": [
            "MyTaskContinue",
            "[JumpBack]SceneAnyEnterWorld"   // 不在大世界时先跳过去
        ]
    },
    "MyTaskContinue": {
        "recognition": "And",
        "all_of": ["InWorld"],               // 确认已在大世界
        "next": ["MyTaskNextStep"]
    }
}
```

关键点：跳转后必须有状态检查节点（如 `InWorld`）确认已在目标场景，避免反复跳转导致死循环。

## 常用接口

### 场景跳转（Scene.json / SceneMenu.json）

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

### 状态检测（Status.json）

| 接口 | 说明 |
|------|------|
| `InWorld` | 在大世界界面 |
| `InEscMenu` | 在 Esc 菜单内 |
| `InBagMenu` | 在背包界面 |
| `InCityTycoonMenu` | 在都市大亨界面 |
| `InExplorationGuideMenu` | 在探索指南界面 |
| `InBattlePassMenu` | 在环期赏令界面 |
| `InCharactersMenu` | 在角色界面 |

### 流程控制

| 接口 | 说明 |
|------|------|
| `EmptyNode` | 空节点（流程中转） |
| `StopNode` | 终止当前任务 |

## `[JumpBack]` 原理

当节点 `next` 包含 `[JumpBack]NodeName` 时，MaaFramework 在识别失败后会跳回 `NodeName` 执行，处理完后再返回原节点的 next 继续尝试。

SceneManager 利用这个特性实现层级回溯：

```
SceneAnyEnterCityTycoonsMenu → [
    直接进入都市大亨（按快捷键）,
    确认已进入,
    [JumpBack]回退到进入大世界流程
]
```

## 扩展新的场景接口

需要新增场景跳转时：

1. 在 `SceneManager/` 下添加 `__ScenePrivate*` 私有节点处理实际导航
2. 在 `Interface/Scene/` 或 `SceneMenu.json` 中添加公共接口节点
3. 在 `Status.json` 中添加状态检测节点（如 `InNewMenu`）

**禁止在 Pipeline 任务中直接引用 `__ScenePrivate*` 节点**——这些节点的名称和逻辑可能随版本变更。
