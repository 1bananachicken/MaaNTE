---
name: task-config
description: MaaNTE 任务配置（tasks/*.json）编写指南。覆盖任务入口、选项类型（switch/input/select）、pipeline_override、i18n、控制器限制等。在添加新任务、修改任务选项、配置 pipeline_override 时使用。
---

# MaaNTE 任务配置编写指南

## 文件位置

`assets/resource/tasks/<TaskName>.json`

新建 task 文件后，必须将其注册到 `assets/interface.json` 的 `import` 数组中，否则 MaaFramework 不会加载：

```jsonc
// assets/interface.json
{
    "task": [],
    "import": [
        "resource/tasks/WithdrawMoney.json",
        "resource/tasks/MyNewTask.json"     // <-- 添加这一行
    ]
}
```

## 基本结构

```jsonc
{
    "task": [
        {
            "name": "MyTask",
            "label": "$task_my_task_label",
            "entry": "MyTaskEntrance",          // Pipeline 入口节点名
            "description": "$task_my_task_desc",
            "option": [                          // 启用的选项列表
                "MyOption1",
                "MyOption2"
            ],
            "controller": [                      // 可选：限制控制器类型
                "Win32",
                "Win32-Front"
            ]
        }
    ],
    "option": {
        // 选项定义（见下文）
    }
}
```

一个文件可定义多个 task（如 Fish.json 含 `Fish` 和 `FishNew`），共享同一组 option 定义。

## 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 任务名（PascalCase） |
| `label` | string | ✅ | UI 标签，`$i18n_key` 格式 |
| `entry` | string | ✅ | Pipeline 入口节点名，对应 pipeline JSON 中的节点 key |
| `description` | string | ❌ | UI 描述，`$i18n_key` 格式（不加 `$` 则为纯文本） |
| `option` | string[] | ❌ | 启用的选项名列表，对应 `option` 块中的 key |
| `controller` | string[] | ❌ | 限制可用控制器：`"Win32"` / `"Win32-Front"` / `"Win32-Background"`。不写 = 通用 |

## 选项类型

### switch（布尔开关）

最常用。用户选择 Yes/No，通过 `pipeline_override` 控制节点 `enabled`：

```jsonc
"MySwitchOption": {
    "type": "switch",
    "label": "$task_xxx_option_yyy",
    "description": "$task_xxx_option_yyy_desc",
    "default_case": "No",                // 默认值（Yes/No）
    "cases": [
        {
            "name": "Yes",
            "label": "$option_switch_case_yes",   // 一般情况用全局开关文案，特殊需求可替换为自定义 i18n key
            "pipeline_override": {
                "SomeNode": { "enabled": true }
            },
            "option": [                   // 可选：Yes 时显示子选项
                "SubOption"
            ]
        },
        {
            "name": "No",
            "label": "$option_switch_case_no",
            "pipeline_override": {
                "SomeNode": { "enabled": false }
            }
        }
    ]
}
```

### input（数值/文本输入）

用户输入值，通过 `{变量名}` 注入到 `pipeline_override`：

```jsonc
"MyInputOption": {
    "type": "input",
    "label": "$task_xxx_option_yyy",
    "description": "$task_xxx_option_yyy_desc",
    "inputs": [
        {
            "name": "count",
            "default": "10",
            "pipeline_type": "int",
            "verify": "^\\d+$"              // 可选：正则校验
        }
    ],
    "pipeline_override": {
        "SomeNode": {
            "custom_action_param": {
                "count": "{count}"           // {变量名} 替换为用户输入值
            }
        }
    }
}
```

### select（下拉选择）

预设多个选项值：

```jsonc
"MySelectOption": {
    "type": "select",
    "label": "$task_xxx_option_yyy",
    "description": "$task_xxx_option_yyy_desc",
    "cases": [
        {
            "name": "0.8",
            "pipeline_override": {
                "SomeNode": { "recognition": { "param": { "threshold": 0.8 } } }
            }
        },
        {
            "name": "0.6",
            "pipeline_override": {
                "SomeNode": { "recognition": { "param": { "threshold": 0.6 } } }
            }
        }
    ]
}
```

## pipeline_override 格式

`pipeline_override` 是 `{ "节点名": { "字段": 值 } }` 的映射，支持的字段包括所有 Pipeline 节点字段：

```jsonc
"pipeline_override": {
    "NodeName": {
        "enabled": true,                    // 启用/禁用节点
        "max_hit": 3,
        "next": ["OtherNode"],
        "pre_delay": 200,
        "post_delay": 200,
        "timeout": 30000,
        "recognition": {
            "param": {
                "roi": [0, 0, 100, 50],
                "threshold": 0.7,
                "expected": ["新文本"]
            }
        },
        "action": {
            "param": {
                "target": [100, 200]
            }
        },
        "custom_action_param": {
            "key": "value"                   // CustomAction 的 JSON 参数
        }
    }
}
```

## 选项级联（子选项）

switch 的 case 中可嵌套 `"option"` 数组，实现"启用某功能后才显示子选项"：

```jsonc
"AutoSkipStory": {
    "type": "switch",
    "cases": [
        {
            "name": "Yes",
            "option": ["AutoSkipStoryDialog"]   // Yes 时才显示
        },
        {
            "name": "No"
        }
    ]
}
```

## i18n

- task/option 的 `label`、`description` 使用 `$key` 格式引用翻译，key 定义在 `assets/resource/locales/interface/` 下五种语言文件中：
  - `zh_cn.json` — 简体中文
  - `zh_tw.json` — 繁体中文
  - `en_us.json` — 英语
  - `ja_jp.json` — 日语
  - `ko_kr.json` — 韩语
- Pipeline 中 OCR 节点的 `expected` 只需填写**完整的中文文本**，多语言同步由 `.github/workflows/i18n-sync.yml` 工作流自动完成。需要跳过时添加 `// @i18n-skip` 标记
- `option_switch_case_yes` / `option_switch_case_no` 是全局 switch case 默认标签，所有语言文件都需定义。一般 switch 直接复用即可；若有特殊需求（如文案不应是简单的"启用/禁用"），可将 case 的 `label` 替换为自定义 `$key`
- 纯文本（不加 `$`）直接显示，不走翻译

## 控制器限制

`controller` 数组限制任务可在哪些控制器下运行：

- `"Win32"` — 后台 SendMessage 模式（需管理员权限）
- `"Win32-Front"` — 前台 Seize 模式（会抢占鼠标）
- `"Win32-Background"` — 后台 SendMessageWithWindowPos 模式

不写 `controller` 字段 = 所有控制器通用。多个值 = 任一匹配即可。

## 完整示例

```jsonc
{
    "task": [
        {
            "name": "MyNewTask",
            "label": "$task_my_new_task_label",
            "entry": "MyNewTaskEntrance",
            "description": "$task_my_new_task_desc",
            "option": [
                "MyNewTaskAutoMode",
                "MyNewTaskLoopCount"
            ],
            "controller": [
                "Win32",
                "Win32-Front"
            ]
        }
    ],
    "option": {
        "MyNewTaskAutoMode": {
            "type": "switch",
            "label": "$task_my_new_task_option_auto_mode",
            "description": "$task_my_new_task_option_auto_mode_desc",
            "default_case": "Yes",
            "cases": [
                {
                    "name": "Yes",
                    "label": "$option_switch_case_yes",
                    "pipeline_override": {
                        "MyNewTaskAutoStep": { "enabled": true }
                    }
                },
                {
                    "name": "No",
                    "label": "$option_switch_case_no",
                    "pipeline_override": {
                        "MyNewTaskAutoStep": { "enabled": false }
                    }
                }
            ]
        },
        "MyNewTaskLoopCount": {
            "type": "input",
            "label": "$task_my_new_task_option_loop_count",
            "description": "$task_my_new_task_option_loop_count_desc",
            "inputs": [
                {
                    "name": "count",
                    "default": "5",
                    "pipeline_type": "int",
                    "verify": "^\\d+$"
                }
            ],
            "pipeline_override": {
                "MyNewTaskEntry": {
                    "max_hit": "{count}"
                }
            }
        }
    }
}
```

## 审查清单

- [ ] 任务 `name` 使用 PascalCase
- [ ] `entry` 节点名在对应 Pipeline JSON 中存在
- [ ] task 文件已注册到 `assets/interface.json` 的 `import` 数组
- [ ] `label` / `description` 的 `$i18n_key` 在所有五种语言文件中已定义
- [ ] OCR `expected` 写完整文本，无需手动维护多语言（CI 自动同步）
- [ ] switch option 有 `default_case`
- [ ] `pipeline_override` 中的节点名在 Pipeline JSON 中存在
- [ ] `custom_action_param` 参数名与 Python CustomAction 解析一致
- [ ] `controller` 限制合理（前台/后台/通用）
- [ ] input option 的 `verify` 正则正确
- [ ] 子选项（case 内 option）在 option 块中有定义
