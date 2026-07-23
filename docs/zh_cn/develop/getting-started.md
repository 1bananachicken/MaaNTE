# 快速开始

以 [#223 自动抚摸小动物](https://github.com/1bananachicken/MaaNTE/issues/223) 为例，走一遍从需求到合并的完整开发流程。最终 PR 见 [#231](https://github.com/1bananachicken/MaaNTE/pull/231)。

## 环境准备

- Git
- Python 3.11+

```bash
git clone --recurse-submodules https://github.com/1bananachicken/MaaNTE.git
cd MaaNTE
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

推荐使用 [VS Code](https://code.visualstudio.com/)，安装 **Maa Pipeline Support** 插件。

## 0. Git 前置与提交规范

本项目遵循 [约定式提交（Conventional Commits）](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。

### 格式

```
<type>(<scope>): <subject>
```

### type（类型）

| 前缀 | 说明 | 示例 |
|------|------|------|
| `feat` | 新增功能 | `feat: 添加抚摸功能` |
| `fix` | 修复 Bug | `fix(FishNew): 修复无鱼饵无鱼鳞币时无法卖鱼` |
| `docs` | 仅文档更改 | `docs: 更新开发文档` |
| `style` | 格式调整（不影响代码含义） | `style: 修复 JSON 缩进` |
| `refactor` | 重构 | `refactor: 调整抚摸功能点击位置至指定 ROI 中心` |
| `perf` | 性能优化 | `perf: 删除 CI 产物中的 deps 目录，减小体积` |
| `test` | 添加或修改测试 | `test: 补充分类器测试用例` |
| `chore` | 构建/依赖/工具变动 | `chore: 更新 MaaFramework 版本` |
| `revert` | 回退 | `revert: 恢复 utils.time 原有逻辑` |
| `ci` | CI/CD 变更 | `ci: 更新 OCR 模型版本` |

### scope（范围）

可选，表示影响范围，如 `FishNew`、`Touch`、`WithdrawMoney`。

### subject（简述）

简明扼要，说明做了什么，不以句号结尾。

### 示例提交

```
feat: 添加抚摸功能

- 新增 Touch.json Pipeline 实现自动循环抚摸
- 新增 TouchLoopCount、TouchDelayAfterF、TouchDelayAfterClick 三个可配置选项
- 新增五语言 i18n 文案
```

### 子模块 (Submodule)

本项目包含 `assets/MaaCommonAssets` 和 `assets/MaaNTEModels` 两个子模块。如果拉取代码后出现模型缺失或 Git 提示子模块被"修改"：

```bash
git submodule update --init --recursive
```

## 1. 确认需求

在 [#223](https://github.com/1bananachicken/MaaNTE/issues/223) 中，用户提出需要一个自动循环抚摸小动物的功能。先在 Issue 里确认需求合理，再开始动手。

## 2. Fork 并创建分支

```bash
git checkout -b feat/touch-animal
```

尽早创建 **Draft PR**，避免重复劳动。

## 3. 编写 Pipeline

Pipeline JSON 位于 `assets/resource/base/pipeline/`，是 MaaNTE 最常改的代码。

### 核心原则：先识别，后操作

```
识别（看屏幕） → 操作（动手） → 下一步（跳转）
```

**铁律：永远先识别再操作。不能假设"点了按钮，下一个画面一定出现"。**

### 真实示例：Touch.json

以下来自 [#231](https://github.com/1bananachicken/MaaNTE/pull/231) 的实际节点。整个流程是：检测到"抚摸"按钮 → 按 F → 点击屏幕中央 → 按 Esc 关闭 → 回到检测，循环直到 100 次后自动停止：

```jsonc
{
    "TouchDetect": {
        "recognition": {
            "type": "OCR",
            "param": {
                "roi": [731, 326, 182, 136],
                "expected": ["抚摸"]
            }
        },
        "action": { "type": "DoNothing" },
        "rate_limit": 0,
        "pre_delay": 0,
        "post_delay": 0,
        "max_hit": 100,          // 最多 100 次，即抚摸 100 次后停止
        "next": ["TouchPressF"]
    },
    "TouchPressF": {
        "recognition": { "type": "DirectHit" },
        "action": {
            "type": "ClickKey",
            "param": { "key": 70 }  // F 键
        },
        "rate_limit": 0,
        "pre_delay": 0,
        "post_delay": 1000,      // 等 1 秒动画
        "next": ["TouchClickCenter"]
    },
    "TouchClickCenter": {
        "recognition": { "type": "DirectHit" },
        "action": {
            "type": "Click",
            "param": { "target": [660, 480] }  // 点击屏幕中央
        },
        "rate_limit": 0,
        "pre_delay": 0,
        "post_delay": 1000,      // 等抚触动画播完
        "next": ["TouchPressEsc"]
    },
    "TouchPressEsc": {
        "recognition": { "type": "DirectHit" },
        "action": {
            "type": "ClickKey",
            "param": { "key": 27 }  // Esc 关闭界面
        },
        "rate_limit": 0,
        "pre_delay": 0,
        "post_delay": 0,
        "next": ["TouchDetect"]  // 回到开头循环
    }
}
```

几个要点：

- `DirectHit` 表示不做识别，直接执行动作（因为前一节点的识别已经确认了状态）
- `rate_limit` / `pre_delay` / `post_delay` **显式设为 0**（协议默认值会引入隐式等待）
- `max_hit: 100` 限制循环次数
- 所有坐标基于 **1280×720**

### 常用识别方式

| 方式 | 关键词 | 使用场景 |
|------|--------|----------|
| 模板匹配 | `TemplateMatch` | 找固定图标、按钮 |
| 文字识别 | `OCR` | 读屏幕文字 |
| 颜色匹配 | `ColorMatch` | 检测特定颜色区域 |
| 同时满足 | `And` + `all_of` | 多条件都满足才命中 |
| 直接命中 | `DirectHit` | 不做识别，直接执行动作 |
| 自定义动作 | `Custom` + `custom_action` | 调用 Python 自定义动作 |

### next 列表与流程控制

按顺序尝试，首个命中的节点执行。**尽可能扩充 next 列表**，覆盖弹窗、加载、异常等所有可能画面。

```jsonc
"next": [
    "MyTaskMainStep",
    "[JumpBack]HandlePopup",
    "[JumpBack]SceneClickBlankToExit"
]
```

- `[JumpBack]` — 处理完中断后返回父节点继续识别
- `[Anchor]` — 动态引用锚点

## 4. 任务配置

在 `assets/resource/tasks/<TaskName>.json` 中定义任务入口和可配置选项。新建后**必须在 `assets/interface.json` 的 `import` 数组中注册**。

真实示例（`tasks/Touch.json`）：

```jsonc
{
    "task": [{
        "name": "Touch",
        "label": "$task_touch_label",
        "entry": "TouchDetect",
        "description": "$task_touch_desc",
        "option": ["TouchLoopCount", "TouchDelayAfterF", "TouchDelayAfterClick"]
    }],
    "option": {
        "TouchLoopCount": {
            "type": "input",
            "label": "$task_touch_option_loop_count",
            "description": "$task_touch_option_loop_count_desc",
            "inputs": [{
                "name": "count",
                "default": "100",
                "pipeline_type": "int",
                "verify": "^\\d+$"
            }],
            "pipeline_override": {
                "TouchDetect": { "max_hit": "{count}" }
            }
        },
        "TouchDelayAfterF": {
            "type": "input",
            "inputs": [{
                "name": "delay", "default": "1000",
                "pipeline_type": "int", "verify": "^\\d+$"
            }],
            "pipeline_override": {
                "TouchPressF": { "post_delay": "{delay}" }
            }
        },
        "TouchDelayAfterClick": {
            "type": "input",
            "inputs": [{
                "name": "delay", "default": "1000",
                "pipeline_type": "int", "verify": "^\\d+$"
            }],
            "pipeline_override": {
                "TouchClickCenter": { "post_delay": "{delay}" }
            }
        }
    }
}
```

三种选项类型：`switch`（开关）、`input`（数值输入）、`select`（下拉选择）。

## 5. i18n 文案

在 `assets/resource/locales/interface/` 下五种语言文件中添加翻译。Task/option 的 `label` 和 `description` 使用 `$key` 格式：

```jsonc
// zh_cn.json
"task_touch_label": "抚摸小动物",
"task_touch_desc": "自动循环抚摸小动物直至成就完成",
"task_touch_option_loop_count": "抚摸次数",
"task_touch_option_loop_count_desc": "最多抚摸多少次后停止",
"task_touch_option_delay_after_f": "F 键后延迟",
"task_touch_option_delay_after_f_desc": "按下 F 键后等待的毫秒数",
"task_touch_option_delay_after_click": "点击后延迟",
"task_touch_option_delay_after_click_desc": "点击屏幕后等待的毫秒数"
```

五种语言文件：`zh_cn.json`、`zh_tw.json`、`en_us.json`、`ja_jp.json`、`ko_kr.json`。

## 6. OCR 文本的 i18n 同步

OCR 节点的 `expected` 只需填写**完整的中文文本**，多语言同步由 `.github/workflows/i18n-sync.yml` 工作流自动完成。

```jsonc
// ✅ 允许：完整文本
"expected": ["抚摸"]

// ❌ 不允许：片段
"expected": ["摸"]

// 需要跳过 i18n 同步时添加标记
"expected": [
    // @i18n-skip
    "自定义正则"
]
```

## 7. 测试

使用 VS Code 的 **Maa Pipeline Support** 插件对节点右键执行即可测试识别和动作。

## 8. 提交与 PR

```bash
git add .
git commit -m "feat: 添加抚摸功能"
git push origin feat/touch-animal
```

在 GitHub 上创建 PR（从你的 fork 到 `1bananachicken/MaaNTE` 的 `dev` 分支）。

提交前请阅读 [PR 规范](./pull-request-guidelines.md)，补充关联 Issue、变更摘要、验证记录，以及必要的截图或日志。

## 接下来看什么

- [PR 规范](./pull-request-guidelines.md)
- [Pipeline 编写指南](./pipeline-guide.md)
- [自定义动作开发](./custom-action.md)
- [场景管理器](./scene-manager.md)
- [节点测试](./node-testing.md)
- [编码规范](./coding-standards.md)
