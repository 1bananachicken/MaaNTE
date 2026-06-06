# PR 规范

本文档用于说明 MaaNTE 项目的 Pull Request（PR）提交、描述、验证与评审要求。提交 PR 前，请先阅读 [快速开始](./getting-started.md) 与 [编码规范](./coding-standards.md)。

## 基本原则

- **目标分支**：所有功能、修复与文档 PR 默认提交到 `dev` 分支。
- **小步提交**：一个 PR 只解决一个明确问题，避免把无关重构、格式化和功能改动混在一起。
- **尽早 Draft**：需求尚未完全确定、实现还在探索时，请先创建 Draft PR，方便维护者提前发现方向问题。
- **先同步上游**：提交评审前 rebase 或 merge 最新 `origin/dev`，减少重复冲突。
- **自己负责**：可以使用 AI 辅助，但提交者必须理解并能解释 PR 中的每一处关键改动。

## 分支与提交

### 分支命名

推荐使用以下格式：

| 类型 | 示例 | 适用场景 |
|------|------|----------|
| `feat/<name>` | `feat/touch-animal` | 新增任务或功能 |
| `fix/<name>` | `fix/fish-sell-stuck` | 修复 Bug |
| `docs/<name>` | `docs/pr-guidelines` | 文档改动 |
| `refactor/<name>` | `refactor/scene-manager` | 不改变行为的结构调整 |
| `chore/<name>` | `chore/update-deps` | 工具、依赖、构建等维护工作 |

### 提交信息

提交信息遵循 [约定式提交（Conventional Commits）](https://www.conventionalcommits.org/zh-hans/v1.0.0/)：

```text
<type>(<scope>): <subject>
```

示例：

```text
feat(Touch): 添加自动抚摸任务
fix(FishNew): 修复无鱼饵时卖鱼流程卡住
docs: 补充 PR 规范
```

`scope` 可选，建议填写任务名、模块名或影响范围。`subject` 使用简短中文描述，不以句号结尾。

## PR 标题

PR 标题同样使用约定式提交格式，便于维护者快速判断变更类型：

| 推荐标题 | 不推荐标题 |
|----------|------------|
| `feat(Touch): 添加自动抚摸任务` | `添加新功能` |
| `fix(FishNew): 修复卖鱼流程卡住` | `修一下鱼` |
| `docs: 补充 Pipeline 调试说明` | `update docs` |

如果 PR 仍在开发中，请使用 GitHub Draft PR，而不是在标题前长期保留 `WIP`。

## PR 描述

PR 描述至少应包含以下信息：

### 关联内容

- 关联 Issue：`Closes #123`、`Fixes #123` 或 `Related #123`
- 如果没有 Issue，请说明需求来源、复现方式或为什么需要这个改动

### 变更摘要

用 2～5 条 bullet 说明改了什么，例如：

- 新增 `Touch` 任务入口与任务配置
- 新增 `TouchDetect` / `TouchPressF` 等 Pipeline 节点
- 补充界面文案与文档说明

### 验证记录

说明你做过哪些验证。不要只写“已测试”，应写清楚测试入口、设备/控制器、结论。

推荐格式：

```markdown
## 验证

- [x] 使用 Maa Pipeline Support 测试 `TouchDetect`，可稳定识别 1280×720 截图
- [x] 使用 Win32-Front 控制器完整运行 `Touch` 任务 3 次，均正常结束
- [x] 检查新增任务已在 `assets/interface.json` 注册
```

### 截图、日志与素材

涉及识别、点击、界面跳转或 Bug 修复时，应尽量提供：

- 游戏界面截图，标注关键 ROI、按钮或模板位置
- 失败前后的 `maa.log` 关键片段
- 新增或修改模板图片的路径
- 修复前后的行为对比

## 变更要求

### Pipeline / 任务改动

- 每一步应遵循“识别 → 操作 → 再识别”，禁止只识别一次后连续点击多个位置。
- 坐标、ROI、模板图片均以 **1280×720** 为基准。
- `next` 列表应覆盖主线、加载、弹窗、返回目标场景等常见状态。
- 禁止用盲目重试、随意 `max_hit` 或硬延迟掩盖识别/流程问题。
- 新增任务必须在 `assets/interface.json` 中注册对应任务配置。
- 新增任务选项时，需要补充对应 i18n 文案。

### Python CustomAction 改动

- 仅在 Pipeline 难以表达时引入 Python，自定义动作不应接管整体流程。
- 长循环必须检查 `context.tasker.stopping`，避免用户停止任务后仍继续执行。
- 用户可见消息使用 `maafocus.PrintT()`，调试信息使用项目日志工具。
- 新增文件后需要在 `agent/custom/action/__init__.py` 中导出。

### 文档与配置改动

- 用户可见行为、任务选项或开发约定发生变化时，应同步更新文档。
- 依赖、构建、发布相关改动需要说明动机，并提交对应锁文件变更。
- 避免提交本地缓存、运行产物、调试截图或无关格式化结果。

## 提交前检查清单

提交 PR 前至少自查：

- [ ] PR 目标分支是 `dev`
- [ ] PR 标题符合约定式提交格式
- [ ] 变更范围单一，没有混入无关改动
- [ ] 已同步最新 `origin/dev`
- [ ] 新增或修改任务时，已检查 `assets/interface.json`、任务配置、Pipeline 与 i18n
- [ ] 涉及识别/点击时，已提供截图、ROI、模板路径或日志证据
- [ ] 已说明验证方式和验证结果
- [ ] 文档已随用户可见行为或开发约定同步更新

## 评审与合并

- 维护者可能要求补充截图、日志、复现步骤或测试记录；请在同一 PR 中继续修正。
- 如果评审意见涉及设计方向，请先回复确认方案，再继续大量改动。
- PR 被合并后，如发现后续问题，请新开 Issue 或 PR 跟进，不要在已合并 PR 中继续讨论新的无关需求。
