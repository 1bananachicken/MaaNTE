---
name: maante-issue-log-analysis
description: 分析 MaaNTE 上游仓库公开 Issue（`https://github.com/1bananachicken/MaaNTE/issues/...` 或 `#1234`）。自动抓取 issue 正文和评论中的日志附件（`MaaNTE-logs-*.zip` 或裸日志文件），下载解压后从 `maa.log`、agent 日志（`debug/custom/`）、`mxu-tauri.log`、`mxu-web-*.log`、`mxu-agent*.log`、`config/*`、`on_error/` 截图、`.dmp` 崩溃转储中筛选关键证据，并结合 MaaNTE 当前仓库代码、MaaFramework 文档、MXU 文档判断根因、给出修复方案。供分析 MaaNTE issue、日志包、识别失败、任务卡死、控制器差异、Pipeline/Agent/MXU 问题时使用。
---

# MaaNTE Issue Log Analysis

## Required Reading

- 开始分析前，先读取同目录的 `KNOWLEDGE.md`，用其中的通用误判规则和已知限制校正分析路径，再读 issue 和日志。
- 如果 issue 涉及控制器差异（前台/后台/Win32 模式）、钓鱼流程、做咖啡、实时辅助、粉爪大劫案这类有状态依赖或特定控制器要求的流程，必须先套用 `KNOWLEDGE.md` 中的对应规则。
- 如果 issue 包含 `.dmp` 文件，必须读取 `.claude/skills/dmp-analysis/SKILL.md` 并严格按其流程执行，不要只凭日志文本猜测崩溃原因。
- 如果用户没有贴出日志、截图、报错文本或清晰复现步骤，不要直接进入严肃分析；先明确说明证据不足，再尽量基于 issue 文本、代码和文档给出初步判断。不要编造根因。

## Scope

- 用于上游公开仓库 `https://github.com/1bananachicken/MaaNTE`。
- 输入可以是完整 issue URL，或 `#1234` 形式的 issue 编号。
- 只分析公开 issue 中可直接访问的附件。
- 如果没有日志包，先明确说明证据不足，再尽量基于 issue 文本、截图、代码和文档给出初步判断。
- 如果维护者已经在 issue 里下了结论，不要直接照抄；仍要用日志和代码自行验证，再把维护者结论作为补强证据。

## Workflow

### 1. 规范化输入

- `#1234` 视为 `https://github.com/1bananachicken/MaaNTE/issues/1234`
- 如果不是 `1bananachicken/MaaNTE`，停止并说明此 skill 不适用。

### 2. 获取 issue 内容

- 使用 `gh issue view` 读取正文和评论。
- 提取这些信息：MaaNTE 版本、控制器类型（Win32 / Win32-Front / Win32-Background）、任务名、预期行为、实际行为、复现步骤、维护者评论。
- 如果维护者已经给出结论，不要直接照抄；仍要用日志和代码自行验证，再把维护者结论作为补强证据。

### 3. 提取日志附件链接

- 关注 `MaaNTE-logs-*.zip` 或类似的日志压缩包，以及裸文件如 `maa.log`、`mxu-tauri.log`、`.dmp` 等。
- 附件可能同时出现在正文和评论中。
- 如果同一个 issue 有多个日志包，先看最新一次复现；如果 issue 在对比不同版本或不同控制器，再补看前面的包。

### 4. 下载并解压日志包

- 用 `curl -L` 下载到 `.cache/issue-logs/issue-<number>/`。
- 解压后用文件工具逐文件读取，不要把整份大日志完整塞进回复。
- 先列一遍解压目录，不要假定结构固定。MaaNTE 日志包根据不同客户端可能包含：
  - `debug/maa.log` — MaaFramework 核心运行时日志（**最权威**）
  - `debug/maa.bak.log` — 前一天的 MaaFramework 日志轮转
  - `debug/custom/YYYY-MM-DD.log` — Python agent 日志（loguru 格式）
  - `debug/custom/runtime.log` — Python agent 日志（标准 logging 回退）
  - `mxu-tauri.log` — MXU Tauri/Rust 后端日志
  - `mxu-web-YYYY-MM-DD.log` — MXU React/TypeScript 前端日志
  - `mxu-agent-<index>-<pid>.log` — MXU 捕获的 agent 子进程 stdout/stderr
  - `config/mxu-MaaNTE.json` — MXU 配置快照
  - `config/maa_pi_config.json` — MaaPiCli 配置快照
  - `config/maa_option.json` — MaaFramework 选项配置
  - `on_error/` — MaaFramework 错误现场截图目录
  - `vision/` — 识别过程可视化截图（如果用户开启了 `save_draw`）
  - `.dmp` 文件 — Windows 崩溃转储
- 如果发现 `.dmp` 文件，记录其文件名和路径，转入步骤 5。

### 5. DMP 崩溃转储分析（如有 `.dmp` 文件则必须执行此步骤）

如果在日志包内或 issue 附件中发现了 `.dmp` 文件，**立即读取 `.claude/skills/dmp-analysis/SKILL.md` 并严格按其流程执行**。不要跳过这一步，不要只凭日志文本猜测崩溃原因。

### 6. 建立时间线

- 先从 issue 文本判断"用户觉得出问题的时刻"。
- MaaNTE 的日志来源多样，按以下优先级串成时间线：
  1. `mxu-tauri.log` — 找 `post_task returned task_id` 锁定本次运行的 task_id
  2. `maa.log` — 用 task_id 跟踪整个任务链执行过程
  3. agent 日志（`debug/custom/`）— 查 Python 自定义动作的执行细节
  4. `mxu-agent*.log` — 看用户在 MXU 运行日志面板里实际看到了什么
  5. `mxu-web-*.log` — 确认 UI 提交了什么配置和任务
- 一个日志包里经常混有很多历史运行，务必用 task_id 过滤，不要把所有 ERROR/WARN 都当成当次复现。
- 如果有 `on_error/` 截图，用它校验当时实际停留画面；如果没有，要检查是未触发 `on_error`，还是被日志清理逻辑删除了（MaaNTE 每次启动会清理旧日志）。

### 7. 区分 issue 当时环境和当前分支

- 先以日志包中的 `config/*` 还原用户当时实际运行的配置和控制器类型。
- 再对照当前仓库代码，判断该问题是当前仍存在，还是当时存在但现在已修复。
- 如果用户日志与当前代码不一致，先按用户版本 tag 复核旧逻辑；若确认已修，再看修复是否已进入 tag / release：已发版建议升级，未发版建议等待 release。

### 8. 回溯到代码和文档

分析顺序：
1. **MaaNTE 仓库** — 任务入口、选项定义、Pipeline 节点、Python 动作。这是最优先的查证来源。
2. **MaaFramework 文档** — Pipeline 运行语义、控制器行为、interface.json 协议。
3. **MXU 文档** — MXU 行为、日志分层、GUI 角色。
4. **仅在必要时**才 clone 上游仓库到 `.cache/upstream-src/<repo>/` 查看源码。
- 输出给用户时，如果提到任务名、选项名、按钮名、提示文案，先到 `assets/resource/locales/interface/zh_cn.json`（UI 文案）和 `assets/resource/locales/agent/zh_cn.json`（agent 文案）查中文，不要直接把 `task_id`、`option_id`、i18n key 当成最终展示文本。

### 9. 只有在满足条件时才下钻第三方仓库

- 先用 issue、日志、MaaNTE 仓库和 MaaFramework/MXU 文档做初步归因。
- 如果怀疑问题在 MaaFramework 或 MXU 实现层，且现有证据不足以确认，再按需 clone 对应上游仓库源码。
- 常见对应关系：
  - `MaaFramework`：`https://github.com/MaaXYZ/MaaFramework` — Pipeline 运行时、控制器、资源加载、任务调度、`maa.log` 中出现的 C++ 源文件
  - `MXU`：`https://github.com/MistEO/MXU` — 前端配置、Tauri 后端编排、实例/任务/agent 生命周期、`mxu-tauri.log` / `mxu-web-*` / `mxu-agent*`

## Log Map

### `maa.log`

- **来源**：MaaFramework 核心运行时（C++）。
- **内容**：资源加载、控制器连接、任务启动、节点识别、动作执行、超时、回调细节、C++ 源文件和函数名。
- **最适合看**：
  - Pipeline 是否按预期推进
  - `next` / `on_error` 是否命中
  - 识别算法细节（模板分数、OCR 结果、动作成功/失败）
  - 控制器、截图、资源加载层面的异常
- **对根因判断最权威**。

### `maa.bak.log`

- **来源**：前一天的 MaaFramework 日志轮转。
- **最适合看**：最新一次复现不在 `maa.log` 中；需要对比前一次成功/失败。

### Agent 日志（`debug/custom/`）

- **来源**：`agent/custom/action/` 下的 Python 自定义动作。
- **文件**：`debug/custom/YYYY-MM-DD.log`（loguru 格式）、`debug/custom/runtime.log`（标准 logging 回退）。
- **内容**：
  - **logger 日志**：通过 `utils.logger` 输出的结构化日志，包含自定义动作执行细节、参数解析、识别结果。MXU 模式下 console_level 强制为 WARNING，仅 WARNING+ 出现在 `mxu-agent*.log`。
  - **maafocus 消息**：通过 `maafocus.Print()/PrintT()` 发送的用户可见消息，经 MaaFramework focus 协议传入 `mxu-agent*.log` 和 `maa.log`。
- **最适合看**：Python 自定义逻辑是否触发、参数解析是否正确、用户看到什么进度/错误提示
- 日志级别和格式见 `../maa-logging/SKILL.md`，maafocus 用法见 `../python-action-guide/SKILL.md`。

### `mxu-tauri.log`

- **来源**：MXU 的 Tauri/Rust 后端。
- **内容**：实例创建、控制器连接、资源加载、任务开始/停止、agent 生命周期、`post_task` 返回值。
- **最适合看**：
  - MXU 是否正确把 UI 操作转成 MaaFramework/Agent 调用
  - 任务有没有被用户手动停止
  - agent 是否成功启动、断开、退出
  - 获取 `task_id` 以便在 `maa.log` 中跟踪

### `mxu-web-YYYY-MM-DD.log`

- **来源**：MXU 的 React/TypeScript 前端。
- **内容**：`interface.json` 加载、配置读写、任务列表解析、控制器选择、启动参数、高层 UI 流程日志。
- **最适合看**：UI 是否加载了正确配置、用户点击后提交了什么任务和配置

### `mxu-agent-<index>-<pid>.log`

- **来源**：MXU 捕获到的 agent 子进程标准输出/标准错误。
- **内容**：子进程控制台输出、MaaFramework stdout 内容、面向 MXU 运行日志面板的消息。
- **最适合看**：用户在 MXU 运行日志面板里实际看到了什么
- **注意**：它是很有用的用户视角，但不是最权威的根因日志。涉及具体运行细节时，优先以 `maa.log` 和 agent 文件日志为准。

### `config/*`

- **来源**：MXU / MaaPiCli 配置快照。
- **常见文件**：
  - `config/mxu-MaaNTE.json` — MXU 完整配置
  - `config/maa_pi_config.json` — MaaPiCli 配置（含 controller 类型、连接参数）
  - `config/maa_option.json` — MaaFramework 选项（含 `save_draw` 等调试开关）
- **最适合看**：实际启用了哪些任务和选项、控制器类型、配置是否与用户描述一致
- **注意**：用户复现后可能改过配置，当前文件不一定就是复现时那一份。如果配置与日志行为冲突，优先信日志。

### `on_error/`

- **来源**：MaaFramework 错误现场截图。
- **最适合看**：实际停留界面、是否被弹窗、加载态、遮罩、分辨率等环境因素干扰
- **规则**：当日志和 issue 文字描述冲突时，**优先相信现场截图**

### `vision/`

- **来源**：`save_draw` 开关开启时的识别过程可视化截图。
- **最适合看**：模板匹配的实际 ROI 位置、OCR 识别区域、算法在画面上的标注

## How To Filter Evidence

### 第一步：从 issue 文本拿到锚点

- MaaNTE 版本（来自 `assets/interface.json` 或用户描述）
- 控制器类型（Win32 / Win32-Front / Win32-Background）
- 任务名/入口名（优先使用 `assets/resource/locales/interface/zh_cn.json` 中的中文名）
- 用户说的"卡住/点错/识别失败/闪退"的画面

### 第二步：从日志里找高价值信号

在 `maa.log` 中搜索：
- `Tasker.Task.Starting` / `Succeeded` / `Failed`
- `Node.Recognition.Failed` 连续重复
- `Node.Action.Failed`
- `SubTaskError` / `TaskChainError`
- `timeout`
- `Warn` / `Error` / `Fatal`
- `ConnectionInfo` / `ConnectFailed`
- `Save image` — 记录了现场截图但用户可能未上传

在 agent 日志中搜索：
- `ERROR` / `CRITICAL` / `WARNING`
- Python traceback（`Traceback (most recent call last)`）
- `ModuleNotFoundError` — 依赖安装问题

在 `mxu-tauri.log` 中搜索：
- `post_task returned task_id` — 获取本次运行的 task_id
- agent 启动失败、断连、被停止
- `error` / `panicked`

在 `mxu-agent*.log` 中搜索：
- Python 异常和 traceback
- 用户实际看到的消息

### 第三步：先锁定"这一次复现"的任务实例

1. 从 `mxu-tauri.log` 找 `post_task returned task_id`
2. 再到 `maa.log` 用这个 `task_id` 跟完整个任务
3. 不要把所有历史运行的 ERROR/WARN 都列出来—只关注当次 task_id 相关
4. 如果 issue 文本说"失败"，但目标 `task_id` 实际 `Tasker.Task.Succeeded`，要明确写出"**本日志未复现用户描述的失败**"

### 第四步：按问题类型侧重

- **Pipeline 识别/动作问题** → `maa.log` + `on_error/` 截图
- **Python 自定义逻辑问题** → agent 日志（`debug/custom/`）+ `mxu-agent*.log`
- **UI / 配置 / 编排问题** → `mxu-web-*` + `mxu-tauri.log` + `config/*`
- **控制器连接/输入问题** → `maa.log` + `config/maa_pi_config.json` + `mxu-tauri.log`
- **崩溃/闪退** → `.dmp` 文件（必须走 dmp-analysis 流程）

### 第五步：回答时只保留关键证据

- 摘几十行足够支撑结论的片段即可。
- 不要把整份日志倾倒进回复。

## Common Patterns

### 控制器不匹配

- 用户描述"功能比别人少"或"某个任务找不到" → 先查 `assets/resource/tasks/<TaskName>.json` 中 `controller` 字段的限制。详见 `KNOWLEDGE.md#controller-mismatch`。

### 识别失败

- `next` 列表中的识别连续失败直到超时：常见于当前画面不在预期分支、模板/OCR 失配、漏了中间节点、被弹窗打断。
- 某个"兜底返回/退出"节点连续成功但流程没有前进：常见于 Pipeline 对当前状态判断错了。
- OCR 模型加载失败（`Failed to load det or rec`、`ocrer_ is null`）：检查 `assets/resource/model/ocr/` 是否有模型文件。
- 用户日志与当前主线代码不一致，且主线已修复：先确认用户版本，必要时切到对应 tag 核对旧逻辑。

### 游戏环境干扰

- 帧生成 / 超分辨率等画质增强功能开启 → 模板匹配和 OCR 结果不可靠。详见 `KNOWLEDGE.md#game-settings-interference`。
- Windows 显示缩放 ≠ 100%、非 1280x720 窗口、非简体中文、路径含中文等环境问题 → 详见 `KNOWLEDGE.md#environmental-requirements`。

### 功能特定 guardrails

- 钓鱼（买饵误识别、FPS 要求、月卡弹窗、鱼饵不足）、做咖啡（角色技能依赖）、实时辅助（传送点限制、快捷键依赖）、粉爪大劫案（不稳定、队伍要求）→ 详见 `KNOWLEDGE.md` 中各功能对应章节。这些规则在分析相关 issue 前必须套用。

### 证据与环境

- issue 文字说"卡死/误点"，但对应 `task_id` 最终 `Tasker.Task.Succeeded`：先明确"**本次日志未复现出用户描述的问题**"，再分析可能性。
- MaaNTE 每次启动会清理旧日志 → 用户如果在复现后重启过 MaaNTE，关键日志可能已丢失。详见 `KNOWLEDGE.md#log-collection-timing`。
- `on_error/` 截图缺失但日志有 `Save image`：可能是用户未上传对应分卷，或被日志清理逻辑在重启时删除。

## Correlating With Code

### MaaNTE 仓库

- 任务入口、选项定义、控制器限制：`assets/resource/tasks/*.json`
- 资源版本和控制器定义：`assets/interface.json`
- Pipeline 节点：`assets/resource/base/pipeline/**/*.json`
- Python 自定义动作：`agent/custom/action/**/*.py`
- Agent 入口和初始化：`agent/main.py`
- 日志系统实现：`agent/utils/logger.py`
- 用户可见消息（maafocus）：`agent/utils/maafocus.py`
- UI 中文文案：`assets/resource/locales/interface/zh_cn.json`
- Agent 中文文案：`assets/resource/locales/agent/zh_cn.json`
- 问题排查文档：`docs/zh_cn/问题排查.md`

### MaaFramework（上游）

- Pipeline 协议：`https://github.com/MaaXYZ/MaaFramework/raw/refs/heads/main/docs/en_us/3.1-PipelineProtocol.md`
- interface.json / agent / controller 语义：`https://github.com/MaaXYZ/MaaFramework/raw/refs/heads/main/docs/en_us/3.3-ProjectInterfaceV2.md`
- 仓库地址：`https://github.com/MaaXYZ/MaaFramework`

### MXU（上游）

- 日志分层、GUI 角色、架构：`https://raw.githubusercontent.com/MistEO/MXU/main/README.md`
- 仓库地址：`https://github.com/MistEO/MXU`

## Linking Code Evidence

- 统一给出对应仓库的远端 GitHub `blob` 行号链接，用尖括号包裹。
- MaaNTE 仓库链接格式：`https://github.com/1bananachicken/MaaNTE/blob/<commit>/<path>#L1-L2`
- `<commit>` 必须是本次分析实际依据的代码版本（默认使用当前 HEAD）。
- 如果引用 MXU、MaaFramework 等上游仓库，也用对应远端 `blob` 链接。

## Output Format

最终回答用这个结构：

```markdown
## Issue 概要

- issue：`#1234`
- 版本 / 控制器：优先写中文
- 任务 / 相关选项：优先写 `assets/resource/locales/interface/zh_cn.json` 中的中文 label/description
- 用户现象：

## 关键证据

<details><summary>点击此处展开</summary>

- `maa.log`：...
- agent 日志（`debug/custom/`）：...
- `mxu-tauri.log`：...
- `mxu-web-*.log`：...
- `mxu-agent*.log`：...
- `config/*`：...
- `on_error/` / `vision/`：...
- 代码依据：远端 GitHub 行号链接

### DMP 崩溃分析

（仅当 issue 存在 .dmp 文件时输出此区域。如果没有 .dmp 文件，删除整个区域。）

</details>

## 根因判断

- 直接结论：
- 证据链：

## 给用户的建议

- 用户现在可以直接尝试的动作：
- 是否建议升级 / 重下完整包 / 同步资源 / 重置配置：
- 是否需要等待开发者修复：
- 是否有临时绕过方案：

## 修复方案

1. 代码 / Pipeline / 配置层修复
2. 需要补充的测试或日志
3. 如问题本身属于不支持场景，给出如何限制入口或改进提示

## 置信度

- 高 / 中 / 低
- 还缺什么证据
```

## Reminders

- **DMP 堆栈必须完整输出**：如果 issue 存在 `.dmp` 文件，最终报告中必须包含 `### DMP 崩溃分析` 区域，禁止省略堆栈帧。分析前必须读取 `.claude/skills/dmp-analysis/SKILL.md`。
- 不要只看一个日志文件下结论。MaaNTE 有多种日志来源（`maa.log`、agent 日志、`mxu-tauri.log`、`mxu-web-*`、`mxu-agent*`），至少交叉验证两个来源。
- 不要把"维护者评论"当成唯一证据。
- 不要把环境告警（如编码警告、路径建议、杀毒软件提示）自动等同于根因。
- 不要把当前分支资源直接当成 issue 当时的真实环境；先看日志包里的 `config/*`。
- 日志和截图冲突时，优先相信现场图，再回头解释 OCR / 模板为何误判。
- 如果结论是"功能不支持"或"by design"，必须给出代码级依据（如任务 JSON 中 `controller` 限制、Pipeline 节点的设计意图）。
- 如果问题本身没有在当前日志中复现，要明确写"证据未复现"，不要硬凑结论。
- 如果 issue 版本很旧，要明确区分"当时的根因"和"当前分支是否已修复"。
- 如果用户日志与当前代码不一致，先按用户版本 tag 复核；若确认已修，再看修复是否已进入 tag / release：已发版建议升级，未发版建议等待 release。
- 如果回答里出现任务名、选项名、按钮名、提示文案，优先使用 `assets/resource/locales/interface/zh_cn.json` 和 `assets/resource/locales/agent/zh_cn.json` 的中文文案。
- 如果回答里引用了具体代码行，直接给远端 GitHub `blob` 行号链接，用尖括号包裹，不要给本地路径加行号。
- 如果一个日志包里混有多天/多次运行，必须用 `task_id` 过滤，不要把所有历史 ERROR 都列出来。
- MaaNTE 每次启动会清理旧日志，如果用户复现后重启了 MaaNTE，要在报告中说明"关键日志可能已被清理"。
- 对控制器相关问题，先查 `assets/resource/tasks/<TaskName>.json` 中 `controller` 字段——很多"功能缺失"实际是控制器不匹配。
- 如果证据表明问题已在新版本修复，明确建议用户升级；如果怀疑安装包、资源文件或配置损坏，明确建议重新下载或重建；如果判断为真实代码缺陷且暂无 workaround，明确建议等待开发者修复。
