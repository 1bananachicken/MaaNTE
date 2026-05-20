---
name: maante-issue-log-analysis
description: 分析 MaaNTE 上游仓库公开 Issue（`https://github.com/1bananachicken/MaaNTE/issues/...` 或 `#1234`）。自动抓取 issue 正文和评论中的日志附件，下载解压后从 `maa.log`、agent 日志、`mxu-tauri.log`、`mxu-web-*.log`、`mxu-agent*.log`、`config/*`、`on_error/` 中筛选关键证据，并结合 MaaNTE、MaaFramework、MXU 的代码和文档判断根因、给出修复方案。供分析 MaaNTE issue、日志包、识别失败、任务卡死、控制器差异、Pipeline/Agent/MXU 问题时使用。
---

# MaaNTE Issue Log Analysis

## Scope

- 用于上游公开仓库 `https://github.com/1bananachicken/MaaNTE`。
- 输入可以是完整 issue URL，或 `#1234` 形式的 issue 编号。
- 只分析公开 issue 中可直接访问的附件。
- 如果 issue 没有日志包，要先明确说明证据不足，再尽量基于 issue 文本、截图、代码和文档给出初步判断。

## Workflow

1. 规范化输入。

    - `#1234` 视为 `https://github.com/1bananachicken/MaaNTE/issues/1234`
    - 如果不是 `1bananachicken/MaaNTE`，停止并说明此 skill 不适用。

2. 获取 issue 内容。

    - 优先读取 issue 页面正文和评论。
    - 提取这些信息：版本、控制器类型、任务名、预期行为、实际行为、复现步骤、维护者评论。
    - 如果维护者已经给出结论，不要直接照抄；仍要用日志和代码自行验证，再把维护者结论作为补强证据。

3. 提取日志附件链接。

    - 关注 `MaaNTE-logs-*.zip` 或类似的日志压缩包。
    - 如果同一个 issue 有多个日志包，先看最新一次复现；如果 issue 在对比不同版本或不同控制器，再补看前面的包。

4. 下载并解压日志包。

    - 用 `curl -L` 或等价方式下载到仓库内临时目录，例如 `.cache/issue-logs/issue-<number>/`。
    - 解压后用文件工具读取，不要把整份大日志完整塞进回复。
    - 先列一遍解压目录，不要假定结构固定。日志包可能包含：
        - 多份 `mxu-agent-<index>-<pid>.log`
        - 多天的 `mxu-web-YYYY-MM-DD.log`
        - Config 目录
        - `on_error/` 截图目录
        - `.dmp` 崩溃转储文件
    - 如果发现 `.dmp` 文件，记录其文件名和路径，转入步骤 5。

5. **DMP 崩溃转储分析（如有 `.dmp` 文件则必须执行此步骤）。**

    如果在日志包内或 issue 附件中发现了 `.dmp` 文件，**立即读取 `.claude/skills/dmp-analysis/SKILL.md` 并严格按其流程执行**。不要跳过这一步，不要只凭日志文本猜测崩溃原因。

6. 建立时间线。

    - 先从 issue 文本判断"用户觉得出问题的时刻"。
    - 再把 `mxu-web-*`、`mxu-tauri.log`、agent 日志、`maa.log`、`mxu-agent*.log` 串成一条时间线。
    - 先用 `mxu-tauri.log` 或 `maa.log` 找本次提交的 `task_id`，因为一个日志包里经常混有很多历史运行。
    - 如果有 `on_error/` 截图，用它校验当时实际停留画面；如果没有，要检查是否是未触发 `on_error`，还是日志导出因体积限制把图片截断了。

7. 回溯到代码和文档。

    - 任务入口、节点名、控制器限制先看 MaaNTE 仓库。
    - Pipeline 运行语义不确定时查 MaaFramework 文档。
    - MXU 行为或日志分层不确定时，先查 MXU README / 文档；只有文档不足或证据已指向实现层时才看源码。
    - 输出给用户时，如果提到任务名、任务说明、选项名、提示文案，先到 `assets/resource/locales/interface/zh_cn.json` 查中文文案，不要直接把 `task id` / `option id` 当成最终展示文本。

8. 只有在满足条件时才下钻第三方仓库。

    - 先用 issue、日志、MaaNTE 仓库和 MaaFramework 文档做初步归因。
    - 如果怀疑问题在 MXU、MaaFramework 或 binding 实现层，且现有证据不足以确认，再按需查看对应上游仓库源码。
    - 常见对应关系：
        - `MXU`：`https://github.com/MistEO/MXU`，前端配置、Tauri 后端编排、实例/任务/agent 生命周期、`mxu-tauri.log` / `mxu-web-*`
        - `MaaFramework`：`https://github.com/MaaXYZ/MaaFramework`，Pipeline 运行时、控制器、资源加载、任务调度、`maa.log`
    - 本地没有时再 clone 到临时目录，例如 `.cache/upstream-src/<repo>/`。

## Log Map

### `maa.log`

- 模块归属：`MaaFramework` 核心运行时。
- 主要内容：资源加载、控制器连接、任务启动、节点识别、动作执行、超时、回调细节、C++ 源文件和函数名。
- 最适合看：
    - Pipeline 是否按预期推进
    - `next` / `on_error` 是否命中
    - 识别算法细节（模板分数、OCR 结果、动作成功失败）
    - 控制器、截图、资源加载层面的异常

### Agent 日志（Python 侧）

- 模块归属：`agent/custom/action/` 下的 Python 自定义动作。
- 主要内容：通过 `utils.logger` 输出的结构化日志，包含自定义动作执行细节、参数解析、识别结果。
- 最适合看：
    - Python 自定义逻辑是否触发
    - 参数解析是否正确
    - 自定义识别/动作的执行流程
- 日志级别和格式见 [maa-logging](../maa-logging/SKILL.md)。

### `mxu-tauri.log`

- 模块归属：MXU 的 Tauri/Rust 后端。
- 主要内容：实例创建、控制器连接、资源加载、任务开始/停止、agent 生命周期。
- 最适合看：
    - MXU 是否正确把 UI 操作转成 MaaFramework/Agent 调用
    - 任务有没有被用户手动停止
    - agent 是否成功启动、断开、退出

### `mxu-web-YYYY-MM-DD.log`

- 模块归属：MXU 的 React/TypeScript 前端。
- 主要内容：`interface.json` 加载、配置读写、任务列表解析、控制器选择、启动参数、高层 UI 流程日志。
- 最适合看：UI 是否加载了正确配置、用户点击后提交了什么任务和配置。

### `mxu-agent*.log`

- 模块归属：MXU 捕获到的 agent 子进程标准输出/标准错误。
- 主要内容：子进程控制台输出、MaaFramework stdout 内容、面向 MXU 运行日志面板的消息。
- 最适合看：用户在 MXU 运行日志面板里实际看到了什么。
- 注意：它是很有用的用户视角，但不是最权威的根因日志。涉及具体运行细节时，优先以 `maa.log` 和 agent 日志为准。

### `config/*`

- 模块归属：MXU 配置快照。
- 常见文件：`config/mxu-MaaNTE.json`、`config/maa_option.json`
- 最适合看：实际启用了哪些任务和选项、配置是否与用户描述一致。

### `on_error/`

- 模块归属：MaaFramework 错误现场截图。
- 最适合看：实际停留界面、是否被弹窗、加载态、遮罩、分辨率等环境因素干扰。
- 当日志和 issue 文字描述冲突时，优先相信现场截图。

## How To Filter Evidence

1. 先从 issue 文本拿到锚点：版本、控制器类型、任务名/入口名、用户说的"卡住/点错/识别失败"的画面。

2. 再从日志里找高价值信号：
    - `Tasker.Task.Starting` / `Succeeded` / `Failed`
    - `Node.Recognition.Failed` 连续重复
    - `Node.Action.Failed`
    - `timeout`
    - `Warn` / `Error` / `Fatal`
    - agent 启动失败、断连、被停止

3. 先锁定"这一次复现"的任务实例，再看细节：
    - 从 `mxu-tauri.log` 找 `post_task returned task_id`
    - 再到 `maa.log` 用这个 `task_id` 跟完整个任务
    - 如果 issue 文本说"失败"，但目标 `task_id` 实际 `Tasker.Task.Succeeded`，要明确写出"本日志未复现用户描述的失败"

4. 对 Pipeline 问题，重点看：`maa.log`、`mxu-tauri.log`

5. 对 Python 扩展问题，重点看：agent 日志、`mxu-agent*.log`

6. 对 UI / 配置 / 编排问题，重点看：`mxu-web-*`、`mxu-tauri.log`、`config/*`

## Common Patterns

- `next` 列表中的识别连续失败直到超时：常见于当前画面不在预期分支中、模板/OCR 失配、漏了中间节点、被弹窗打断。
- 某个"兜底返回/退出"节点连续成功但流程没有前进：常见于 Pipeline 对当前状态判断错了。
- 用户日志与当前主线代码不一致，且主线已修复：先确认用户版本，必要时切到对应 tag 核对旧逻辑。
- issue 文字说"卡死/误点"，但对应 `task_id` 最终 `Tasker.Task.Succeeded`：先明确"本次日志没有复现出用户描述的问题"，再分析可能性。

## Correlating With Code

### MaaNTE

- 任务入口、选项定义：`assets/resource/tasks/*.json`
- 资源版本信息：`assets/interface.json`
- Pipeline 节点：`assets/resource/base/pipeline/**/*.json`
- Python 自定义动作：`agent/custom/action/**/*.py`
- 本地化文案：`assets/resource/locales/interface/zh_cn.json`

### MaaFramework

- Pipeline 协议：`https://github.com/MaaXYZ/MaaFramework/raw/refs/heads/main/docs/en_us/3.1-PipelineProtocol.md`
- interface.json / agent / controller 语义：`https://github.com/MaaXYZ/MaaFramework/raw/refs/heads/main/docs/en_us/3.3-ProjectInterfaceV2.md`

### MXU

- 日志分层、GUI 角色、架构：`https://raw.githubusercontent.com/MistEO/MXU/main/README.md`

## Linking Code Evidence

- 统一给出对应仓库的远端 GitHub `blob` 行号链接，用尖括号包裹。
- MaaNTE 仓库链接格式：`https://github.com/EeeMaoY/MaaNTE/blob/<commit>/<path>#L1-L2`
- `<commit>` 必须是本次分析实际依据的代码版本。
- 如果引用 MXU、MaaFramework 等上游仓库，也用对应远端 `blob` 链接。

## Output Format

最终回答用这个结构：

````markdown
## Issue 概要

- issue：`#1234`
- 版本 / 控制器：优先写 `zh_cn` 中文任务名
- 任务 / 相关选项：优先写 `assets/resource/locales/interface/zh_cn.json` 中的中文 label/description
- 用户现象：

## 关键证据

<details><summary>点击此处展开</summary>

- `maa.log`：...
- agent 日志：...
- `mxu-tauri.log`：...
- `mxu-web-*.log`：...
- `mxu-agent.log` / `on_error`：...
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
````

## Reminders

- **DMP 堆栈必须完整输出**：如果 issue 存在 `.dmp` 文件，最终报告中必须包含 `## DMP 崩溃分析` 区域，禁止省略堆栈帧。
- **DMP 分析必须先读 skill**：发现 `.dmp` 时必须先读取 `.claude/skills/dmp-analysis/SKILL.md` 再动手。
- 不要只看一个日志文件下结论。
- 不要把"维护者评论"当成唯一证据。
- 不要把环境告警自动等同于根因。
- 如果结论是"功能不支持"，必须给出代码级依据。
- 如果回答里出现任务名、任务描述、选项名、提示文案，优先使用 `assets/resource/locales/interface/zh_cn.json` 的中文文案。
- 如果回答里引用了具体代码行，直接给远端 GitHub `blob` 行号链接，用尖括号包裹。
- 如果日志和 issue 文字描述不一致，必须显式说明"证据未复现"还是"证据已复现但用户表述不精确"。
