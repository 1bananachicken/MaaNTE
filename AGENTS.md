# MaaNTE - 智能体编码指南

## 构建 / 检查 / 测试

```bash
# 环境初始化
git submodule update --init --recursive
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 构建发行包（MaaFramework + MXU）
python build.py                          # 默认：MFAA + MXU
python build.py --mode=mxu               # 仅 MXU
python build.py --mode=mfaa              # 仅 MFAA
python build.py --skip-download          # 跳过下载，仅本地组装
python build.py --compress=false         # 跳过压缩打包
python build.py --tag v1.0.0             # 指定版本号

# Python 语法校验（尚无 pytest 测试套件）
python -m py_compile agent/custom/action/**/*.py

# 格式化（Prettier 用于 JSON/YAML）
pnpm install
pnpm exec prettier --check .
pnpm exec prettier --write .

# i18n 同步
.github/workflows/i18n-sync.yml         # CI 工作流，自动同步 OCR 文本翻译

# 空白字符检查
git diff --check

# JSON schema 校验（VS Code 内置，schema 位于 deps/tools/）
# Pipeline JSON    -> deps/tools/pipeline.schema.json
# interface.json   -> deps/tools/interface.schema.json
# tasks/*.json     -> deps/tools/interface_import.schema.json
```

> **注：** 自动化测试（`node-testing.md`）尚未就绪。单节点测试通过 VS Code "Maa Pipeline Support" 插件手动执行（右键运行）。

---

## 仓库目录结构

```
agent/                         # Python 自定义逻辑
  custom/action/               #   CustomAction / CustomRecognition 类
  utils/                       #   日志、i18n、maafocus、屏幕、时间工具
assets/resource/               # 游戏资源配置
  base/pipeline/               #   Pipeline JSON（场景、任务、通用按钮）
  base/image/                  #   模板图片（1280x720 基准）
  locales/interface/           #   5 个 i18n 文件：zh_cn, zh_tw, en_us, ja_jp, ko_kr
  tasks/                       #   任务配置 JSON（选项定义）
assets/interface.json          #   全局导入注册表（任务 + 资源）
docs/zh_cn/develop/            #   正式开发者文档
install-mxu/                   # MXU 运行时构建（独立的 Python 环境副本）
```

---

## Pipeline JSON 约定

- **V2 格式：** `"recognition": {"type": "...", "param": {...}}, "action": {...}`
- **命名：** 帕斯卡命名，任务/模块前缀（如 `FishNewEntrance`）。私有节点用 `__` 前缀。
- **坐标 / ROI / 图片 / target** 均基于 **1280x720**。
- **避免硬延迟**（`pre_delay`/`post_delay`/`timeout`）。优先使用中间识别节点或 `pre_wait_freezes`/`post_wait_freezes`。
  不需要时需显式设为 `rate_limit: 0`、`pre_delay: 0`、`post_delay: 0`（协议默认值为 1000ms/200ms/200ms）。
- **`next` 首轮命中原则：** `next` 列表应尽可能覆盖所有可能的画面状态。拒绝一切形式的重试机制，力争一次流程完成所有任务。
- **识别 -> 动作 -> 重新识别** 循环。禁止"识别一次，然后连续点 A、B、C"。
- **禁止盲目重试 / 滥用 `max_hit`。** 必须找到根因。
- **推荐处理弹窗和加载：** 好的流程不只走主线，也应处理弹窗、加载、不在目标场景等中间态。参考在 `next` 中添加 `[JumpBack]SceneAnyEnterWorld`、`[JumpBack]SceneClickBlankToExit`、`[JumpBack]SceneLoading`。
- **OCR：** `expected` 填写完整文本；CI 自动同步翻译。部分匹配/正则的条目加 `// @i18n-skip` 注释。
- **识别算法类型：** `TemplateMatch`（图片 + 阈值 + ROI）、`OCR`（全文 + 阈值 ~0.3）、`ColorMatch`（RGB 距离 method:40 + lower/upper + count + connected）、`DirectHit`、`And`/`Or`（复合）、`Custom`（Python 实现）。
- **动作类型：** `Click`、`LongPress`、`Swipe`、`ClickKey`、`Custom`、`DoNothing`、`StopTask`。
- **优先使用 SceneManager 公开接口**（`Interface/Scene/`）。禁止直接引用 `__ScenePrivate*`。

---

## Python CustomAction / CustomRecognition 约定

- **仅用于 Pipeline 无法表达的逻辑。** "Pipeline 管流程，Python 管难点。"
- **注册：** `@AgentServer.custom_action("snake_case_name")`（动作）或 `@AgentServer.custom_recognition("snake_case_name")`（识别）。名称必须与 Pipeline 的 `custom_action` / `custom_recognition` 一致。
- **导入注册：** 在 `agent/custom/action/__init__.py` 中添加 import + `__all__` 条目。
- **用户可见消息：** `from utils.maafocus import PrintT` → `PrintT(context, "key", arg1, ...)`。禁止 `print()`。
- **调试日志：** `from utils.logger import logger` → `logger.debug()`、`logger.warning()`、`logger.error()`。使用 `%` 风格格式化字符串，不用 f-string。
- **长循环：** 每次迭代检查 `context.tasker.stopping`。
- **坐标：** 全部基于 1280x720。
- **错误处理：** 返回 `CustomAction.RunResult(success=False)` 或 `None` 的 `CustomRecognition.AnalyzeResult`。优雅地捕获异常。
- **图片获取：** `controller.post_screencap().wait()` → `controller.cached_image`（numpy BGR 数组）。
- **通用工具：** `from Common.utils import get_image, click_rect, match_template_in_region`。
- **模块级状态：** 使用模块全局变量，配合独立的 `_reset` 动作。

---

## 任务配置 JSON 约定

- 位置：`assets/resource/tasks/<TaskName>.json`。在 `assets/interface.json` 的 `import` 数组中注册。
- 结构：`{"task": [{name, label, entry, description, option}], "option": {...}}`。
- 选项类型：`switch`（是/否 + `cases`）、`input`（含 `verify` 正则 + `pipeline_type`）、`select`（下拉框 + `items`）。
- `pipeline_override` 使用 `"{value}"` 模板替换用户值（如 `"count": "{count}"`）。
- 控制器限制：可选 `"controller": {"type": "Win32-Front"}`。
- 配套修改清单：任务 `.json` + pipeline `.json` + 5 个 locale 文件 + `interface.json`。

---

## Git 约定

- **禁止直接改动 `dev` 分支。** 所有改动必须在独立分支上进行，通过 PR 合并。
- **禁止 AI 自行 merge `dev`** 到功能分支或在分支间合并，除非人工明确要求。
- **分支目标：** 所有 PR 合并到 `dev`。分支命名：`feat/<name>`、`fix/<name>`、`docs/<name>`、`refactor/<name>`、`chore/<name>`。
- **提交信息：** Conventional Commits 格式：`<type>(<scope>): <subject>`。
  类型：`feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、`chore`、`revert`、`ci`。
- **PR 标题：** Conventional Commits 格式。使用 Draft PR（不要 `WIP` 前缀）。
- **PR 描述：** 关联 Issue（`Closes`/`Fixes`/`Related`）、变更摘要（2-5 条）、验证记录、截图/日志。

---

## i18n 约定

- 五个语言文件：`zh_cn.json`、`zh_tw.json`、`en_us.json`、`ja_jp.json`、`ko_kr.json`，位于 `assets/resource/locales/interface/`。
- 标签使用 `$key` 格式。OCR `expected` 填写完整中文文本；`.github/workflows/i18n-sync.yml` 自动同步。
- 新增/修改任务时，5 个语言文件必须同步更新。

---

## Visual Studio Code 配置

- **格式化器：** Python → `ms-python.black-formatter`（保存时自动格式化）。JSON/YAML → Prettier（保存时自动格式化）。Markdown → `DavidAnson.vscode-markdownlint`。
- **Pipeline JSON schema** 通过 `.vscode/settings.json` 自动关联。
- **推荐插件：** Maa Pipeline Support（单节点测试）。

---

## 新增功能时需修改的文件

| 场景 | 涉及文件 |
|---|---|
| 新增任务选项 | `assets/resource/tasks/*.json` + pipeline JSON + 5 个 locale 文件 + `interface.json` |
| 新增 Pipeline 节点 | `assets/resource/base/pipeline/**/*.json` |
| 新增 Python 动作 | `agent/custom/action/<name>.py` + `agent/custom/action/__init__.py` |
| 新增场景 | `assets/resource/base/pipeline/Interface/Scene/Status.json` + `SceneManager/` 私有节点 |
