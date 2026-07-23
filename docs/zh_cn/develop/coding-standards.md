# 编码规范

## AI 编程规范

### 禁止无脑使用 AI 开发

- 直接向 AI 下达笼统指令，例如「帮我开发 xxx 功能并提 PR」「帮我修这个 bug 并提 PR」。
- 在关键模块中，用 AI 生成大量难以维护、难以理解的「黑盒」代码。
- 在关键模块中提交自己看不懂、无法掌控的代码。

> [!CAUTION]
> 禁止在未向 AI 提供游戏界面截图、界面跳转逻辑等上下文的情况下，让 AI 直接编写 Pipeline。
> MaaFramework 的 Pipeline 强依赖游戏界面与业务逻辑，缺乏界面信息的 AI 只能依赖幻觉和项目已有代码拼凑，产出代码质量极低。
> 充分的信息至少包括：每个识别节点需提供 `roi` 与模板图片，并说明界面间的跳转关系（从哪个界面、点击什么、跳转到何处）。
> 不满足以上条件的 PR 将被直接关闭。

### 推荐的 AI 开发方式

- 先学习本项目的编码规范，自行做架构设计，或将 AI 的建议作为参考。
- 用 AI 做有目标的增量开发，自行 review 生成代码是否符合预期。
- 确认无误后再提交 PR。

## Pipeline 低代码规范

### 命名：PascalCase

节点名称使用 PascalCase，同一任务内以任务名或模块名为前缀。例如 `FishNewEntrance`、`WithdrawMoneyEntrance`、`TouchDetect`。内部/私有节点以 `__` 开头：`__ScenePrivateWorldEnterBag`。

### 禁止硬延迟

尽可能少使用 `pre_delay`、`post_delay`、`timeout`。通过增加中间识别节点避免盲目 sleep。

只在必须等待画面稳定时使用 `pre_wait_freezes` / `post_wait_freezes`，其他时候应尽量避免延迟。
**不要为了执行稳定而使用延迟，而是通过增加中间节点判断，因为延迟实际上是在掩盖问题，在用户设备存在高延迟时仍然不会稳定。**

### `next` 第一轮即命中

尽可能扩充 `next` 列表，保证任何游戏画面都处于预期中，实现一次截图就命中目标节点。
**项目一般拒绝一切形式的重试机制，一定要保证在一次流程中完成所有任务，除非遇到无法解决的问题。**

### 识别 → 操作 → 再识别

每一步操作都基于识别。

**推荐：** 识别 A → 点击 A → 识别 B → 点击 B

**禁止：** 整体识别一次 → 点击 A → 点击 B → 点击 C

例如：

1. 界面跳转中，需要 识别跳转按钮 → 点击跳转按钮 → 识别界面已跳转完成。
   _你没法保证点完关闭按钮之后画面是否还和之前一样。极端情况下游戏弹出新池子公告，直接点下一个节点可能点到抽卡里去。_
2. 点击会更改账号数据的按钮时，需要 识别提交按钮 → 点击提交按钮 → 识别按钮点击成功。
   _你没法保证每个用户的网络都是顺畅的，点击按钮事件未与服务器成功交互，整个交互界面会卡死不动。_

### 不要盲目重试、添加限制

**推荐：** 遇到 bug 时找问题根因，详细到具体哪个节点失败、哪个识别不符合预期，去修补对应节点的识别、操作问题。

**禁止：** 同样的操作再试一次、盲目添加 `max_hit`。

1. 当点击没反应时再点一次 → 通过 `pre_wait_freezes`、`post_wait_freezes` 等待画面静止，或增加中间节点确认按钮可点击后再执行。
2. 当某个子任务失败后再跑一次 → 重试只能略微提高成功率，并不能根本解决问题，只会让代码变得难以维护。
3. 当一个节点出现死循环后加 `max_hit` → 出现死循环一般是识别问题、逻辑缺陷导致，盲目加 `max_hit` 只会让逻辑中断。

### 处理弹窗和加载

好的流程不是"主线能跑就行"，而是：正常主线能跑、弹窗能处理、加载能等过去、不在目标场景时能自动跳过去。

常见做法是在 `next` 里挂：

- `[JumpBack]SceneAnyEnterWorld`
- `[JumpBack]SceneClickBlankToExit`
- `[JumpBack]SceneLoading`

### OCR 写完整文本

`expected` 写完整文本，不写半截。多语言处理由 CI 工作流自动完成。需要片段或手写正则时添加 `// @i18n-skip` 标记。

### 先复用，再新增

写新节点前，先查已有 Pipeline 是否已有现成能力。优先使用 [场景管理器](./scene-manager.md) 的公共接口，禁止直接引用 `__ScenePrivate*` 内部节点。

## Python CustomAction 规范

Python CustomAction 仅用于处理 Pipeline 难以实现的复杂逻辑。**流程控制由 Pipeline 负责，Python 只处理难点。**

一句话：**Pipeline 管流程，Python 管难点。**

_没有必要的 Python 逻辑会大大增加代码复杂度，造成下一位开发者开发调试极其困难。_

## 配套文件

MaaNTE 里一个功能改动常常不只改一个地方。

### 新增或修改任务

- `assets/resource/tasks/*.json`
- `assets/resource/base/pipeline/**/*.json`
- `assets/resource/locales/interface/zh_cn.json` 等五种语言文件
- `assets/interface.json`（需注册 task 文件）

### 新增 Python CustomAction

- 在 `agent/custom/action/` 下新建 Python 文件
- 在 `agent/custom/action/__init__.py` 中导入并加入 `__all__`

## 资源规范

### 分辨率：720p 基准

所有图片、坐标（`roi`、`target`、`box`）均以 **1280×720** 为基准。MaaFramework 在运行时会根据用户设备自动转换。推荐使用 **Maa Pipeline Support** 插件进行截图和坐标换算。

### HDR / 颜色管理

当被提示 "HDR" 或 "自动管理应用的颜色" 等功能已开启时，不要进行截图、取色等操作，可能导致模板效果与用户实际显示不符。

## 常见坑

| 坑 | 处理 |
|-----|------|
| 模型或依赖目录缺失 | `git submodule update --init --recursive` |
| 直接引用了 `__ScenePrivate*` 节点 | 应引用 `Interface/Scene/` 目录暴露的公共接口节点 |
| 只顾主线，不处理弹窗/加载 | 把弹窗、加载、中间态视为正常情况 |
| 改了任务但没补文案 | 五种语言文件都要添加 i18n key |
| OCR `expected` 写了片段 | 写完整文本，需要跳过时加 `@i18n-skip` |
| 本地能跑但用户不行 | 检查帧率、颜色滤镜、HDR、分辨率是否与基准一致 |
