# PRD：GeForce NOW 云游戏窗口支持

> [!NOTE]
> 本文档为产品需求文档（PRD），描述需求与设计约束，不包含最终实现。
> 状态：**草案**。涉及"待实测验证"的条目需在实现前于真实环境确认。

## 1. 背景与目标

### 1.1 背景

MaaNTE 目前只支持本地 Windows 客户端（进程 `HTGame.exe`，Unreal 窗口类 `UnrealWindow`，标题 `异环` / `NTE`）。而部分用户通过 **GeForce NOW（GFN）云游戏** 游玩 NTE，存在两种形态：

1. **Chrome 网页客户端** — Chrome 应用式窗口，进程 `chrome.exe`，窗口类 `Chrome_WidgetWin_1`，标题 **"NTE: Neverness to Everness on GeForce NOW"**（已由 `screenshot/NTE window name.png` 实测确认）。
2. **GFN 原生桌面客户端** — 进程 `GeForceNOW.exe`（基于 CEF）。该窗口对常规缩放操作有抵抗行为，社区存在专用工具 [GeForceNowWindowMover](https://github.com/Th3C0D3R/GeForceNowWindowMover) 用于强制移动/缩放。

当前两条窗口发现路径均无法命中 GFN 窗口：

- **MaaFramework 控制器**（`assets/interface.json`）：`class_regex` 只匹配 `UnrealWindow`，`window_regex` 只匹配 `异环|NTE`。
- **Agent 辅助查找器**（`agent/utils/win32_process.py`）：硬编码 `DEFAULT_GAME_PROCESS_NAME = "HTGame.exe"`，且按"最大客户区面积"选窗、无标题过滤 —— 对 Chrome 完全不可用（`chrome.exe` 存在数十个进程，且用户日常浏览窗口会赢得面积启发式）。

### 1.2 目标

- 用户使用 GFN（Chrome 网页版或原生客户端）游玩 NTE 时，MaaNTE 能自动找到游戏窗口并正常连接控制器。
- Agent 辅助查找器（分辨率检测、窗口缩放）在 GFN 场景下正常工作或优雅降级。
- 对 GFN 场景的能力边界（前台限制、缩放限制）给出明确的用户提示。

### 1.3 非目标

- 不支持 GFN 移动端 / TV 端。
- 不自动化 GFN 自身的登录、排队、会话续期流程（这些属于流程外状态，见 §6 风险）。
- 本期不覆盖 Edge 及其他 Chromium 系浏览器（见 §8 后续工作）。

## 2. 现状分析

### 2.1 主路径：MaaFramework Win32 控制器

`assets/interface.json` 声明三个 Win32 控制器变体，由 GUI 壳（MFAA/MXU）传给 MaaFramework 的 `MaaToolkitDesktopWindowFindAll` 完成窗口枚举与正则匹配：

```jsonc
"win32": {
    "class_regex": "UnrealWindow",
    "window_regex": "^\\s*(异环|NTE)\\s*$",
    "screencap": "Background",
    "mouse": "SendMessageWithCursorPos",
    "keyboard": "PostMessage"
}
```

GFN 窗口的类名与标题均不匹配上述正则，因此控制器列表中永远找不到 GFN 会话。

### 2.2 辅助路径：Agent 自有查找器

`agent/utils/win32_process.py` 的调用链：

```text
get_pids_by_name("HTGame.exe")          # 进程快照取 PID 集合
  → find_windows_by_process(...)        # EnumWindows + PID/可见性/类名/客户区过滤
  → find_window_by_process(...)         # 选择器：selected_hwnd > last_hwnd(滞回) > 最大客户区
```

使用方：

- `agent/main.py` 的 `_check_game_resolution()` — Agent 启动时的分辨率检测告警；
- `agent/custom/action/Common/resize_game_window.py` — `resize_game_window` 自定义动作，经 `ensure_game_window_resolution()` 把客户区强制为 1280x720。

对 GFN 的失效点：

| 失效点 | 说明 |
| --- | --- |
| 进程名硬编码 | 只找 `HTGame.exe`，GFN 场景下游戏进程在云端，本地只有 `chrome.exe` / `GeForceNOW.exe` |
| 无标题过滤 | `find_windows_by_process` 仅有布尔 `require_title`，无法用正则区分"GFN 游戏窗口"与"普通 Chrome 窗口" |
| 面积启发式误选 | 用户日常浏览的 Chrome 主窗口客户区往往大于 GFN 游戏窗口，"最大客户区"会选错 |

## 3. 目标窗口特征

| 目标 | 进程名 | 窗口类 | 标题模式 | 缩放行为 |
| --- | --- | --- | --- | --- |
| 本地客户端（现状） | `HTGame.exe` | `UnrealWindow` | `^\s*(异环\|NTE)\s*$` | `SetWindowPos` 正常生效 |
| GFN Chrome 网页版 | `chrome.exe` | `Chrome_WidgetWin_1` | `NTE: Neverness to Everness on GeForce NOW`（已实测确认） | 外框可正常缩放，但窗口模式下页面自绘头部破坏帧几何（见 R8），实际要求 F11 全屏运行 |
| GFN 原生客户端 | `GeForceNOW.exe` | `CEFCLIENT`（已实测确认，EnumWindows 运行时取证；社区旧文档的 `CEF-OSC-WIDGET` 为老版本客户端） | `NTE: Neverness to Everness on GeForce NOW`（已实测确认，与 Chrome 版一致） | 接受标准 `MoveWindow`/`SetWindowPos` 外部缩放（GFNWindowMover 即用此方式），流窗口无边框、无页面头部；客户区调至 1280x720 可获得干净 720p 帧 |

> [!NOTE]
> 关于"用 Chrome 进程找窗口是否可行"：**可行，但有前提。**
> Chrome 为多进程架构（渲染、GPU、工具进程等数十个 `chrome.exe`），但**只有主浏览器进程拥有可见的顶层窗口**。现有 `get_pids_by_name` → PID 集合过滤 → `EnumWindows` 的流程天然兼容多进程，无需修改。
> 真正的缺口是**选择性**：`chrome.exe` + `Chrome_WidgetWin_1` 会命中用户所有 Chrome 窗口，必须新增标题正则过滤，且面积启发式只能在标题命中的候选集内生效（见 FR2）。

## 4. 功能需求

### FR1 — 新增 GFN 控制器条目（`assets/interface.json`）

新增两个控制器：

```jsonc
{
    "name": "GFN-Chrome",
    "label": "$controller_gfn_chrome_label",
    "type": "Win32",
    "win32": {
        "class_regex": "Chrome_WidgetWin_1",
        "window_regex": "NTE.*on GeForce NOW",
        "screencap": "PrintWindow",
        "mouse": "Seize",
        "keyboard": "Seize"
    }
},
{
    "name": "GFN-App",
    "label": "$controller_gfn_app_label",
    "type": "Win32",
    "win32": {
        "class_regex": "CEFCLIENT",
        "window_regex": "NTE.*on GeForce NOW",
        "screencap": "PrintWindow",
        "mouse": "Seize",
        "keyboard": "Seize"
    }
}
```

- 首选前台模式（`PrintWindow` + `Seize`）：`SendMessage`/`PostMessage` 后台注入对 Chrome/CEF 的 GPU 合成窗口普遍不可靠（见 §6 风险 R1）。若实测证明某种后台组合可用，可增补后台变体。
- 键盘输入必须能透传到 GFN 串流（浏览器把按键转发到云端主机），WASD 等长按键位为验证重点。
- 配套新增 i18n 标签键 `controller_gfn_chrome_label` / `controller_gfn_app_label`，同步 5 个 locale 文件（`assets/resource/locales/interface/{zh_cn,zh_tw,en_us,ja_jp,ko_kr}.json`）。

### FR2 — 查找器新增标题正则过滤（`agent/utils/win32_process.py`）

- `find_windows_by_process()` 与 `find_window_by_process()` 新增 `title_regex=None` 参数，向后兼容（默认不过滤，现有调用方行为不变）。
- 过滤在 `EnumWindows` 回调内与类名过滤同层执行：标题不匹配的窗口直接不进入候选集。
- 选择器（`selected_hwnd` > `last_hwnd` 滞回 > 最大客户区）逻辑不变，但**只作用于标题命中后的候选集**，从根上消除"选中用户浏览窗口"的误选。
- 复用现有 `_match_class_name` 的"字符串精确 / 正则搜索"双模式风格实现标题匹配。

### FR3 — Agent 启动时的窗口自动探测（`agent/main.py`）

`_check_game_resolution()` 按优先级探测：

1. `HTGame.exe`（现状逻辑，保持不变）；
2. `chrome.exe` + `title_regex="NTE.*on GeForce NOW"` + `hwnd_class="Chrome_WidgetWin_1"`；
3. `GeForceNOW.exe` + 实测确认后的类名/标题。

- 探测结果（`native` / `gfn_chrome` / `gfn_app` / `not_found`）记录为模块级状态（遵循仓库"模块全局变量 + 独立 `_reset` 动作"约定），供后续自定义动作查询运行模式。
- 探测到 GFN 模式时输出 INFO 级日志说明当前运行形态。

### FR4 — `resize_game_window` 动作的 GFN 行为（`agent/custom/action/Common/resize_game_window.py`）

| 运行模式 | 行为 |
| --- | --- |
| 本地客户端 | 现状不变：`ensure_game_window_resolution()` 强制客户区 1280x720 |
| GFN Chrome | 走 `resize_client_area()` 常规路径；但窗口模式下页面头部仍破坏帧几何（R8），缩放无法根治，实际引导用户 F11 全屏 |
| GFN 原生客户端 | **自动缩放**：走 `resize_client_area(manage_title_bar=False)`，保持无边框、不强制 `WS_CAPTION`（GFNWindowMover 源码证实 GFN 窗口接受标准 `MoveWindow` 缩放）。缩放未生效时优雅降级（`reason=gfn_app_resize_failed`，任务不中断），提示用户在 GFN 设置固定 720p 串流或用窗口工具调整 |

### FR5 — 用户可见提示（`utils/maafocus`）

新增 PrintT 消息键（含 5 语言翻译）：

- 检测到 GFN 窗口时的模式提示（"当前通过 GeForce NOW 运行，仅支持前台模式"）；
- GFN 分辨率不匹配告警（引导设置 720p 串流）；
- GFN 原生客户端缩放跳过说明。

禁止 `print()`，调试细节走 `utils.logger`。

## 5. 非功能需求

- **零新增 Python 依赖**：全部基于现有 `ctypes` Win32 封装扩展。
- **平台守卫**：所有 Win32 逻辑维持 `sys.platform` 守卫（参照 `resize_game_window.py` 现有写法）。
- **日志规范**：遵循 `maa-logging` 约定 —— `%` 风格格式化、DEBUG 记探测细节、WARNING 记降级路径。
- **向后兼容**：本地客户端用户的行为与性能不受任何影响；新参数全部带默认值。

## 6. 风险与开放问题

| # | 风险 / 开放问题 | 影响 | 应对 |
| --- | --- | --- | --- |
| R1 | Chrome/CEF 的 GPU 合成窗口对 `SendMessage`/`PostMessage` 后台注入普遍不可靠 | GFN 用户大概率**只能前台运行**，无法使用后台任务 | PRD 即明确 GFN 控制器为前台模式；文档与 UI 提示中声明该限制 |
| R2 | 云端串流的压缩伪影/码率波动可能拉低 TemplateMatch 匹配分 | 识别节点在 GFN 下命中率下降 | 验收阶段用代表性任务实测；必要时对少量模板放宽 `threshold` 或补充 GFN 专用模板 |
| R3 | ~~GFN 原生客户端窗口类未确认~~ **已关闭**：运行时 EnumWindows 取证确认为 `CEFCLIENT`（`GeForceNOW.exe`，pid 级验证）。注意社区旧文档的 `CEF-OSC-WIDGET` 与 Electron 的 `Chrome_WidgetWin_1` 均不适用于当前客户端 | — | 探测代码保留 `class=` 日志字段，便于未来客户端更新后再次核对 |
| R4 | 窗口标题可能随 GFN 客户端语言变化 | `window_regex` 漏匹配 | 优先匹配语言无关片段（游戏英文名 + `GeForce NOW`）；收集多语言标题样本 |
| R5 | 用户同时开多个 Chrome 窗口（含多个 GFN 页签）的极端情况 | 选窗歧义 | FR2 的标题过滤 + 现有 `selected_hwnd`/滞回机制兜底；GUI 侧用户可手动选窗 |
| R6 | GFN 自身的排队、闲置踢出、会话到期画面 | 任务流程外状态，Pipeline 无法恢复 | 非目标（§1.3）；提示用户保持会话活跃，长任务失败时日志可定位 |
| R7 | ~~任务级控制器限制与新控制器名的兼容性~~ **已关闭**：已梳理全部 12 个限定控制器的任务。GFN 两控制器与 `Win32-Front` 模式一致（PrintWindow + Seize），10 个任务已在配置中放开 `GFN-Chrome`/`GFN-App`（PinkPawHeist、MakeCoffee、MakeCoffeeLite、Furniture、BagelSpam、SoundDodge、RealTime、SyncCharacterAbilityCityAbility、ClaimRewards、WithdrawMoney）；`FountainCheckin`、`WitchDivination` 保持仅 `Win32-Front`——两者依赖抓包定位坐标（`Navi/coordinate_position.py`，pcap/pktmon），GFN 场景下游戏运行在云端、本机无游戏流量，原理上不可用 | — | 新增任务时按 `.claude/skills/task-config/SKILL.md` 的控制器限制规则声明 |
| R8 | **（实测确认）** GFN Chrome 网页版窗口模式下，页面自绘头部（标题条）占据客户区顶部约 26px，游戏视频被下移且按比例缩放，所有固定 ROI 识别失败（实测 InWorld 失败 5563 次、SceneManager 连按 ESC 569 次死循环） | GFN Chrome 窗口模式完全不可用 | 运行前提：16:9 显示器 + F11 全屏（头部消失、视频铺满、帧缩放为干净 1280x720）+ Windows 缩放 100%；文档与 UI 提示中声明 |

## 7. 验收标准

- [ ] 在同时打开 ≥3 个普通 Chrome 窗口的环境中，`GFN-Chrome` 控制器能唯一命中 GFN 游戏窗口。
- [ ] 控制器连接成功，截图分辨率为 1280x720，鼠标/键盘输入在串流中生效（含 WASD 长按）。
- [ ] Agent 启动探测顺序正确：本地客户端优先，其次 GFN Chrome，最后 GFN 原生客户端；探测结果日志清晰。
- [ ] `find_window_by_process` 增加 `title_regex` 后，现有调用方（`HTGame.exe` 路径）行为回归一致。
- [ ] 代表性任务（如 `MakeCoffee`、`PinkPawHeist`）在 GFN 下端到端跑通。（原定的 `FountainCheckin` 依赖抓包定位坐标，GFN 下原理上不可用，见 R7）
- [ ] GFN 原生客户端路径优雅降级：跳过缩放、输出明确的用户引导消息，任务不因缩放失败而崩溃。
- [ ] 5 个 locale 文件的新增键完整同步，`pnpm exec prettier --check` 通过。

## 8. 后续工作

按仓库 PR 约定拆分实现（全部合入 `dev`，分支命名 `feat/<name>`）：

1. `interface.json` 新增控制器 + 5 locale 标签；
2. `win32_process.py` 标题过滤扩展 + `main.py` 探测顺序；
3. `resize_game_window` GFN 行为 + maafocus 消息；
4. 文档更新（本 PRD 回填实测数据、用户使用说明）。

未来扩展：Edge 及其他 Chromium 系浏览器（同窗口类、不同进程名，FR2 的进程名列表参数天然支持）；GFN 原生客户端强制缩放方案（若用户反馈需要）。
