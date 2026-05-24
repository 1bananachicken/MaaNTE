# MaaNTE Issue Log Analysis — Knowledge Base

## Controller Mismatch

- MaaNTE 有三种控制器模式，定义在 `assets/interface.json` 的 `controller` 数组中：
  - **Win32（默认/后台）**：`mouse: SendMessageWithCursorPos`，需管理员权限
  - **Win32-Front（前台）**：`mouse: Seize`，会抢占鼠标，不需管理员权限
  - **Win32-Background（后台备用）**：`mouse: SendMessageWithWindowPos`，需管理员权限
- 不同任务的控制器要求定义在各 `assets/resource/tasks/<Task>.json` 的 `controller` 字段：
  - `RealTime.json:14` → `Win32-Front`
  - `SoundDodge.json:15` → `Win32-Front`
  - `PinkPawHeist.json:12` → `Win32-Front`
  - `Furniture.json:9` → `Win32-Front`
  - `FountainCheckin.json:9` → `Win32-Front`
  - `ClaimRewards.json:9-10` → `Win32` 或 `Win32-Front`
  - `WithdrawMoney.json:12-13` → `Win32` 或 `Win32-Front`
- 用户报告"功能比别人少"或"找不到某个任务"时，先检查控制器是否匹配。
- 主页控制器切换按钮已知有 bug，用户必须在"设置 > 连接设置"中更改。
- 前台控制器（`Seize`）抢占鼠标是 by design，不是 bug。

## Environmental Requirements

- **管理员权限**：Win32 和 Win32-Background 需要。不满足会导致窗口连接失败。
- **游戏分辨率**：必须 `1280x720` 窗口模式。新版钓鱼适配了任意 16:9 分辨率，但其他功能仍以 720p 为基准。见 `docs/zh_cn/问题排查.md`。
- **Windows 缩放**：必须 100%。非 100% 会导致点击坐标全部偏移（MaaFramework 限制）。
- **路径限制**：必须纯英文路径，不允许全角字符和特殊符号（MaaFramework 限制）。见 `docs/zh_cn/问题排查.md`。
- **游戏语言**：必须简体中文。OCR 训练数据仅基于简体中文。
- **.NET 10.0 Desktop Runtime**：启动必需。
- **杀毒软件**：可能阻止窗口连接或任务启动，需加白名单。

## Game Settings Interference

- 帧生成、超分辨率等画质增强功能必须关闭，会干扰模板匹配和 OCR。
- 判断方法：如果 `on_error/` 截图异常 + 日志识别分数在阈值边缘，优先引导检查画质设置。

## Fishing-Specific

- 买鱼饵误识别：鱼饵商店和方斯选项的 ROI 可能重叠。可建议降低 `鱼饵识别阈值`（默认 0.8，最低 0.6）。相关 Pipeline 见 `assets/resource/base/pipeline/Fish/`。
- 卖鱼对帧率敏感，`docs/eng/trouble_shooting.md` 建议 120 FPS。
- 隔夜钓鱼被月卡弹窗打断是已知限制，暂不视为 bug。
- 鱼饵不足导致任务提前结束是 by design。

## Coffee-Specific

- "没有收益/没有全连击" → 需要娜娜莉和白藏的城市技能，是游戏机制不是 MaaNTE bug。`docs/zh_cn/问题排查.md` 中有说明。

## Real-Time Assist

- 传送点支持"维特海默塔"和"ReroRero 电话亭"，定义在 `assets/resource/tasks/RealTime.json` 的 `teleport_witte`、`teleport_phone` 选项，Pipeline 实现见 `assets/resource/base/pipeline/realtime_assist/RealTimeTeleport.json`。
- "无法跳过重要剧情提示"是已知的功能缺失。
- 依赖 F1/F2/F5 快捷键，用户修改游戏键位会导致功能失效。

## PinkPaw Heist

- 被标记为"极不稳定"（见 `docs/zh_cn/introduction/` 中相关介绍文档），只有一个方案且需要特定队伍（娜娜莉、薄荷、早雾）。
- 容错超时后退出至主菜单是 by design。

## Log Collection Timing

- 日志文件在 MaaNTE 运行时被锁定，用户必须在关闭 MaaNTE 后收集。
- MaaNTE 每次启动会清理旧日志（逻辑在 `agent/main.py`）。如果用户复现后重启了 MaaNTE，关键日志可能已永久丢失。
- `maa.bak.log` 和 `debug/custom/` 中前一天的文件可能保存了被清理前的日志。

## Multiple Clients

- MaaNTE 支持 MXU、MaaPiCli、MFAAvalonia 三种客户端。
- 不同客户端产生的日志文件不同：
  - MXU：有 `mxu-tauri.log`、`mxu-web-*`、`mxu-agent*`、`config/mxu-MaaNTE.json`
  - MaaPiCli：有 `config/maa_pi_config.json`，无 MXU 系列日志
- 如果日志包缺乏 MXU 系列文件，先确认用户使用的客户端，不要当成"日志不完整"。

## Guardrails for Future Analysis

- 不要把维护者评论、机器人评论、或单张截图当成最终结论；必须回到日志和代码确认。
- 对控制器相关问题，先回答"用户当前使用的控制器是否与任务要求匹配"，再回答"如果不匹配，是否是 UI 引导问题"。
- 如果 `maa.log` 中任务最终 `Succeeded` 但用户抱怨"没效果"，先确认用户是否理解任务预期行为（可能是游戏机制理解偏差）。
- Python agent 日志用 `%`-style 格式化而非 f-string（详见 `maa-logging` skill），不要因格式化风格怀疑日志质量。
