# 实时辅助

## 简介

这是一个**实时辅助**功能，会无限次循环运行。

需要使用前台控制器。~~（好像是废话）~~

## 功能

### 检测间隔

每轮检测之间等待的时间，单位为毫秒。一般情况下无需改动。

### 自动传送

处于地图界面时 ，会在你选中传送点时自动点击“传送”按钮。

目前支持下列传送点：

- “维特海默塔”
- “ReroRero电话亭”

### 自动跳过剧情

当处于可以跳过的剧情时，自动点击跳过按钮。

支持下列功能：

- 自动勾选“今日不再提示”

## 配置详解

### 自动剧情

“[自动跳过剧情](#自动跳过剧情)”功能的总开关。关闭时会同时关闭“自动勾选‘今日不再提示’”功能。

**具体实现**：开关 `RealTimeAutoSkipStory` ，开启时提供 `RealTimeAutoSkipStoryDialog` 子选项入口，将 `RealTimeSkipStory` 、 `RealTimeSkipStoryDialogConfirm` 覆写为 `"enabled": true` ；关闭时将 `RealTimeSkipStory` 、 `RealTimeSkipStoryDialog` 、 `RealTimeSkipStoryDialogConfirm` 、 `RealTimeSkipStoryDialogCheckbox` 覆写为 `"enabled": false` 。

### 自动勾选“今日不再提示”

“自动勾选‘今日不再提示’”功能开关。

### 自动传送

“自动传送”功能的总开关。关闭时会同时关闭所有子传送点的检测。

**具体实现**：开关 `RealTimeAutoTeleport` ，开启时提供 `RealTimeAutoTeleportWitte` 和 `RealTimeAutoTeleportPhone` 子选项入口；关闭时将 `RealTimeTeleportWitte` 、 `RealTimeConfirmTeleportWitte` 、 `RealTimeTeleportPhone` 、 `RealTimeConfirmTeleportPhone` 均覆写为 `"enabled": false` 。

#### 维特海默塔

单独控制是否检测并自动传送至“维特海默塔”传送点。开启后，软件会识别该传送点详情页并自动点击传送按钮。

**具体实现**：开关 `RealTimeAutoTeleportWitte` ，将 `RealTimeTeleportWitte` 、 `RealTimeConfirmTeleportWitte` 覆写为 `"enabled": true` ；关闭时将 `RealTimeTeleportWitte` 、 `RealTimeConfirmTeleportWitte` 覆写为 `"enabled": false` 。

#### ReroRero电话亭

单独控制是否检测并自动传送至“ReroRero电话亭”传送点。开启后，软件会识别该传送点详情页并自动点击传送按钮。

**具体实现**：开关 `RealTimeAutoTeleportPhone` ，将 `RealTimeTeleportPhone` 、 `RealTimeConfirmTeleportPhone` 覆写为 `"enabled": true` ；关闭时将 `RealTimeTeleportPhone` 、 `RealTimeConfirmTeleportPhone` 覆写为 `"enabled": false` 。

### 检测间隔

每轮检查的循环间隔

**具体实现**：`int` 类型输入框 `RealTimeCheckInterval` ，通过 `^\\d+$` 校验数据。覆写 `RealTimeSleep` 的 `post_delay` 参数实现。
