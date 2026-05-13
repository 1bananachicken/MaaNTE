# 超强音

## 简介

自动演奏音游"超强音"小游戏，支持选曲、自动连打和帧率调节。

## 功能

### 自动选曲

自动选择目标歌曲进行演奏。启用后自动选择默认歌曲（迷星叫），关闭后可手动指定目标歌曲。

### 自动连打

启用后连续演奏指定次数，关闭则仅演奏一次。支持连打直到活力耗尽，或使用固定次数。

### 目标帧率

演奏循环的每秒帧率，影响识别和按键的响应速度。

## 配置详解

### 自动选曲

"自动选曲"功能开关。启用后自动选择"迷星叫"，关闭时需要手动指定目标歌曲。

**具体实现**：开关 `自动选曲`，默认关闭。开启时设置 `RhythmSelectSong` 的 `auto_select` 为 `true`；关闭时提供 `演奏目标歌曲选择` 子选项。

#### 目标歌曲

手动选择要演奏的歌曲。仅在关闭"自动选曲"时可用。

**具体实现**：下拉选择框 `演奏目标歌曲选择`，支持以下歌曲：

- `Heroic_Appearance`
- `Destiny`
- `Everlasting_Dazing_Summer(Short_Ver.)`
- `Everlasting_Dazing_Summer`
- `迷星叫`

覆写 `RhythmSelectSong` 的 `custom_action_param.song_name` 参数。

### 自动连打

"自动连打"功能开关。启用后连续演奏，关闭则仅演奏一次。

**具体实现**：开关 `自动连打`，默认关闭。开启时提供 `连打模式` 子选项；关闭时设置 `RhythmRepeatCheck` 的 `auto_repeat_count` 为 `0`、`auto_repeat_max` 为 `false`。

#### 连打模式

- **Max（连打直到活力耗尽）**：自动连打直到活力耗尽
- **固定次数**：使用固定次数，需额外设置连打次数

**具体实现**：开关 `连打模式`，默认关闭。Max 模式设置 `RhythmRepeatCheck` 的 `auto_repeat_max` 为 `true`；固定次数模式提供 `连打次数` 子选项。

##### 连打次数

连续演奏的次数。

**具体实现**：`int` 类型输入框 `连打次数`，默认 `5`。覆写 `RhythmRepeatCheck` 的 `custom_action_param` 中的 `auto_repeat_count` 和 `auto_repeat_max`。

### 目标帧率

演奏循环的每秒帧率，建议设置在 30~120 之间。

**具体实现**：`int` 类型输入框 `目标帧率`，默认 `60`。覆写 `RhythmPlaying` 的 `custom_action_param.target_fps` 参数。
