# 音频闪避与反击

## 简介

基于**音频识别**的自动闪避与反击功能。通过识别游戏音效实时触发闪避和反击操作。

需要使用[color:red]**桌面端-前台**[/color]控制器。

## 功能

### 闪避

实时监听游戏音频，识别到敌方攻击音效时自动触发闪避操作。

### 反击

在闪避成功后识别反击音效，自动触发反击操作。

## 配置详解

### 启用音频闪避

"音频闪避"功能的总开关。关闭时会同时关闭闪避和反击。

**具体实现**：开关 `SoundDodgeEnable`，默认开启。关闭时将 `SoundDodgeEnableConfig.attach.enable_sound_trigger` 覆写为 `false`。

### 仅闪避模式

开启后只执行闪避，不执行反击。

**具体实现**：开关 `SoundDodgeAllAttacks`，默认仅闪避。开启时将 `SoundDodgeModeConfig.attach.dodge_all_attacks` 设置为 `true`，两种音效都触发闪避；关闭时设置为 `false`，反击音效触发反击。

### 闪避阈值

闪避音效识别阈值，范围 0.0~1.0，**越低越敏感**。如果漏闪避，请调低数值；如果误闪避，请调高数值。

**具体实现**：`string` 类型输入框 `SoundDodgeThreshold`，默认 `0.13`。通过 `^0\.\d+$|^1\.0*$` 校验。覆写 `SoundDodgeThresholdConfig.attach.threshold`。

### 反击阈值

反击音效识别阈值，范围 0.0~1.0，**越低越敏感**。如果漏反击，请调低数值；如果误反击，请调高数值。

**具体实现**：`string` 类型输入框 `SoundCounterThreshold`，默认 `0.12`。通过 `^0\.\d+$|^1\.0*$` 校验。覆写 `SoundCounterThresholdConfig.attach.counter_attack_threshold`；仅闪避模式下命中该音效时仍执行闪避。
