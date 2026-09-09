# 钓鱼任务

## 简介

自动执行钓鱼任务的挂机功能。包含旧版和新版两套任务入口。

## 功能

### 钓鱼任务

自动循环执行钓鱼：抛竿、等待上钩、收杆，并支持自动卖鱼和自动买鱼饵。

### 钓鱼任务（新）

重构的钓鱼功能，理论上能无限钓鱼。

> [!WARNING]
> 仍无法直接处理被月卡打断的跨夜钓鱼，可以尝试设置定时任务来继续钓鱼，实际效果无法保证。
> 钓鱼功能仅自动买鱼饵和自动卖鱼会抢占鼠标。

## 配置详解

### 循环次数

设置钓鱼任务的循环次数。仅在不启用"无限循环"时生效。

**具体实现**：`int` 类型输入框 `FishLoopTime`，通过 `^\d+$` 校验数据。覆写 `FishStart` 的 `max_hit` 参数。

### 每次钓鱼数量

设置每次循环自动钓鱼的次数，建议不超过 99。

**具体实现**：`int` 类型输入框 `FishNumber`，默认 `99`。覆写 `FishGameStart` 的 `custom_action_param.count` 和 `FishNewCast` 的 `max_hit` 参数。

### 无限循环

启用后钓鱼任务将无限循环，直到手动停止。启用时会忽略"循环次数"设置。

**具体实现**：开关 `FishLoopInfinite`，默认禁用。启用时修改 `FishEntrance`、`FishGameStart`、`AutoSellFish`、`AutoBuyFishBait`、`FishBaitHandled` 的 `next` 指向 `FishLoopStart` 形成循环。

### 自动卖鱼

是否启用自动卖鱼功能。在鱼鳞币不足时自动出售背包中的鱼。

**具体实现**：开关 `FishSellAuto`。覆写 `AutoSellFish` 和 `FishNewOpenFishMaster` 的 `enabled` 状态。

### 自动买鱼饵

每次自动购买 99 个鱼饵（上限）。识别到鱼饵不足时购买，若当前鱼饵用完则切换为万能鱼饵。

**具体实现**：开关 `FishBuyBaitAuto`。覆写 `AutoBuyFishBait`、`FishGotoBuyBait`、`FishNewGotoBuyBait` 的 `enabled` 状态。启用时提供 `FishBaitThreshold` 子选项。

### 鱼饵识别阈值

如果无法识别点击鱼饵位置，请调低该数值。

**具体实现**：下拉选择框 `FishBaitThreshold`，可选值 `0.8`、`0.7`、`0.6`。覆写 `AutoBuyFishBait` 的 `custom_action_param.found_bait_threshold` 参数。

新版 `FishNew` 的控条使用实时闭环：每帧在同一个小型 HSV 区域中同时识别绿条和光标，分别跟踪两者的位置与速度。控制器将绿条中央约 40% 作为跟随走廊，在走廊内不追逐单一中心点，预测将离开走廊时才输出有时长上限的 A/D 短脉冲。绿条中心可以连续移动，宽度变化作为独立事件连续确认后更新。控条结束后会正常处理钓获、鱼逃走或已回到钓鱼界面的状态，不再每条鱼都执行错误重进流程。

新版商店不会直接购买最高分模板候选。程序会收集多个候选，逐个点击并 OCR 右侧详情中的商品名称；只有确认是“万能鱼饵”后，才继续点击“最多”按钮并购买。
