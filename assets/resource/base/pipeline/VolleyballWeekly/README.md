# 排球模块开发规范

## 功能模块一律用 Pipeline JSON

**新功能节点优先用 pipeline JSON 实现，不要写 Python CustomAction。**

- 简单时序：`ClickKey` + `pre_delay`/`post_delay` 即可
- 条件分支：用 `ColorMatch` / `TemplateMatch` / `OCR` 做 recognition，命中才执行动作
- 循环：利用 `next` 列表回指自身
- 只有 pipeline JSON 确实表达不了的逻辑（复杂计算、多步状态机、外部接口）才写 Python

## 功能单元开关

每个独立功能单元在 task JSON 里配一个 `switch` 选项，默认开启，通过 `pipeline_override` 设置节点 `enabled` 控制。用户在 GUI 勾选即可开关，无需改代码。
