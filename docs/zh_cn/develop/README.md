# MaaNTE 开发者文档

本目录包含 MaaNTE 项目的全部开发者文档。

## 阅读路线

建议按以下顺序阅读：

1. 搭环境、跑起来、完成第一次改动和 PR → `getting-started.md`
2. 提交 PR 前检查格式、描述与验证记录 → `pull-request-guidelines.md`
3. 了解 Pipeline 编写规范 → `pipeline-guide.md`
4. 查阅编码规范 → `coding-standards.md`
5. 需要编写 Python 自定义逻辑 → `custom-action.md`
6. 理解场景跳转机制 → `scene-manager.md`
7. 需要调试单节点 → `node-testing.md`

## 文档索引

### Tier 1 — 快速上手

| 文档 | 说明 |
|------|------|
| [快速开始](./getting-started.md) | 以真实案例（#223 → #231）走一遍完整开发流程 |
| [PR 规范](./pull-request-guidelines.md) | PR 标题、描述、验证记录、评审与提交前检查清单 |

### Tier 2 — 参考手册

| 文档 | 说明 |
|------|------|
| [自定义动作开发](./custom-action.md) | Python CustomAction 编写、Controller API、Pipeline 集成 |
| [本地路线寻路接口](./local-route-navigation.md) | local_route_navigation Pipeline 入口、LocalRouteNavigation 类接口、路线 JSON 格式 |
| [节点测试](./node-testing.md) | 如何编写和运行节点测试，验证识别是否稳定命中 |
| [DMCA / Abuse 提报模板](./dmca-abuse-template.md) | 仿冒/搬运/带毒仓库的一键复用提报文案（AGPL-3.0） |
| [DeepWiki — MaaNTE](https://deepwiki.com/1bananachicken/MaaNTE) | 带 AI 的在线项目文档速览 |
| [Pipeline 协议](https://maafw.com/docs/3.1-PipelineProtocol/) | MaaFramework 官方 Pipeline 协议全文 |

### Tier 3 — 规范与约束

| 文档 | 说明 |
|------|------|
| [编码规范](./coding-standards.md) | Pipeline / Python 编码规则、提交前检查、常见坑 |
| [PR 规范](./pull-request-guidelines.md) | 提交前检查、变更要求、评审协作约定 |

## Pipeline 基础组件

日常开发最常用的可复用节点，建议所有 Pipeline 开发者查询以便复用。

| 文档 | 说明 | 路径 |
|------|------|------|
| [场景管理器](./scene-manager.md) | 从任意界面自动导航到目标场景 | `Interface/Scene/` |
| [InScene 场景识别](./in-scene.md) | 判断当前画面所在场景 | `Interface/Scene/Status.json` |
| [通用按钮](./common-buttons.md) | 各场景入口按钮 | `Common/Button/` |
| [Custom 动作与识别](./custom-action.md) | 通用 Python 工具：alt_click等 | `agent/custom/action/Common/` |
| [本地路线寻路](./local-route-navigation.md) | 按路线 JSON 执行地图寻路 | `LocalRouteNavigation.json` |

## 高级组件参考

按需查阅。仅在使用对应组件时需要阅读。

| 文档 | 说明 |
|------|------|
| 自动战斗 | ⚠ 开发中 |
| 自动导航 | ⚠ 开发中 |

## 任务维护文档

仅在维护对应任务时需要阅读。

| 文档 | 说明 |
|------|------|
| 待补充 | 待补充 |

## 快速跳转

| 我想做什么 | 该看哪里 |
|-----------|---------|
| 第一次参与，从零开始 | [getting-started.md](./getting-started.md) |
| 准备提交 PR | [pull-request-guidelines.md](./pull-request-guidelines.md) |
| 改 Pipeline 节点 | [pipeline-guide.md](./pipeline-guide.md) |
| 写 Python 自定义逻辑 | [custom-action.md](./custom-action.md) |
| 调用本地路线寻路 | [local-route-navigation.md](./local-route-navigation.md) |
| 场景跳转/界面导航 | [scene-manager.md](./scene-manager.md) |
| 调试单个节点 | [node-testing.md](./node-testing.md) |
| 查阅编码规范 | [coding-standards.md](./coding-standards.md) |

## 参考

- [MaaFramework 文档](https://maafw.com/docs/1.1-QuickStarted)

## 交流

开发QQ群：1092630280
