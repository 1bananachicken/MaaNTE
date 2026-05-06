<!-- markdownlint-disable MD028 MD029 MD033 MD041 -->
<div align="center">
  <img alt="LOGO" src="./assets/logo.png" width="256" height="256" />

# MaaNTE

  <p align="center">
    MAA异环小助手
    <br/>
    <a href="docs/README_en.md">English</a> | <a href="https://github.com/1bananachicken/MaaNTE/issues">报告 Bug</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Platform-Windows-0078D7?style=flat-square&logo=Windows" alt="Platform" />
    <img src="https://img.shields.io/badge/Language-Python%20%2F%20PipeLine-blue?style=flat-square&logo=Python" alt="Language" />
    <img alt="license" src="https://img.shields.io/github/license/1bananachicken/MaaNTE?style=flat-square">
    <br>
    <a href="https://maafw.com/" target="_blank"><img alt="website" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/maafw.svg"></a>
    <a href="https://mirrorchyan.com/zh/projects?rid=MaaNTE" target="_blank"><img alt="mirrorc" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/mirrorc-zh.svg"></a>
    <br/>
    <img alt="commit" src="https://img.shields.io/github/commit-activity/m/1bananachicken/MaaNTE?color=%23ff69b4">
    <img src="https://img.shields.io/github/stars/1bananachicken/MaaNTE?style=flat-square&logo=github&color=darkgreen" alt="Stars" />
  </p>

由 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 强力驱动！

</div>

> [!Tip]
>
> 本项目还处于早期开发阶段，欢迎提交PR和Issue

> [!Tip]
>
> 官方 B 站账号: [MaaNTE-Official](https://space.bilibili.com/3546893080594665)
>
> QQ交流1群: 1103323319 (已满)
>
> QQ交流2群: 1101147419 (已满)
>
> QQ交流3群: 1075143235

> [!Caution]
>
> 近期出现大量账号二次分发本软件，甚至冒充官方账号。通过非官方途径获取的软件可能**含有病毒**，并且一般不是最新的版本，请注意甄别。

# ⚠️ 免责声明与风险提示

> [!Note]
>
> 本软件为第三方工具，通过识别游戏画面模拟常规交互动作，简化《异环》的游戏玩法。本项目遵循相关法律法规。本项目旨在简化用户的重复性操作，不会破坏游戏平衡或提供不公平优势，也绝不会修改任何游戏文件或数据。
>
> 本项目开源、免费，且仅供学习和交流使用，请勿用于任何商业或营利性目的。开发者团队拥有本项目的最终解释权。因使用本软件而产生的任何问题，均与本项目及开发者无关。

> [!Caution]
>
> 根据[《异环》公平游戏宣言](https://yh.wanmei.com/news/gamebroad/20260202/260701.html)：
>
> 严禁使用任何第三方工具破坏游戏公平性。我们将严厉打击使用外挂、加速器、作弊软件、宏脚本等非法工具的行为，这些行为包括但不限于自动挂机、技能加速、无敌模式、瞬移、修改游戏数据等操作。一经查实，我们将视违规严重程度及违规次数，采取包括但不限于扣除违规收益、冻结游戏账号、永久封禁游戏账号等措施。
>
> **您应充分了解并自愿承担使用本工具可能带来的所有风险。**

## 🧾 开源许可

本软件使用 [GNU Affero General Public License v3.0 only](https://spdx.org/licenses/AGPL-3.0-only.html) 开源。

## ✨ 功能一览

- 🎣 自动钓鱼
  - 🐟 自动卖鱼
  - 🪝 自动买鱼饵
- 🥤 自动做咖啡
  - 🔨 驱赶所有顾客
- 💰 自动提取一咖舍收益
  - 📦 自动补货
- 💎 自动领取奖励
  - 🎁 活跃度
  - 📅 环期赏令
- 🕛 实时辅助
  - 🗼 自动传送
  - ⏩ 自动跳剧情
- ⚔️ 自动闪避
  - 🛡️ 基于音频识别的自动闪避/反击
- 🎵 自动超强音
  - 🥁 循环演奏迷星叫

## ❓常见问题

### 📄问题排查手册

先前往[问题排查手册](docs/zh_cn/问题排查.md)寻找解决方案。

### 🤔找不到怎么启动？

- 普通用户请下载release版本，有开发需求再clone仓库

## 💡注意事项

游戏需要运行在1280x720分辨率，窗口化，钓鱼新算法支持120帧运行

## 💻开发指南
MaaNTE开发者群: 1092630280

<details><summary>点击展开</summary>

### 快速开始

0. 配置环境

- 安装 python>=3.11
- 建议使用vscode作为IDE进行开发，并安装 [vscode插件“maa-support”](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) 进行调试

1. Fork项目

- 点击 `Fork`，继续点击 `Create Fork`

2. 克隆自己 fork 的项目到本地，并拉取子模块

```bash
git clone --recursive https://github.com/<你的用户名>/MaaNTE.git
```

3. 下载MaaFramework的 [release包](https://github.com/MaaXYZ/MaaFramework/releases)，解压到 `deps` 文件夹中

4. 提交PR

- 新功能开发请提交到dev分支

更多开发文档可以参考 [M9A文档站](https://1999.fan/zh_cn/develop/development.html)

</details>

## ☕鸣谢

### 开源项目

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  基于图像识别的自动化黑盒测试框架
- [MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)
  《明日方舟》小助手，全日常一键长草！
- [MaaEnd](https://github.com/MaaEnd/MaaEnd)
  基于视觉 AI 的「明日方舟：终末地」自动化工具
- [M9A](https://github.com/MAA1999/M9A)
  重返未来：1999 小助手
- [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  基于 Avalonia UI 构建的 MaaFramework 通用 GUI 解决方案
### ❤️贡献/参与者

感谢全部参与到测试与开发中的开发者(´▽`ʃ♡ƪ)

[![Contributors](https://contributors-img.web.app/image?repo=1bananachicken/MaaNTE&max=200&columns=15)](https://github.com/1bananachicken/MaaNTE/graphs/contributors)

## 广告

如果觉得软件对你有帮助，帮忙点个 Star 吧！~（网页最上方右上角的小星星），这就是对我们最大的支持了！
