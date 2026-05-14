<!-- markdownlint-disable MD028 MD029 MD033 MD041 -->
<div align="center">
  <img alt="LOGO" src="../assets/logo.png" width="256" height="256" />

# MaaNTE

  <p align="center">
    MAA Assistant for Neverness to Everness
    <br/>
    <a href="../README.md">中文</a> | <a href="README_zh-tw.md">繁體中文</a> | <a href="https://github.com/1bananachicken/MaaNTE/issues">Report Bug</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Platform-Windows-0078D7?style=flat-square&logo=Windows" alt="Platform" />
    <img src="https://img.shields.io/badge/Language-Python%20%2F%20PipeLine-blue?style=flat-square&logo=Python" alt="Language" />
    <img alt="license" src="https://img.shields.io/github/license/1bananachicken/MaaNTE?style=flat-square">
    <br>
    <a href="https://maafw.com/" target="_blank"><img alt="website" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/maafw.svg"></a>
    <a href="https://mirrorchyan.com/zh/projects?rid=MaaNTE" target="_blank"><img alt="mirrorc" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/mirrorc-en.svg"></a>
    <a href="https://space.bilibili.com/3546893080594665" target="_blank"><img alt="Bilibili" src="https://img.shields.io/badge/Bilibili-MaaNTE--Official-00A1D6?logo=bilibili"></a>
    <a href="https://discord.gg/EjwrrGzy" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-MaaNTE--Official-5865F2?logo=discord"></a>
    <br/>
    <img alt="commit" src="https://img.shields.io/github/commit-activity/m/1bananachicken/MaaNTE?&style=flat-square&logo=github&color=darkgreen">
    <img src="https://img.shields.io/github/stars/1bananachicken/MaaNTE?style=flat-square&logo=github&color=darkgreen" alt="Stars" />
  </p>

Powered by [MaaFramework](https://github.com/MaaXYZ/MaaFramework) !

</div>

> [!Warning]
>
> This document was translated using AI and may contain inaccuracies. Corrections are welcome!

> [!Caution]
>
> Recently, a large number of accounts have been redistributing this software, even impersonating official accounts. Software obtained through unofficial channels **may contain viruses** and is generally not the latest version. Please verify carefully.
>
> We have recently discovered that some individuals are using this project to spread viruses. Please make sure to download only from the official channels.

> [!Tip]
>
> This project is still in early development. PRs and Issues are welcome.

> [!Tip]
>
> Official Bilibili account: [MaaNTE-Official](https://space.bilibili.com/3546893080594665)
>
> QQ Group 1: 1103323319 (Full)
>
> QQ Group 2: 1101147419 (Full)
>
> QQ Group 3: 1075143235 (Full)
>
> QQ Group 4: 713114598

# Disclaimer and Risk Notice

> [!Note]
>
> This software is a third-party tool designed to simplify gameplay in *Neverness to Everness* by simulating regular interaction actions. This project complies with relevant laws and regulations. It aims to reduce repetitive operations for users, does not disrupt game balance or provide unfair advantages, and will never modify any game files or data.
>
> This project is open-source, free, and intended for learning and exchange purposes only. Commercial or profit-driven use is prohibited. The development team reserves the right of final interpretation of this project. Any issues arising from the use of this software are unrelated to this project and its developers.

> [!Caution]
>
> According to the [*Neverness to Everness* Fair Play Declaration](https://yh.wanmei.com/news/gamebroad/20260202/260701.html):
>
> The use of any third-party tools to undermine game fairness is strictly prohibited. We will crack down on the use of cheats, speed hacks, exploit tools, macro scripts, and other unauthorized tools. These behaviors include, but are not limited to, auto-afk, skill acceleration, invincibility mode, teleportation, and modification of game data. Once verified, we will take measures including but not limited to deducting illicit gains, freezing game accounts, and permanently banning game accounts, depending on the severity and frequency of violations.
>
> **You should fully understand and voluntarily bear all risks that may arise from using this tool.**

## 📄Open Source License

This software is open-sourced under [GNU Affero General Public License v3.0 only](https://spdx.org/licenses/AGPL-3.0-only.html).

## ✨Features

- 🎣 Auto Fishing
  - 🐟 Auto Sell Fish
  - 🪝 Auto Buy Bait
- 🥤 Auto Make Coffee
  - 🔨 Chase away all customers
- 💰 Auto Collect One Coffee House Revenue
  - 📦 Auto Restock
- 💎 Auto Claim Rewards
  - 🎁 Activity Points
  - 📅 Cycle Bounty
- 🕛 Real-time Assistance
  - 🗼 Auto Teleport
  - ⏩ Auto Skip Story
- ⚔️ Auto Dodge
  - 🛡️ Audio-based auto dodge/counter
- 🎵 Auto Rhythm Game

## ❓FAQ

### 📄Troubleshooting

Please refer to the [Trouble shooting Guide](zh_cn/trouble_shooting.md) (English).

### 🤔Can't find how to start?

- Regular users please download the release version. Only clone the repository if you plan to develop.

## 💡Notes

The game must run in windowed mode at 1280×720 resolution. The new fishing algorithm supports 120 FPS.

## 💻Development Guide

<details><summary>Click to expand</summary>

MaaNTE Developer QQ Group: 1092630280

### Quick Start

0. Set up environment

- Install Python >= 3.11
- It is recommended to use VS Code as your IDE and install the [vscode extension "maa-support"](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) for debugging.

1. Fork the project

- Click `Fork`, then click `Create Fork`.

2. Clone your forked repository and pull submodules

```bash
git clone --recursive https://github.com/<your-username>/MaaNTE.git
```

3. Download the [release package of MaaFramework](https://github.com/MaaXYZ/MaaFramework/releases) and extract it into the `deps` folder.

4. Submit a PR

- New feature development should be submitted to the `dev` branch.

For more development documentation, refer to the [M9A documentation site](https://1999.fan/zh_cn/develop/development.html).

</details>

## ☕Acknowledgements

### Open Source Projects

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  An automation black-box testing framework based on image recognition.
- [MXU](https://github.com/MistEO/MXU)
  MaaFramework Next UI
- [MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)
  Assistant for Arknights – automate your daily grind with one click!
- [MaaEnd](https://github.com/MaaEnd/MaaEnd)
  Vision AI based automation tool for Arknights: Endfield.
- [M9A](https://github.com/MAA1999/M9A)
  Assistant for Reverse: 1999.
- ~~[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  A general-purpose GUI solution for MaaFramework built with Avalonia UI.~~

### Contributors

Thanks to all the developers who participated in testing and development (´▽`ʃ♡ƪ)

[![Contributors](https://contributors-img.web.app/image?repo=1bananachicken/MaaNTE&max=1000)](https://github.com/1bananachicken/MaaNTE/graphs/contributors)

## ☕Buy Us a Coffee

If MaaNTE has saved you a lot of time, how about buying the developers a coffee?

Your support is our biggest motivation to keep updating! 🥰

[<img width="200" alt="Sponsor Us" src="https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png">](https://afdian.com/a/MaaNTE)

## ⭐Star History

If you find this software helpful, please give us a Star! (the little star at the top right of the page) – that's the greatest support for us!

<a href="https://www.star-history.com/?repos=1bananachicken%2FMaaNTE&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
 </picture>
</a>
