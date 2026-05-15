<!-- markdownlint-disable MD028 MD029 MD033 MD041 -->
<div align="center">
  <img alt="LOGO" src="../assets/logo.png" width="256" height="256" />

# MaaNTE

  <p align="center">
    MAA異環小助手
    <br/>
    <a href="docs/eng/README_en.md">English</a> | <a href="../README.md">简体中文</a> | <a href="docs/README_zh-tw.md">繁體中文</a> | <a href="docs/jp/README_jp.md">日本語</a> | <a href="https://github.com/1bananachicken/MaaNTE/issues">报告 Bug</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Platform-Windows-0078D7?style=flat-square&logo=Windows" alt="Platform" />
    <img src="https://img.shields.io/badge/Language-Python%20%2F%20PipeLine-blue?style=flat-square&logo=Python" alt="Language" />
    <img alt="license" src="https://img.shields.io/github/license/1bananachicken/MaaNTE?style=flat-square">
    <br>
    <a href="https://maafw.com/" target="_blank"><img alt="website" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/maafw.svg"></a>
    <a href="https://mirrorchyan.com/zh/projects?rid=MaaNTE" target="_blank"><img alt="mirrorc" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/mirrorc-zh.svg"></a>
    <a href="https://space.bilibili.com/3546893080594665" target="_blank"><img alt="Bilibili" src="https://img.shields.io/badge/Bilibili-MaaNTE--Official-00A1D6?logo=bilibili"></a>
    <a href="https://discord.gg/EjwrrGzy" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-MaaNTE--Official-5865F2?logo=discord"></a>
    <br/>
    <img alt="commit" src="https://img.shields.io/github/commit-activity/m/1bananachicken/MaaNTE?&style=flat-square&logo=github&color=darkgreen">
    <img src="https://img.shields.io/github/stars/1bananachicken/MaaNTE?style=flat-square&logo=github&color=darkgreen" alt="Stars" />
  </p>

由 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 強力驅動！

</div>

> [!Warning]
>
> 本文件由 AI 翻譯，可能存在不準確之處。歡迎修正！

> [!Caution]
>
> 近期出現大量帳號二次分發本軟體，甚至冒充官方帳號。透過非官方途徑取得的軟體可能**含有病毒**，且通常不是最新版本，請注意辨別。
>
> 近日，我們發現有人借本專案之名傳播病毒，請務必從官方管道下載。

> [!Tip]
>
> 本專案仍處於早期開發階段，歡迎提交 PR 和 Issue

> [!Tip]
>
> 官方 B 站帳號：[MaaNTE-Official](https://space.bilibili.com/3546893080594665)
>
> QQ交流1群：1103323319（已滿）
>
> QQ交流2群：1101147419（已滿）
>
> QQ交流3群：1075143235（已滿）
>
> QQ交流4群：713114598

# ⚠️免責聲明與風險提示

> [!Note]
>
> 本軟體為第三方工具，透過辨識遊戲畫面模擬常規互動動作，簡化《異環》的遊戲玩法。本專案遵循相關法律法規。本專案旨在簡化使用者的重複性操作，不會破壞遊戲平衡或提供不公平優勢，也絕不會修改任何遊戲檔案或資料。
>
> 本專案開源、免費，僅供學習和交流使用，請勿用於任何商業或營利目的。開發者團隊擁有本專案的最終解釋權。因使用本軟體而產生的任何問題，均與本專案及開發者無關。

> [!Caution]
>
> 根據[《異環》公平遊戲宣言](https://yh.wanmei.com/news/gamebroad/20260202/260701.html)：
>
> 嚴禁使用任何第三方工具破壞遊戲公平性。我們將嚴厲打擊使用外掛、加速器、作弊軟體、巨集指令碼等非法工具的行為，這些行為包括但不限於自動掛機、技能加速、無敵模式、瞬移、修改遊戲資料等操作。一經查實，我們將視違規嚴重程度及違規次數，採取包括但不限於扣除違規收益、凍結遊戲帳號、永久封禁遊戲帳號等措施。
>
> **您應充分瞭解並自願承擔使用本工具可能帶來的所有風險。**

## 🧾開源許可

本軟體使用 [GNU Affero General Public License v3.0 only](https://spdx.org/licenses/AGPL-3.0-only.html) 開源。

## ✨功能一覽

- 🎣 自動釣魚
  - 🐟 自動賣魚
  - 🪝 自動買魚餌
- 🥤 自動做咖啡
  - 🔨 驅趕所有顧客
- 💰 自動提取一咖舍收益
  - 📦 自動補貨
- 💎 自動領取獎勵
  - 🎁 活躍度
  - 📅 環期賞令
- 🕛 即時輔助
  - 🗼 自動傳送
  - ⏩ 自動跳劇情
- ⚔️ 自動閃避
  - 🛡️ 基於音訊辨識的自動閃避／反擊
- 🎵 自動超強音
  - 🥁 循環演奏迷星叫

## ❓常見問題

### 📄問題排查手冊

請先前往[問題排查手冊](zh_cn/問題排查.md)尋找解決方案。

### 🤔找不到怎麼啟動？

- 一般使用者請下載 release 版本，有開發需求再 clone 倉庫

## 💡注意事項

遊戲需要在 1280×720 解析度下以視窗模式執行，釣魚新演算法支援 120 幀執行。

## 💻開發指南

<details><summary>點擊展開</summary>

MaaNTE 開發者群：1092630280

### 快速開始

0. 設定環境

- 安裝 Python >= 3.11
- 建議使用 VS Code 作為 IDE 進行開發，並安裝 [vscode 外掛「maa-support」](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) 進行除錯

1. Fork 專案

- 點選 `Fork`，繼續點選 `Create Fork`

2. 複製自己 fork 的專案到本機，並拉取子模組

```bash
git clone --recursive https://github.com/<你的使用者名稱>/MaaNTE.git
```

3. 下載 MaaFramework 的 [release 包](https://github.com/MaaXYZ/MaaFramework/releases)，解壓縮到 `deps` 資料夾中

4. 提交 PR

- 新功能開發請提交到 dev 分支

更多開發文件可以參考 [M9A 文件站](https://1999.fan/zh_cn/develop/development.html)

</details>

## ❤️鳴謝

### 開源專案

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  基於圖像辨識的自動化黑箱測試框架
- [MXU](https://github.com/MistEO/MXU)
  MaaFramework Next UI
- [MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)
  《明日方舟》小助手，全日常一鍵長草！
- [MaaEnd](https://github.com/MaaEnd/MaaEnd)
  基於視覺 AI 的「明日方舟：終末地」自動化工具
- [M9A](https://github.com/MAA1999/M9A)
  重返未來：1999 小助手
- ~~[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  基於 Avalonia UI 構建的 MaaFramework 通用 GUI 解決方案~~

### 貢獻／參與者

感謝全部參與到測試與開發中的開發者 (´▽`ʃ♡ƪ)

[![Contributors](https://contributors-img.web.app/image?repo=1bananachicken/MaaNTE&max=1000)](https://github.com/1bananachicken/MaaNTE/graphs/contributors)

## ☕請我們喝咖啡

如果 MaaNTE 幫你省下了不少時間，不妨請開發者喝杯咖啡？

你的支持就是我們持續更新的最大動力 🥰

[<img width="200" alt="贊助我們" src="https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png">](https://afdian.com/a/MaaNTE)

## ⭐Star 記錄

如果覺得軟體對你有幫助，幫忙點個 Star 吧！（網頁最上方右上角的小星星），這就是對我們最大的支持了！

<a href="https://www.star-history.com/?repos=1bananachicken%2FMaaNTE&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
 </picture>
</a>
