<!-- markdownlint-disable MD028 MD029 MD033 MD041 -->
<div align="center">
  <img alt="LOGO" src="/assets/logo.png" width="256" height="256" />

# MaaNTE

  <p align="center">
    MAA異環小助手
    <br/>
    <a href="https://docs.maante.org/"><strong>公式サイト</strong></a>
    ·
    <a href="../eng/README_en.md">English</a> | <a href="../README.md">简体中文</a> | <a href="../README_zh-tw.md">繁體中文</a> | <a href="https://github.com/1bananachicken/MaaNTE/issues">バグ報告</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Platform-Windows-0078D7?style=flat-square&logo=Windows" alt="Platform" />
    <img src="https://img.shields.io/badge/Language-Python%20%2F%20PipeLine-blue?style=flat-square&logo=Python" alt="Language" />
    <img alt="license" src="https://img.shields.io/github/license/1bananachicken/MaaNTE?style=flat-square">
    <br>
    <a href="https://docs.maante.org/ja_jp/" target="_blank"><img alt="website" src="https://img.shields.io/badge/Website-docs.maante.org-00A98F?style=flat-square"></a>
    <a href="https://maafw.com/" target="_blank"><img alt="website" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/maafw.svg"></a>
    <a href="https://mirrorchyan.com/zh/projects?rid=MaaNTE" target="_blank"><img alt="mirrorc" src="https://raw.githubusercontent.com/MaaXYZ/MaaFramework/refs/heads/main/docs/static/mirrorc-zh.svg"></a>
    <a href="https://space.bilibili.com/3546893080594665" target="_blank"><img alt="Bilibili" src="https://img.shields.io/badge/Bilibili-MaaNTE--Official-00A1D6?logo=bilibili"></a>
    <a href="https://discord.gg/e6mPMRYQpR" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-MaaNTE--Official-5865F2?logo=discord"></a>
    <br/>
    <img alt="commit" src="https://img.shields.io/github/commit-activity/m/1bananachicken/MaaNTE?&style=flat-square&logo=github&color=darkgreen">
    <img src="https://img.shields.io/github/stars/1bananachicken/MaaNTE?style=flat-square&logo=github&color=darkgreen" alt="Stars" />
  </p>

[MaaFramework](https://github.com/MaaXYZ/MaaFramework) によって強力に駆動されています！

</div>

> [!Warning]
>
> このドキュメントは AI 翻訳を使用しており、不正確な箇所がある可能性があります。修正の提案を歓迎します。

> [!Caution]
>
> 最近、多くのアカウントがこのソフトウェアを二次配布したり、公式アカウントを装ったりする事例が発生しています。非公式な手段で入手したソフトウェアには**ウイルスが含まれている可能性**があり、通常は最新版ではありません。[GitHub Releases](https://github.com/1bananachicken/MaaNTE/releases) などの公式チャネルからダウンロードしてください。

> [!Tip]
>
> 本プロジェクトはまだ初期開発段階にあり、PR や Issue の提出を歓迎します。
>
> ダウンロード、ドキュメント、トラブルシューティングは **[公式サイト docs.maante.org](https://docs.maante.org/)** をご利用ください。
> QQ グループへの参加は [公式 QQ グループページ](https://docs.maante.org/zh_cn/qq-group/) から — 空きがあるグループが自動的に割り当てられます。
>
> その他のチャンネル：
> 公式 B 站アカウント: [MaaNTE-Official](https://space.bilibili.com/3546893080594665)
>
> 公式Discord: [Discord](https://discord.gg/e6mPMRYQpR)

### ⚠️免責事項とリスク警告

> [!Note]
>
> 本ソフトウェアはサードパーティ製ツールであり、ゲーム画面を認識し通常の操作をシミュレートすることで、『異環』のゲームプレイを簡素化するものです。本プロジェクトは関連する法律および規制を遵守しています。ユーザーの反復作業を簡素化することを目的としており、ゲームのバランスを損なったり不公平な優位性を提供したりすることはありません。また、ゲームファイルやデータを改変することもありません。
>
> 本プロジェクトはオープンソースかつ無料であり、学習および交流の目的でのみご利用ください。商業目的や営利目的での使用はご遠慮ください。開発チームは本プロジェクトに関する最終な解釈権を有します。本ソフトウェアの使用に起因するいかなる問題についても、本プロジェクトおよび開発者は一切の責任を負いません。

> [!Caution]
>
> [『異環』公平ゲーム宣言](https://nte.perfectworld.com/jp/article/news/gamebroad/20260206/260831.html) によると：
>
> ゲームの公平性を損なう非公式ツールの使用は固く禁止されています。チート、スピードハック、マクロスクリプトなどの外部ツールの利用は厳しく取り締まり、違反が確認された場合はアカウント停止等の措置を取ります。違反行為には、オートプレイ、スキルのスピードアップ、無敵化、瞬間移動、ゲームデータの改ざんなどが含まれますが、これらに限られません。
>
> **本ツールの使用によって生じる可能性のあるすべてのリスクを十分に理解し、自発的に引き受けてください。**

### 🧾オープンソースライセンス

本ソフトウェアは [GNU Affero General Public License v3.0 only](https://spdx.org/licenses/AGPL-3.0-only.html) の下でオープンソース化されています。

## ✨機能一覧

- 🎣 自動釣り
  - 🐟 自動で魚を売る
  - 🪝 自動で餌を買う
- 🥤 自動でコーヒーを作る
  - 🔨 すべての顧客を追い出す
- 💰 一咖舎の収益を自動受取
  - 📦 自動補充
  - 🏷️ 風向標に基づく商品の自動入れ替え
- 🪑 家具の自動受取
  - 🏠 ウィーナー/エデン/スカイビュー/金都/峰林の 5 つのアパートに対応
- 💎 報酬の自動受取
  - 🎁 アクティビティポイント
  - 📅 環期賞令
- 🐾 ピンクパウ強盗
  - 🔄 無限ループ放置対応
  - 📊 収益統計とログ記録
- 🧩 自動テトリス
  - 🤖 内蔵 AI による自動判断
  - 🔁 スタミナが尽きるまで自動連打
- 🕛 リアルタイム支援
  - 🗼 自動テレポート
  - ⏩ ストーリーの自動スキップ
  - 🎁 自動拾得
- ⚔️ 自動回避
  - 🛡️ 音声認識に基づく自動回避/反撃
- 🎵 自動リズムゲーム
  - 🥁 迷星叫のループ演奏
- 🎹 自動ピアノ演奏
  - 📂 任意の MIDI インポートに対応
- 📋 プリセットタスク
  - 💤 放置タスク / 🗺️ リアルタイム支援

## ❓よくある質問

### 📄トラブルシューティング

まずは[トラブルシューティングマニュアル](../jp/トラブルシューティングマニュアル.md)をご確認ください。

### 🤔起動方法がわからない？

- 一般ユーザーは [Release 版](https://github.com/1bananachicken/MaaNTE/releases) をダウンロードしてください。開発する場合のみリポジトリをクローンしてください。

### 💡注意事項

ゲームは 1280×720 解像度のウィンドウモードで実行する必要があります。釣りの新アルゴリズムは 120 FPS に対応しています。

### 💻開発ガイド

開発に参加したい、またはプロジェクトを詳しく知りたい方はこちら 👉 [開発者ドキュメント](../zh_cn/develop/README.md)

コントリビューションをお待ちしています！一緒に MaaNTE をもっと強力にしましょう！💪

## ❤️謝辞

### オープンソースプロジェクト

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  画像認識に基づく自動化ブラックボックステストフレームワーク
- [MXU](https://github.com/MistEO/MXU)
  MaaFramework Next UI
- [MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)
  『アークナイツ』小助手 — デイリー作業をワンクリックで自動化！
- [MaaEnd](https://github.com/MaaEnd/MaaEnd)
  視覚 AI ベースの『アークナイツ：エンドフィールド』自動化ツール
- [M9A](https://github.com/MAA1999/M9A)
  『リバース：1999』小助手
- ~~[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  Avalonia UI で構築された MaaFramework 汎用 GUI ソリューション~~

### 貢献者

テストと開発に参加してくれたすべての開発者に感謝します (´▽`ʃ♡ƪ)

[![Contributors](https://contributors-img.web.app/image?repo=1bananachicken/MaaNTE&max=1000)](https://github.com/1bananachicken/MaaNTE/graphs/contributors)

## ☕コーヒーをおごる

MaaNTE があなたの時間を節約してくれたなら、開発者にコーヒーをおごってみませんか？

あなたのサポートが、継続的な更新への最大の原動力です 🥰

[<img width="200" alt="スポンサー" src="https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png">](https://afdian.com/a/MaaNTE)

## ⭐Star History

このソフトウェアが役に立ったと思ったら、ぜひ Star を押してください！（ページ右上の小さな星）— それが私たちへの最大のサポートです！

<a href="https://www.star-history.com/?repos=1bananachicken%2FMaaNTE&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=1bananachicken/MaaNTE&type=date&legend=top-left" />
 </picture>
</a>
