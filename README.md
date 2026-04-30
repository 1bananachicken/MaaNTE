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
> QQ交流1群: 1103323319
> 若1群已满请加2群: 1101147419

> [!Caution]
>
> 近期出现大量账号二次分发本软件，甚至冒充官方账号。通过非官方途径获取的软件可能**含有病毒**，并且一般不是最新的版本，请注意甄别。

## ✨ 功能一览

- 🎣 自动钓鱼
  - 🐟 自动卖鱼
  - 🪝 自动买鱼饵
- 🥤 自动做咖啡
  - 🔨 驱赶所有顾客
- 💰 自动提取一咖舍收益
  - 📦 自动补货

## ❓常见问题

### 🤔找不到怎么启动？

- 普通用户请下载release版本，有开发需求再clone仓库

### 🤔钓鱼不工作怎么办？

- 是否以管理员身份运行
- 游戏分辨率是否调整到1280*720
- 是否勾选了自动钓鱼
- 不要把程序放在中文路径下
- 关闭杀毒软件

### 🤔右下角弹窗提示不支持Mirror酱？

这是自动更新没有配置，不影响，不用管

## 💡注意事项

游戏需要运行在1280x720分辨率，窗口化，钓鱼新算法支持120帧运行

自动做咖啡需要设置鼠标输入方式 `Seize` ，会出现抢占鼠标的情况（正在测试非抢占鼠标的方法，将先在测试版发布）

## 💻开发指南

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

### 开源库

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  基于图像识别的自动化黑盒测试框架
- [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  基于 Avalonia UI 构建的 MaaFramework 通用 GUI 解决方案
- [M9A](https://github.com/MAA1999/M9A)
  代码以及文档参考

### ❤️贡献/参与者

感谢全部参与到测试与开发中的开发者(´▽`ʃ♡ƪ)

[![Contributors](https://contributors-img.web.app/image?repo=1bananachicken/MaaNTE&max=105&columns=15)](https://github.com/1bananachicken/MaaNTE/graphs/contributors)

## 广告

如果觉得软件对你有帮助，帮忙点个 Star 吧！~（网页最上方右上角的小星星），这就是对我们最大的支持了！
