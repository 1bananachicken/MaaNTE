<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img alt="LOGO" src="./assets/logo.png" width="256" height="256" />
</p>

<div align="center">

[English](docs/README_en.md)

# MaaNTE

😊MAA异环小助手😊

由 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 强力驱动！

本项目还处于早期开发阶段，欢迎提交PR和Issue

QQ交流1群: 1103323319

若1群已满请加2群: 1101147419

</div>

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

自动做咖啡需要设置鼠标输入方式 Seize ，会出现抢占鼠标的情况（正在测试非抢占鼠标的方法，将先在测试版发布）

## 💻开发指南
1. Fork项目
- 点击 `Fork`，继续点击 `Create Fork`
2. 克隆自己 fork 的项目到本地，并拉取子模块
```bash
git clone --recursive https://github.com/<你的用户名>/MaaNTE.git
```
3. 下载MaaFramework的 [release包](https://github.com/MaaXYZ/MaaFramework/releases)，解压到 `deps` 文件夹中
4. 配置环境
- 安装 python>=3.11
- 建议使用vscode作为IDE进行开发
- 强烈建议安装 [vscode插件](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) 进行调试
5. 提交PR
  - 新功能开发请提交到dev分支

更多开发文档可以参考 [M9A文档站](https://1999.fan/zh_cn/develop/development.html)

## ☕鸣谢

### 开源库
- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  基于图像识别的自动化黑盒测试框架
- [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  基于 Avalonia UI 构建的 MaaFramework 通用 GUI 解决方案
- [M9A](https://github.com/MAA1999/M9A)
  代码以及文档参考
