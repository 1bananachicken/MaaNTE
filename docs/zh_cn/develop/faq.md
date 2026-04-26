# FAQ

## 0. 我是第一次使用 git，这是什么？视频演示中那个黑框框命令行哪来的？

黑框框是 git bash，几乎任何现代软件的开发都离不开 git，建议先参考 [菜鸟教程](https://www.runoob.com/git/git-install-setup.html) 或搜索一些视频，学习完 git 后再来进行后续开发工作。

## 1. 我是第一次使用 Python，在命令行输入 `python ./configure.py` 或 `python -m pip install MaaFW` 之后没有反应？没有报错，也没有提示成功，什么都没有

Win10 或者 Win11 系统自带了一份 "Python"，但它其实只是一个安装器，是没法用的。  
你需要做的是关闭它或者删除它的环境变量，然后自己去 Python 官网下载并安装一份 Python。  
[参考方法](https://www.bilibili.com/read/cv24692025/)

## 2. 使用 MaaDebugger 或 MaaPiCli 时弹窗报错，应用程序错误：应用程序无法正常启动

![缺少运行库](https://github.com/user-attachments/assets/942df84b-f47d-4bb5-98b5-ab5d44bc7c2a)

一般是电脑缺少某些运行库，请安装一下 [vc_redist](https://aka.ms/vs/17/release/vc_redist.x64.exe) 。

## 3. 我应该如何打包我的项目？

你需要按照项目推荐的[开发流程](./how_to_develop.md)发布一个版本，[CI](/.github/workflows/install.yml) 会自动完成打包工作。具体的工作方式请参考 [GitHub Actionss 文档](https://docs.github.com/zh/actions)。

## 4. 我在这个仓库里提了 Issue 很久没人回复

这里是《项目模板》仓库，它仅仅是一个模板，一般很少会修改，开发者也较少关注。  
在此仓库请仅提问模板相关问题，其他问题最好前往对应的仓库提出，如果有 log，最好也带上它（`debug/maa.log` 文件）

- MaaFW 本身及 MaaPiCli 的问题：[MaaFramework/issues](https://github.com/MaaXYZ/MaaFramework/issues)
- MaaDebugger 的问题：[MaaDebugger/issues](https://github.com/MaaXYZ/MaaDebugger/issues)
- 不知道算是哪里的、其他疑问等：[讨论区](https://github.com/MaaXYZ/MaaFramework/discussions)

## 5. OCR 文字识别一直没有识别结果，报错 "Failed to load det or rec", "ocrer_ is null"

**请仔细阅读文档**，你无视了前面步骤的报错。我不想解释了，请再把本文档仔细阅读一遍！

## 6. 在开发过程中遇到了其他问题

闭门造车大概率无法解决任何问题，你可以加入 [MaaFramework 开发交流群](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=9sleK6URhEG0v3QeTmpFueCjF26wibEH&authKey=LBZc5FxWa3M%2BiWj3rpBfRmqg9PD9jJNaxpp3xTqTcGxsp1Am3kd1uzxQXiP4w8w4&noverify=0&group_code=595990173) 以寻求帮助。

> [!WARNING]
> 在提问前，请完整阅读 [MaaFramework 开发文档](https://maafw.com/docs/1.1-QuickStarted)以及[如何开发](./how_to_develop.md)，通常情况下他们能解决大多数问题。
> 在提问时，请 **具体指出文档中困惑的章节** / **分享你的项目文件中的具体内容** / **提供完整的报错信息** ，不然你能得的大概率只有类似 _“请先读文档”_ 这样的回答。
