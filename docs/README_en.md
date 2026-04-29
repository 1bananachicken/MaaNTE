<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img alt="LOGO" src="../assets/logo.png" width="256" height="256" />
</p>
<div align="center">

# MaaNTE
😊 MAA Assistant for NTE 😊

Powered by [MaaFramework](https://github.com/MaaXYZ/MaaFramework)!

This project is in early development. PRs and Issues are welcome!

</div>

## ✨ Features
- 🎣 Auto Fishing
  - 🐟 Auto Sell Fish
  - 🪝 Auto Buy Bait
- 🥤 Auto Coffee Brewing
- 💰 Auto Extract Café Revenue

## ❓ FAQ

### 🤔 Can't find how to launch?
- Regular users should download the **release version**. Only clone the repo if you have development needs.

### 🤔 Fishing not working?
- Make sure the program is run as **administrator**
- Check that the game resolution is set to **1280×720**
- Make sure **Auto Fishing** is enabled
- Do not place the program in a **path containing Chinese characters**
- Try **disabling your antivirus** software

### 🤔 Pop-up in the bottom right says Mirror is not supported?
This is due to auto-update not being configured. It does not affect functionality — you can ignore it.

## 💡 Notes
- The game must run at **1280×720 resolution** in **windowed mode**
- The new fishing algorithm supports running at **120 FPS**
- Auto coffee brewing requires setting mouse input to **Seize** mode — this will temporarily take over your mouse *(a non-seize method is being tested and will be released in beta first)*

## 💻 Development Guide

1. Fork the project
   - Click `Fork`, then click `Create Fork`

2. Clone your forked repo locally and pull submodules
```bash
git clone --recursive https://github.com/<your-username>/MaaNTE.git
```

3. Download the MaaFramework [release package](https://github.com/MaaXYZ/MaaFramework/releases) and extract it into the `deps` folder

4. Set up the environment
   - Install **Python >= 3.11**
   - **VSCode** is recommended as your IDE
   - Strongly recommended: install the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=nekosu.maa-support) for debugging

5. Submit a PR
   - New features should be submitted to the **dev branch**

For more development documentation, refer to the [M9A docs](https://1999.fan/zh_cn/develop/development.html)

## ☕ Credits

### Open Source Libraries
- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)
  Image-recognition-based automation black-box testing framework
- [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)
  Universal GUI solution for MaaFramework built with Avalonia UI
- [M9A](https://github.com/MAA1999/M9A)
  Code and documentation reference
