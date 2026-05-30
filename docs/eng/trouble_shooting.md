# MaaNTE Troubleshooting Guide

This page is intended to help you quickly identify common setup, connection, and task issues. If you notice anything inaccurate or unclear, corrections are welcome.

## Running Issues

### Cannot Start

#### A popup says `To run this application, you must install .NET`

- Download and install the [.NET 10.0 Desktop Runtime](https://dotnet.microsoft.com/zh-cn/download/dotnet/10.0).

### Unable to Connect to the Game Window

1. Make sure Neverness to Everness is already open.
2. Make sure you are running MaaNTE as an administrator.
3. Set the game to `1280x720` resolution and windowed mode.

## Task Issues

### General

#### I have fewer features than others

- Some features only appear after selecting a specific controller, so try switching controllers.
- The controller switching option on the home page is currently unreliable. Please change it in `Settings > Connection Settings`. The foreground controller is mouse `Seize`, and the background controller is mouse `SendMessageWithWindowPos`.

#### A task finishes immediately after launch

- Check whether the corresponding task is enabled in the `Task List` on the left.
- Disable in-game features that affect image quality, such as frame interpolation and super-resolution.

#### Unable to connect to a window or start a task

- Make sure MaaNTE is added to your antivirus whitelist, or temporarily disable your antivirus software.
- Make sure the game language is set to Simplified Chinese.

#### Unable to click or perform operations normally

- Make sure MaaNTE is installed in a path that contains only English characters and no full-width characters. It is best to avoid special characters too.
- Make sure you are running MaaNTE as an administrator.
- If clicks do not work properly, try changing the mouse input mode to `Seize`.
- Make sure Windows display scaling is set to 100%.

#### Mouse capture issues

- In `Settings > Connection Settings`, change the mouse mode to `SendMessageWithWindowPos`. However, some tasks that require a foreground controller still need `Seize`, which will capture the mouse.

#### Recognition failure

- Disable in-game features that affect image quality, such as frame interpolation and super-resolution.

## Additional Issues

### Auto-Fishing

#### Cannot start fishing

- See [Task Issues - General - Unable to connect to a window or start a task](#unable-to-connect-to-a-window-or-start-a-task)

#### The rod does not cast automatically

- See [Task Issues - General - Unable to click or perform operations normally](#unable-to-click-or-perform-operations-normally)

#### The fish is not reeled in with the A/D keys

- See [Task Issues - General - Unable to click or perform operations normally](#unable-to-click-or-perform-operations-normally)

#### Unable to sell catch

- Try setting the game to 120 FPS.
- The beta version is currently testing a new solution; please stay tuned.

#### Unable to purchase bait

- Lower the `Bait Detection Threshold`.

#### Cannot catch fish

- The beta version is currently testing a new solution; please stay tuned.

#### Fishing quests end prematurely

- Make sure you still have enough bait for each round of fishing.

### Real-time Assistant

#### The window keeps moving around

- You must use the foreground controller. Go to `Settings > Connection Settings` and set the mouse to `Seize`.

#### Auto-Story cannot skip “Important Story” prompts

- We are planning to add this feature; please stay tuned.

### Auto-Coffee

#### Essentially, this automatically attacks everyone

#### No rewards / No full combos

- Requires Nanari and Shirakura’s City Skills.
