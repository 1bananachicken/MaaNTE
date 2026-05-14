# MaaNTE Problem Troubleshooting Guide
 This text was translated using a translation tool; please let me know if you spot any errors.

## operational issues
### cannot start
>A pop-up window displays the message: `To run this application, you must install .NET`.
>> Go and download [.NET 10.0 ](https://dotnet.microsoft.com/zh-tw/download/dotnet/thank-you/sdk-10.0.300-windows-x64-installer) and install ` .NET 10.0 Desktop Runtime` .

### Unable to Connect to Window
1. Make sure you have opened Neverness to Everness(NTE).
2. Make sure you are running MaaNTE as an administrator.
3. Set NTE to a resolution of `1280x720` and run it in windowed mode.


## Task Issues
### General
1. I have fewer features than others
1-1. Some features only appear when a specific controller is selected; try switching controllers.
1-2. The controller switching feature on the home page is currently malfunctioning; please change it in `Settings > Connection Settings`. The foreground controller is the mouse `Seize`, and the background controller is the mouse `SendMessageWithWindowPos`.

2. Task completes immediately upon launch
2-1. Check if the corresponding task is checked in the `Task List` on the left.
2-2. Disable in-game features that affect image quality, such as frame interpolation and super-resolution.

3. Unable to connect to a window or start a task
3-1. Ensure you have added MaaNTE to your antivirus software’s whitelist, or temporarily disable your antivirus software.
3-2. Ensure the game language is set to Simplified Chinese.

4. Unable to click or perform operations normally
4-1. Ensure your MaaNTE is located in a path consisting solely of English characters and contains no full-width characters (it’s best to avoid any special characters as well).
4-2. Ensure you are running MaaNTE as an administrator.
4-3. If you cannot click normally, try changing the mouse input mode to `Seize`.
4-4. Ensure Windows screen scaling is set to 100%.

5. Mouse Seizure Issues
5-1. In `Settings > Connection Settings`, change the mouse mode to `SendMessageWithWindowPos`. However, some tasks that require a foreground controller need `Seize` (which will seize the mouse).

6. Recognition Failure
6-1. Disable in-game features that affect image quality, such as frame interpolation and super resolution.


## Additional Issues
### Auto-Fishing
1. Unable to start fishing
1-1. Refer to the section above: `Task Issues - General - Unable to connect to a window or start a task`

2. Rod does not cast automatically
2-1. Refer to the section above: `Task Issues - General - Unable to click or perform operations normally`

3. Fish is not reeled in using the A/D keys
3-1. Refer to the section above: `Task Issues - General - Unable to click or perform operations normally`

4. Unable to sell catch
4-1. Try setting the game to 120 FPS.
4-2. The beta version is currently testing a new solution; please stay tuned.

5. Unable to purchase bait
5-1. Lower the `Bait Detection Threshold`.

6. Cannot catch fish
6-1. The beta version is currently testing a new solution; please stay tuned.

7. Fishing quests end prematurely
7-1. Ensure you have enough bait left for each round of fishing.


### Real-time Assistant
1. Window keeps moving around
1-1. You must use the foreground controller (go to `Settings > Connection Settings` and set the mouse to `Seize`).

2. Auto-Story cannot skip “Important Story” prompts
2-1. We are planning to add this feature; please stay tuned.


### Auto-Coffee
1. Essentially, this automatically attacks everyone.

2. No rewards / No full combos
2-1. Requires Nanari and Shirakura’s City Skills.
