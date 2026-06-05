# 在线地图实时定位

`MapWebViewLocator` 会打开在线地图，并在网页原生 Leaflet 图层中显示角色位置和朝向。它打开的主 WebView 只负责面向用户展示地图，不提供调试状态栏，也不在网页内执行标定。需要标定时，单独运行 `MapWebViewCalibration` 打开专用标定 WebView。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `agent/custom/action/map_webview/locator/action.py` | 截图循环、NCC 定位、方向预测、坐标转换、本地只读状态服务和 WebView 生命周期。 |
| `agent/custom/action/map_webview/locator/window.py` | 打开在线地图、轮询只读状态并注入展示脚本。 |
| `agent/custom/action/map_webview/locator/overlay.js` | 查找 Leaflet 地图、创建 marker、更新位置和旋转角度。 |
| `agent/custom/action/map_webview/calibration/action.py` | 独立标定 custom action、实时 NCC 定位、坐标拟合、标定文件保存和重置。 |
| `agent/custom/action/map_webview/calibration/window.py` | 打开专用标定窗口、轮询标定状态并提交网页点击结果。 |
| `agent/custom/action/map_webview/calibration/overlay.js` | 在专用标定窗口内采集 Leaflet 点击坐标并显示操作提示。 |
| `assets/resource/base/image/map/map_webview_pointer.png` | 网页地图使用的方向指针。 |
| `config/map_webview_calibration.json` | 本机标定数据。该文件属于运行时配置，不提交到仓库。 |

## 运行入口

`assets/resource/base/pipeline/MapLocator.json` 提供两个 custom action：

```json
{
  "MapWebViewLocator": {
    "action": "Custom",
    "custom_action": "map_webview_locator"
  },
  "MapWebViewCalibration": {
    "action": "Custom",
    "custom_action": "map_webview_calibration"
  }
}
```

运行 `MapWebViewLocator` 时，程序会从 Maa controller 获取游戏截图，使用 `MapLocatorNcc` 计算本地 `map.jpg` 坐标，使用 `AnglePredictor` 计算朝向，再将坐标转换为 Leaflet 坐标发送给 WebView。

WebView 仅轮询：

```text
GET /state.json
```

状态内容只包含展示所需字段：

```json
{
  "onlinePoint": [7.984375, 59.09375],
  "angle": 123.4
}
```

## 标定

本地 `map.jpg` 像素坐标和在线地图 Leaflet 坐标需要通过标定建立映射。每个标定点格式如下：

```json
{
  "local": [14034.0, 9768.0],
  "online": [7.984375, 59.09375]
}
```

至少需要三个有效标定点。建议在不同方向上采样，并使用六个以上点降低误差。

标定由 `map_webview_calibration` custom action 独立处理。直接运行 `MapWebViewCalibration` 时会打开专用标定 WebView：

1. 在游戏中移动到容易确认的地标。
2. 等待标定窗口提示已经识别到游戏位置。
3. 在在线地图中找到同一地标。
4. 按住 `Shift` 点击在线地图中的对应位置。
5. 移动到其他地标并重复操作，至少采集三个有效点。
6. 如需清空旧点，按住 `Ctrl + Shift` 点击标定地图。

主定位窗口 `MapWebViewLocator` 仍然只用于用户查看实时位置，不包含标定入口或调试状态栏。

如需脚本化维护标定文件，也可以通过 `pair` 增加一个点：

```json
{
  "pair": {
    "local": [14034.0, 9768.0],
    "online": [7.984375, 59.09375]
  }
}
```

也可以通过 `pairs` 批量增加或替换邻近点：

```json
{
  "pairs": [
    {
      "local": [14034.0, 9768.0],
      "online": [7.984375, 59.09375]
    }
  ]
}
```

设置 `"replace": true` 会先清空旧点再写入新点。设置 `"reset": true` 会清空标定文件。

标定文件默认路径：

```text
config/map_webview_calibration.json
```

`MapWebViewLocator` 启动时只读取标定文件，不会修改它。也可以通过 `online_transform` 直接传入六个转换系数。

## 参数

`MapWebViewLocator` 支持以下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `map_url` | `https://www.ghzs666.com/yh-map#/` | 在线地图地址。 |
| `update_interval` | `0.1` | 状态更新最小间隔，单位为秒。 |
| `title` | `MaaNTE Online Map` | WebView 标题。 |
| `width` | `1280` | WebView 宽度。 |
| `height` | `820` | WebView 高度。 |
| `webview_debug` | `false` | 是否启用 pywebview 调试模式。 |
| `pointer_image` | 内置指针 | 自定义指针图片路径。 |
| `calibration_path` | `config/map_webview_calibration.json` | 标定文件路径。 |
| `online_transform` | 无 | 手工指定六个转换系数。 |
| `big_map_path` | 内置 `map.jpg` | 自定义本地大地图路径。 |
| `angle_backend` | 环境变量或 `cpu` | ONNX 方向模型后端。 |
| `pointer_roi` | `[73, 60, 64, 64]` | 方向模型截图区域。 |
| `angle_threshold` | `0.0` | 方向结果最低置信度。 |

## 验证

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  agent\custom\action\map_webview\calibration\action.py `
  agent\custom\action\map_webview\calibration\window.py `
  agent\custom\action\map_webview\locator\action.py `
  agent\custom\action\map_webview\locator\window.py `
  agent\custom\action\__init__.py

node --check agent\custom\action\map_webview\locator\overlay.js
node --check agent\custom\action\map_webview\calibration\overlay.js

git diff --check
```
