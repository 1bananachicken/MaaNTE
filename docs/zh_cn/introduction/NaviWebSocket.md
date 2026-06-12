# Navi 本地 WebSocket

`online_map_navigation` custom action 会在同一个截图循环中执行 NCC 定位、方向预测和路径寻路。它会把实时位置和方向广播给本地地图前端，也会通过同一个双向 WebSocket 接收在线地图工具发送的路径点并依序寻路。Maa 同时只能运行一个 action，因此在线地图实时定位和路径寻路共用这个组合 action：

```json
{
  "OnlineMapNavigation": {
    "action": "Custom",
    "custom_action": "online_map_navigation",
    "custom_action_param": {
      "host": "0.0.0.0",
      "port": 14514,
      "tolerance": 5,
      "frame_interval": "0.1",
      "debug": false,
      "angle_backend": "auto"
    }
  }
}
```

该入口位于 `assets/resource/base/pipeline/OnlineMapNavigation.json`，任务设置位于 `assets/resource/tasks/OnlineMapNavigation.json`。

默认监听地址：

```text
ws://127.0.0.1:14514
```

可在任务设置中覆盖监听地址、端口、采样间隔、到达容差、调试模式，并在 `auto`、`cpu` 和 `directml` 三个方向推理后端之间选择。采样间隔单位为秒，最低限制为 `0.05` 秒。

消息格式：

```json
{
  "type": "navi-state",
  "version": 1,
  "position": {
    "pixelX": 5788,
    "pixelY": 8902,
    "score": 0.82,
    "mode": "local",
    "sourceWidth": 11264,
    "sourceHeight": 11264
  },
  "angle": 123.4,
  "angleConfidence": 0.96,
  "timestamp": 1770000000.0
}
```

当某一帧没有识别到位置或方向时，对应字段为 `null`。WebSocket 服务启动后页面会自动重连；未收到路径点时，该任务仍会持续广播实时定位状态。

`sourceWidth` 和 `sourceHeight` 表示 NCC 底图尺寸。前端会按自身在线地图尺寸缩放坐标，例如 `11264 x 11264` 的 NCC 底图坐标映射到 `22528 x 22528` 在线地图时会放大 2 倍。
