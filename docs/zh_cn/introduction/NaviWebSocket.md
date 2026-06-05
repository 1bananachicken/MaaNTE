# Navi 本地 WebSocket

`navi_websocket` custom action 会在同一个截图循环中执行 NCC 定位和方向预测，再把两个结果作为同一条状态广播给本地地图前端。Maa 同时只能运行一个 action，因此实时地图应直接运行组合 action：

```json
{
  "NaviWebSocket": {
    "action": "Custom",
    "custom_action": "navi_websocket",
    "custom_action_param": {
      "host": "0.0.0.0",
      "port": 14514,
      "debug": false,
      "frame_interval": "0.1",
      "angle_backend": "auto"
    }
  }
}
```

该入口位于 `assets/resource/base/pipeline/NaviWebSocket.json`，任务设置位于 `assets/resource/tasks/NaviWebSocket.json`。

默认监听地址：

```text
ws://127.0.0.1:14514
```

可在任务设置中覆盖监听地址、端口、采样间隔、调试模式，并在 `auto`、`cpu` 和 `directml` 三个方向推理后端之间选择。

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

当某一帧没有识别到位置或方向时，对应字段为 `null`。WebSocket 服务在首次产生 Navi 结果时启动；页面会自动重连。

`sourceWidth` 和 `sourceHeight` 表示 NCC 底图尺寸。前端会按自身在线地图尺寸缩放坐标，例如 `11264 x 11264` 的 NCC 底图坐标映射到 `22528 x 22528` 在线地图时会放大 2 倍。
