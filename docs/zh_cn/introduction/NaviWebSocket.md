# Navi 本地 WebSocket

`online_map_navigation` custom action 会在同一个截图循环中执行 NCC 定位、方向预测和路径寻路。它会把实时位置和方向广播给本地地图前端，也会通过同一个双向 WebSocket 接收在线地图工具发送的路径点并依序寻路。Maa 同时只能运行一个 action，因此在线地图实时定位和路径寻路共用这个组合 action：

```json
{
  "OnlineMapNavigation": {
    "action": "Custom",
    "custom_action": "online_map_navigation",
    "custom_action_param": {
      "port": 14514,
      "tolerance": 5,
      "frame_interval": "0.1",
      "position_backend": "auto",
      "debug": false,
      "angle_backend": "auto"
    }
  }
}
```

该入口位于 `assets/resource/base/pipeline/OnlineMapNavigation.json`，任务设置位于 `assets/resource/tasks/OnlineMapNavigation.json`。

服务固定监听 `0.0.0.0`，不可通过任务设置或 Pipeline 参数修改。客户端连接地址：

```text
ws://127.0.0.1:14514
```

可在任务设置中覆盖端口、采样间隔、到达容差和调试模式。定位方式支持 `auto`、`coordinate`、`map`，默认 `auto`；方向推理后端支持 `auto`、`cpu`、`directml`。采样间隔单位为秒，最低限制为 `0.05` 秒。

消息格式：

```json
{
  "type": "navi-state",
  "version": 1,
  "position": {
    "x": -134394.56,
    "y": 199913.53,
    "z": 11416.17,
    "pixelX": 4090,
    "pixelY": 6750,
    "sourceWidth": 11264,
    "sourceHeight": 11264,
    "score": 1.0,
    "mode": "coordinate"
  },
  "angle": 123.4,
  "angleConfidence": 0.96,
  "timestamp": 1770000000.0
}
```

`position` 使用游戏原始世界坐标：

- 网络定位直接发送 `x`、`y`、`z`。
- 视觉定位通过标定变换逆算并发送 `x`、`y`，不发送 `z`。
- 为兼容旧版网页，定位成功时同时发送 `pixelX`、`pixelY`、`sourceWidth`、`sourceHeight`。新客户端应使用原始坐标字段。
- 网络坐标暂时中断时保留并发送最后一次坐标，`mode` 为 `coordinate_stale`。
- 没有可转换的位置时 `position` 为 `null`。

方向未识别时 `angle` 为 `null`。WebSocket 服务启动后页面会自动重连；未收到路径点时，该任务仍会持续广播实时定位状态。
