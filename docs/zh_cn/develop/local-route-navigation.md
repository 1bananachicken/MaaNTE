# 本地路线寻路接口

Navi 由独立的 C++ Agent 实现，源码位于 `agent/cpp-navi/`。Python Agent 继续承载其他 CustomAction，MaaFramework 会根据 CustomAction 注册名把任务分发给对应 Agent。

## Pipeline 入口

正式入口为 `local_route_navigation`：

```jsonc
{
    "GoToFountain": {
        "action": "Custom",
        "custom_action": "local_route_navigation",
        "custom_action_param": {
            "json_path": "penquan",
            "route_name": "penquan",
            "segment_index": 1,
            "tolerance": 5,
            "frame_interval": 0.1,
            "angle_backend": "auto",
            "position_backend": "auto",
            "debug": false
        }
    }
}
```

`local_route_navigation_unit_test` 使用相同的 C++ 实现，但固定加载 `penquan` 路线，供节点调试使用。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `json_path` | string | 必填 | 路线 JSON 路径。相对路径会在 `assets/resource/routes/` 中查找，可省略 `.json`。 |
| `route_name` | string | `""` | 匹配 route 的 `name` 或 `id`；为空时使用第一个 route。 |
| `segment_index` | int | `1` | 路线段序号，从 1 开始。 |
| `tolerance` | float | `5.0` | 到达判定距离。 |
| `frame_interval` | float | `0.1` | 视觉模式的控制间隔，最低为 `0.05` 秒；网络姿态模式固定以 60 Hz 运行。 |
| `angle_backend` | string | `"auto"` | ONNX 方向模型后端：`auto`、`directml` 或 `cpu`。 |
| `position_backend` | string | `"auto"` | `map` 使用视觉定位；`coordinate` 严格使用网络坐标；`auto` 按 Npcap、pktmon、视觉顺序选择。 |
| `debug` | bool | `false` | 输出定位和方向调试图到 `debug/cpp-navi/`。 |

动作返回成功表示路线段已到达终点。参数错误、路线为空、定位后端不可用、运行异常或任务停止会返回失败。

## 路线 JSON

路线文件推荐放在 `assets/resource/routes/`。支持在线地图导出的 route/segment 结构：

```jsonc
{
    "version": 1,
    "routes": [
        {
            "id": "route-1",
            "name": "main",
            "segments": [
                {
                    "points": [
                        { "lat": 51.9, "lng": -36.0 },
                        { "lat": 51.8, "lng": -29.0 }
                    ]
                }
            ]
        }
    ]
}
```

也支持简单路径列表或包含 `points`、`waypoints`、`path` 的对象。

| 字段 | 说明 |
|------|------|
| `pixelX` / `pixelY` | 11264 x 11264 底图像素坐标。 |
| `target_x` / `target_y` | 像素坐标别名。 |
| `x` / `y`（可选 `z`） | 游戏原始世界坐标，使用 C++ 标定变换转为地图像素。 |
| `lat` / `lng` | maante-map 路线坐标，按在线地图尺寸转换。 |

`sourceWidth/sourceHeight` 或 `sourceSize` 可声明像素坐标来源尺寸；未声明时按 11264 x 11264 解析。

## 运行机制

C++ Agent 的主要组件如下：

- `PositionProvider`：统一管理网络坐标和视觉地图定位。
- `MapLocator`：从 1280 x 720 截图裁剪小地图，执行多尺度带遮罩 NCC 匹配。
- `AnglePredictor`：使用 `pointer_model.onnx` 推理朝向。
- `WaypointNavigator`：PID 转向、前进按键和到达判定。
- `RouteSession` / `RouteRunner`：路线状态和路径点推进。

网络定位启用后，位置和摄像机朝向来自同一数据包，不再运行视觉定位和方向模型。样本暂时中断时状态为 `coordinate_stale`，导航不会使用过期位置。

## 构建

首次构建需要初始化 MaaUtils 子模块并下载 MaaDeps：

```powershell
git submodule update --init --recursive
python tools/ci/download_maadeps.py
cmake -S agent/cpp-navi -B agent/cpp-navi/build -G "Visual Studio 17 2022" -A x64
cmake --build agent/cpp-navi/build --config RelWithDebInfo
```

主发布脚本会自动构建并打包 C++ Agent：

```powershell
python build.py --mode=mxu
```

MXU 包中 `interface.json` 同时声明 Python Agent 和 C++ Navi Agent。C++ 可执行文件及其运行库位于 `agent/cpp-navi/`。

## 调试与标定

`debug=true` 不打开 GUI 窗口，而是持续覆盖写入：

- `debug/cpp-navi/map_locator.png`
- `debug/cpp-navi/minimap.png`
- `debug/cpp-navi/angle_predictor.png`

网络坐标后端优先使用已安装的 Npcap；不可用时尝试 `pktmon_backend.dll`。pktmon 通常需要管理员权限，DLL 由 Python 依赖 `pktmon-interface` 提供。

标定数据位于本地 `config/navi_coordinate_calibration.json`。修改后运行：

```powershell
python scripts/update_navi_coordinate_transform.py
```

脚本会拟合平面相似变换，并更新 `agent/cpp-navi/src/coordinate.cpp` 中的生成常量。至少需要 3 个相距较远且不共线的标定点。
