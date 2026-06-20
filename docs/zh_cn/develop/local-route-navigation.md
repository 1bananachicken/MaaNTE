# 本地路线寻路接口

本地路线寻路位于 `agent/custom/action/Navi/`，用于根据地图定位和方向推理，把角色移动到路线 JSON 中定义的路径点。它有两种入口：

- Pipeline 入口：`local_route_navigation` CustomAction，一次执行一个 route segment。
- Python 入口：`LocalRouteNavigation` 类，适合在同一个函数中连续执行多个 segment，避免重复初始化定位器和方向模型。

## Pipeline 调用

注册名：`local_route_navigation`

基础节点定义见 `assets/resource/base/pipeline/LocalRouteNavigation.json`：

```jsonc
{
    "LocalRouteNavigation": {
        "action": "Custom",
        "custom_action": "local_route_navigation",
        "custom_action_param": {
            "json_path": "",
            "route_name": "",
            "segment_index": 1,
            "frame_interval": 0.1,
            "tolerance": 5,
            "angle_backend": "auto",
            "debug": false
        }
    }
}
```

同一文件还提供了一个测试节点 `LocalRouteNavigationUnitTest`，用于直接调用 Python 类接口执行喷泉路线：

```jsonc
{
    "LocalRouteNavigationUnitTest": {
        "action": "Custom",
        "custom_action": "local_route_navigation_unit_test",
        "custom_action_param": {
            "frame_interval": 0.1,
            "tolerance": 5,
            "angle_backend": "auto",
            "debug": false
        }
    }
}
```

`local_route_navigation_unit_test` 内部固定加载 `penquan` 路线，并执行 `route_name="penquan"`、`segment_index=1`。它用于验证 `LocalRouteNavigation` Python 类接口，不用于正式任务流程。

在其他 Pipeline 中复用时，直接覆盖 `custom_action_param`：

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
            "debug": false
        }
    }
}
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `json_path` | string | 必填 | 路线 JSON 路径。可以传绝对/相对路径；如果文件不存在，会到 `assets/resource/routes/` 下按文件名查找，省略 `.json` 也可以。 |
| `route_name` | string | `""` | 多 route 文件中的 route 名称或 id。为空时使用第一个 route。 |
| `segment_index` | int | `1` | 要执行的 segment。对外按 1 开始编号，`1` 表示第一个 segment。 |
| `tolerance` | float | `5.0` | 到达判定距离。当前位置到目标点距离小于等于该值时认为到达。 |
| `frame_interval` | float | `0.1` | 截图、定位和方向推理的采样间隔，最低会限制为 `0.05` 秒。 |
| `angle_backend` | string | `"auto"` | 方向模型推理后端。常用值：`auto`、`directml`、`cpu`。 |
| `position_backend` | string | `"auto"` | 位置来源。`map` 使用小地图匹配；`auto` 优先使用网络定位，网络核心无法启用时回退地图；`coordinate` 严格要求网络核心可用。网络定位成功启用后不会再执行位置视觉匹配。 |
| `debug` | bool | `false` | 是否打开定位和方向预测调试窗口，并输出原始网络坐标、映射坐标及定位来源日志。 |

CustomAction 返回 `success=True` 表示该 segment 跑到终点；参数错误、路线为空、运行异常或任务停止会返回失败。

## Python 类接口

当一个 Python CustomAction 里需要连续执行多个 segment 时，使用 `LocalRouteNavigation`。不要在同一个函数里反复调用 Pipeline 入口，否则每次都会重新初始化定位器和方向模型。

```python
from .Navi.local_route_navigation import LocalRouteNavigation


with LocalRouteNavigation(
    context,
    "test",
    tolerance=5.0,
    frame_interval=0.1,
    angle_backend="auto",
    debug=False,
) as navigator:
    navigator.run_route(route_name="1", segment_index=1)
    navigator.run_route(route_name="1", segment_index=2)
```

### 构造参数

```python
LocalRouteNavigation(
    context,
    route_json=None,
    *,
    tolerance=5.0,
    angle_backend="auto",
    frame_interval=0.1,
    debug=False,
)
```

`route_json` 可以是路径、`dict` 或 `list`。传路径时会在类初始化时加载整份 JSON；传 `dict/list` 时直接使用内存数据。

### 方法

```python
navigator.load_route_json(route_json)
```

加载或替换当前实例持有的路线 JSON。

```python
navigator.run_route(route_json=None, *, route_name="", segment_index=1) -> bool
```

执行指定路线段。`route_json` 为空时使用实例已加载的 JSON；传入路径或内存 JSON 时只对本次执行生效。返回 `True` 表示到达终点。

```python
navigator.close()
```

释放按键状态、调试窗口和底层导航资源。推荐用 `with LocalRouteNavigation(...) as navigator:` 自动释放。

## 路线 JSON 格式

路线文件推荐放在 `assets/resource/routes/`。`json_path` 传 `test` 时会解析到 `assets/resource/routes/test.json`。

### 多 route / segment 格式

这是在线地图导出的主要格式：

```jsonc
{
    "version": 1,
    "routes": [
        {
            "id": "route-1",
            "name": "main",
            "segments": [
                {
                    "id": "segment-1",
                    "name": "1",
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

`route_name` 会匹配 route 的 `name` 或 `id`。`segment_index` 按数组顺序选择 segment。

### 简单路径格式

简单列表或单 route 对象也可以：

```jsonc
[
    { "pixelX": 1000, "pixelY": 2000 },
    { "pixelX": 1200, "pixelY": 2300 }
]
```

```jsonc
{
    "sourceWidth": 11264,
    "sourceHeight": 11264,
    "points": [
        { "x": 1000, "y": 2000 },
        { "x": 1200, "y": 2300 }
    ]
}
```

支持的路径点字段：

| 字段 | 说明 |
|------|------|
| `pixelX` / `pixelY` | 像素坐标。 |
| `target_x` / `target_y` | 像素坐标别名。 |
| `x` / `y` | 像素坐标。`coordinate` 不是 `online` 时生效。 |
| `lat` / `lng` | maante-map 路线保存的 world 坐标，会转换为当前地图像素坐标。 |

`lat/lng` 的转换与 maante-map 保持一致：`worldOriginPixel=(11264,11264)`，`pixelsPerWorldUnit=44`，在线地图尺寸为 `22528 x 22528`，再缩放到当前 NCC 底图尺寸。像素坐标的缩放由 `sourceWidth/sourceHeight` 或 `sourceSize` 控制；如果没有提供，默认按 `11264 x 11264` 地图尺寸解析。

## 运行机制

`LocalRouteNavigation` 内部持有：

- `RouteSession`：当前路线点、进度和运行状态。
- `RouteRunner`：按顺序执行路线点。
- `WaypointNavigator`：截图定位、方向推理、按键和鼠标转向。

执行时会先把路线点转换到当前 `MapLocator` 的地图尺寸，再从当前位置附近的路径点开始跑。每到达一个点，`RouteSession` 推进到下一个点，直到所有点完成。

## 注意事项

- 长路线拆成多个 segment 时，Pipeline 入口适合一次跑一个 segment；Python 内部连续跑多个 segment 时用 `LocalRouteNavigation` 类复用资源。
- `tolerance` 太小会导致角色在目标点附近反复调整；地图定位误差较大时应适当增大。
- `position_backend=coordinate` 依赖 Scapy 和系统抓包驱动。网络定位成功启用后，位置计算完全不再调用小地图视觉定位；网络样本暂时中断时位置状态为 `coordinate_stale`，不会回退视觉。方向角推理仍然使用截图。`auto` 模式仅在网络核心启动失败时使用地图定位。
- WebSocket 位置统一使用游戏原始坐标。网络定位发送 `x/y/z`；视觉定位通过标定逆变换发送 `x/y`，不发送无法可靠推导的高度 `z`。网络样本短暂中断时会保留最后一次位置用于地图展示，但该位置不会被导航逻辑视为有效实时位置。
- `config/navi_coordinate_calibration.json` 仅用于本地调试计算，不会在运行时读取。修改标定点后执行 `python scripts/update_navi_coordinate_transform.py`，脚本会计算最佳变换并更新 `coordinate_position.py` 中的常量。
- 需要手动标定时，可把该文件改为标定点格式。开启 `debug=true` 从 `Navi coordinate position/validation` 日志读取 `raw` 网络坐标，并在地图上取得对应的 11264×11264 像素坐标。至少填写 3 个相距较远且不共线的点：

```json
{
    "version": 1,
    "points": [
        {
            "raw": [12345.67, -23456.78, 100.0],
            "world": [51.909091, -36.0]
        },
        {
            "raw": [13345.67, -22456.78, 102.0],
            "map": [4310, 5980]
        },
        {
            "raw": [11345.67, -21456.78, 98.0],
            "map": [4050, 5870]
        }
    ]
}
```

每个点可填写 `map: [pixelX, pixelY]`，也可直接填写在线地图坐标 `world: [lat, lng]`。启动时会自动选择原始三维坐标中的运动平面并拟合旋转、缩放和平移。建议使用 4～8 个覆盖路线区域的点；日志中的 `error` 是拟合后的地图像素均方根误差。
- 坐标抓取核心不属于 Agent 源码。加密产物放在项目根目录 `thirdparty/`，并提供可导入的 `nte_coordinate_api` 模块。模块只需公开 `CoordinateCapture` 类及其 `start()`、`read(max_age=1.0)`、`close()` 方法；`read()` 返回原始 `(x, y, z)` 或 `None`。MXU 组装时会原样复制该目录。
- PyArmor 的 `pyarmor_runtime.pyd` 与 Python ABI 绑定，必须使用最终运行环境相同的 Python 主次版本和架构生成。当前发布环境目标为 Python 3.12 x64，不能使用 Python 3.10、3.13 或 3.14 生成的 runtime。
- `debug=true` 会打开 OpenCV 调试窗口，只用于本地调试。
- 路线执行依赖实时截图、地图定位和方向模型，不能在没有 Maa `Context` 和控制器的普通单元测试里完整运行。
