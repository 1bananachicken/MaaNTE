# 地图传送

通过游戏内置的地图索引系统自动传送到指定传送点。

## Pipeline 调用

注册名：`map_teleport_to_point`

```jsonc
{
    "MapTeleport": {
        "action": "Custom",
        "custom_action": "map_teleport_to_point",
        "custom_action_param": {
            "teleport_point_id": "fountain"
        }
    }
}
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `teleport_point_id` | string | 必填 | 传送点 ID，对应 JSON 中 `id` 字段。也支持别名 `teleport_id`。 |
| `teleport_points_file` | string | `map_teleport/teleport_points.json` | 传送点数据文件路径，相对于 `assets/resource/base/`。 |
| `template_threshold` | float | `0.8` | 模板匹配置信度阈值。 |
| `ocr_threshold` | float | `0.5` | OCR 文本匹配阈值。 |
| `max_area_switches` | int | `15` | 地区切换最大尝试次数。 |
| `action_delay` | float | `0.5` | 各步骤间延迟（秒）。 |

## 传送点 JSON 格式

文件位于 `assets/resource/base/map_teleport/teleport_points.json`：

```jsonc
{
    "version": 1,
    "teleport_points": [
        {
            "id": "fountain",
            "name": "喷泉传送点",
            "areaName": "绘空",
            "iconIndex": 2,
            "selectionName": "推荐地点",
            "iconPath": "image/map_teleport/teleport_icon/phone_booth.png",
            "description": "地图索引中使用的传送点"
        }
    ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，对应 `teleport_point_id`。 |
| `name` | string | 传送点显示名称，用于用户提示。 |
| `areaName` | string | 地图索引中地区名称，通过 OCR 匹配。 |
| `iconIndex` | int | 目标传送图标在列表中的序号（从 1 开始）。 |
| `selectionName` | string | 主选项名称，通过 OCR 匹配并点击。 |
| `iconPath` | string | 传送点图标模板图片路径，相对于 `assets/resource/base/`。 |
| `description` | string | 传送点描述文字（可选）。 |
