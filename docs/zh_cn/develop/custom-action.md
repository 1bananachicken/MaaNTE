# Custom 动作与识别

以下 Custom Action 位于 `agent/custom/action/Common/`，可在 Pipeline 中直接使用。

## click_override

自定义点击。通过 `custom_action_param` 指定目标 rect，或使用当前识别结果的 box。

- 注册名：`click_override`
- 参数 `custom_action_param`：`{ "target": [x, y, w, h] }`
- 若未提供 `custom_action_param`，则使用 `argv.box`（识别结果 box）

```jsonc
{
    "action": {
        "type": "Custom",
        "param": {
            "custom_action": "click_override",
            "custom_action_param": { "target": [100, 200, 50, 50] }
        }
    }
}
```

## alt_click

Alt + 点击。先按下 Alt 键，再点击识别结果 box 位置，最后松开 Alt。

- 注册名：`alt_click`
- 无需额外参数，点击位置由识别结果的 `box` 决定

```jsonc
{
    "recognition": { "type": "TemplateMatch", "param": { "template": "xxx.png" } },
    "action": {
        "type": "Custom",
        "param": { "custom_action": "alt_click" }
    }
}
```

## Common 工具函数

`agent/custom/action/Common/utils.py` 提供常用辅助函数：

| 函数 | 说明 |
|------|------|
| `get_image(controller)` | 截图，返回 numpy array |
| `click_rect(controller, rect, delay)` | 点击指定 rect 的中心 |
| `match_template_in_region(img, region, template, min_similarity, green_mask)` | 在区域内做模板匹配，返回 `(hit, score, x, y)` |

```python
from Common.utils import get_image, click_rect, match_template_in_region

img = get_image(controller)
hit, score, x, y = match_template_in_region(img, [0, 0, 1280, 720], template, 0.8)
if hit:
    click_rect(controller, [x, y, 50, 50])
```

## 编写新的 Custom Action

详细的 CustomAction 编写参考 `agent/custom/action/` 下的现有实现。核心原则：

- **流程控制由 Pipeline JSON 负责，Python 只处理难点**
- 所有坐标基于 **1280×720**
- 用户消息使用 `maafocus.PrintT()`，调试日志使用 `utils.logger`
- 长循环中检查 `context.tasker.stopping`
