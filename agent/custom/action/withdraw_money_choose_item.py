import re
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


def _screencap(controller):
    job = controller.post_screencap()
    job.wait()
    return controller.cached_image


def _filtered_boxes(result):
    """返回 filtered_results 中所有命中的 Rect 列表"""
    if result is None or not result.hit:
        return []
    boxes = []
    for r in result.filtered_results:
        if r.box is not None:
            boxes.append(r.box)
    return boxes


def _box_to_rect(box):
    if isinstance(box, (list, tuple)):
        return list(box)
    return [box.x, box.y, box.w, box.h]


def _click_rect(controller, rect):
    cx = rect[0] + rect[2] // 2
    cy = rect[1] + rect[3] // 2
    controller.post_touch_down(cx, cy).wait()
    time.sleep(0.05)  # 间隔太短概率失效
    controller.post_touch_up().wait()


def _parse_value(text):
    """从 OCR 文本中提取价格，支持 1,234、1.5K 等格式，返回 float 或 None"""
    if not text:
        return None
    text = text.strip().upper().replace(",", "").replace("，", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(K)?\s*/", text)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2) == "K":
        value *= 1000
    return value


@AgentServer.custom_action("withdraw_money_choose_item")
class WithdrawMoneyChooseItem(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        def run_recog(name):
            return context.run_recognition(name, _screencap(controller))

        def run_act(name):
            return context.run_action(name)

        def click_all_grey_backgrounds():
            """点击当前截图中的所有灰色角标"""
            result = run_recog("WithdrawMoneyGreyBackground")
            for box in _filtered_boxes(result):
                _click_rect(controller, _box_to_rect(box))
                time.sleep(0.5)

        def collect_product_values():
            """识别当前截图中所有商品 /h 价格，返回 [(value, rect), ...]"""
            items = []
            result = run_recog("WithdrawMoneyItemValue")
            for r in result.filtered_results if result else []:
                text = getattr(r, "text", "")
                value = _parse_value(text)
                if value is not None and r.box is not None:
                    items.append((value, _box_to_rect(r.box)))
            return items

        # Step 0: 先向上滑动一次，确保在列表顶部
        run_act("WithdrawMoneySwipeUp")
        time.sleep(1)

        # Step 1: 关掉灰色背景角标
        click_all_grey_backgrounds()

        # Step 2: 向下滑动
        run_act("WithdrawMoneySwipeDown")
        time.sleep(1)

        # Step 3: 再次关掉灰色背景角标
        click_all_grey_backgrounds()

        down_items = []
        seen = set()

        # Step 4: 向下位置匹配商品价格 /h
        for value, rect in collect_product_values():
            key = (rect[0], rect[1])
            if key not in seen:
                seen.add(key)
                down_items.append((value, rect))

        # Step 5: 向上滑动
        run_act("WithdrawMoneySwipeUp")
        time.sleep(1)

        up_items = []

        # Step 6: 向上位置匹配商品价格 /h
        for value, rect in collect_product_values():
            key = (rect[0], rect[1])
            if key not in seen:
                seen.add(key)
                up_items.append((value, rect))

        # Step 7: 按 value 去重，从大到小排序，只点前五个
        all_items = [(v, r, "down") for v, r in down_items] + [
            (v, r, "up") for v, r in up_items
        ]
        # 按 value 去重（保留第一个出现的 rect）
        deduped: dict[float, tuple] = {}
        for value, rect, swipe in all_items:
            if value not in deduped:
                deduped[value] = (value, rect, swipe)
        sorted_items = sorted(deduped.values(), key=lambda x: x[0], reverse=True)
        top5 = sorted_items[:5]

        current_swipe = "up"

        for value, rect, item_swipe in top5:
            if item_swipe != current_swipe:
                if item_swipe == "up":
                    run_act("WithdrawMoneySwipeUp")
                    current_swipe = "up"
                else:
                    run_act("WithdrawMoneySwipeDown")
                    current_swipe = "down"
                time.sleep(1)
            _click_rect(controller, rect)
            time.sleep(0.5)

        return CustomAction.RunResult(success=True)
