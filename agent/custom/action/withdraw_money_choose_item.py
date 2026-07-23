import time
import re
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from utils.logger import logger


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
    controller.post_touch_move(cx, cy).wait()  # 先移动再点击，防止误滑动
    controller.post_touch_down(cx, cy).wait()
    time.sleep(0.05)  # 间隔太短概率失效
    controller.post_touch_up().wait()
    time.sleep(0.05)


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

        def collect_product_values(recog_name):
            """识别当前截图中所有商品 /h 价格，返回 [(value, rect), ...]"""
            items = []
            result = run_recog(recog_name)
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
        time.sleep(1)

        # Step 4: 向下位置匹配商品价格 /h
        down_items = [
            (v, r, "down")
            for v, r in collect_product_values("WithdrawMoneyItemValueDown")
        ]

        # Step 5: 向上滑动
        run_act("WithdrawMoneySwipeUp")
        time.sleep(0.1)  # 等待动画

        # Step 6: 向上位置匹配商品价格 /h
        up_items = [
            (v, r, "up") for v, r in collect_product_values("WithdrawMoneyItemValueUp")
        ]

        # Step 7: 按 value 从大到小排序，只点前五个
        all_items = down_items + up_items
        sorted_items = sorted(all_items, key=lambda x: x[0], reverse=True)
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
            logger.debug(f"click {value} in {rect} of {item_swipe}")
            time.sleep(0.5)

        return CustomAction.RunResult(success=True)
