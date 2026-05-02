import time
import json
from pathlib import Path
import cv2

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from .utils import get_image, click_rect, match_template_in_region

# ── 时间驱动的食物点击序列 ───────────────────────────────────────────────────
# delay_from_start: 点击开始营业后经过多少秒执行本组点击
# 依据：总时 2:00，food_2 剩余 1:51（经过 9s），food_3 剩余 1:44（经过 16s）
FOOD_SEQUENCES = [
    {
        "delay_from_start": 0.5,    # food_1：立即开始
        "clicks": [[496, 673], [408, 536], [264, 428]],
    },
    {
        "delay_from_start": 9.0,    # food_2：经过 9 秒
        "clicks": [[663, 668], [856, 685], [697, 445]],
    },
    {
        "delay_from_start": 16.0,   # food_3：经过 16 秒
        "clicks": [[113, 659], [135, 526], [155, 431]],
    },
]

# 每个点击之间等待游戏响应的秒数
CLICK_DELAY = 1.75

# 时间条的搜索区域（模板匹配 timer.png）
TIMER_ROI = [0, 0, 1280, 720]

# 开始营业按钮的搜索区域（模板匹配 start.png）
START_ROI = [900, 600, 380, 120]   # 扩大搜索范围提高鲁棒性

# 退出/结算按钮的搜索区域（模板匹配 exit.png）
EXIT_BUTTON_ROI = [0, 0, 1280, 720]

# 领取奖励按钮的搜索区域（模板匹配 claim.png）
CLAIM_ROI = [681, 539, 187, 38]
# ─────────────────────────────────────────────────────────────────────────────


def _load_template(path: Path, name: str):
    if not path.exists():
        print(f"[coffee] 警告：模板不存在 {path.name}，相关步骤将跳过")
        return None
    tmpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if tmpl is None:
        print(f"[coffee] 警告：无法读取模板 {name}")
    return tmpl


def _wait_for(controller, template, roi, label, check_freq=0.2, stopping_fn=None):
    """轮询直到模板出现，只检测不点击，返回 False 表示被 stopping 打断。"""
    if template is None:
        print(f"[coffee] 跳过等待 {label}（无模板）")
        return True
    while True:
        if stopping_fn and stopping_fn():
            return False
        img = get_image(controller)
        matched, _, _, _ = match_template_in_region(img, roi, template)
        if matched:
            print(f"[coffee] 检测到 {label}")
            return True
        time.sleep(check_freq)


def _wait_and_click(controller, template, roi, label, check_freq=0.5, stopping_fn=None):
    """轮询直到模板匹配成功后点击中心，返回 False 表示被 stopping 打断。"""
    if template is None:
        print(f"[coffee] 跳过等待 {label}（无模板）")
        return True
    while True:
        if stopping_fn and stopping_fn():
            return False
        img = get_image(controller)
        matched, _, mx, my = match_template_in_region(img, roi, template)
        if matched:
            cx = mx + template.shape[1] // 2
            cy = my + template.shape[0] // 2
            print(f"[coffee] 检测到 {label}，点击 ({cx}, {cy})")
            click_rect(controller, [mx, my, template.shape[1], template.shape[0]])
            return True
        time.sleep(check_freq)


@AgentServer.custom_action("auto_make_coffee")
class AutoMakeCoffee(CustomAction):
    def __init__(self):
        super().__init__()
        abs_path = Path(__file__).parents[3]
        image_dir = (
            abs_path / "assets/resource/base/image/auto_make_coffee"
            if (abs_path / "assets").exists()
            else abs_path / "resource/base/image/auto_make_coffee"
        )

        self.enter_template  = _load_template(image_dir / "enter.png",       "enter")
        self.level_template = _load_template(image_dir / "level_chose.png", "level_chose")
        self.start_template  = _load_template(image_dir / "start.png",       "start")
        self.timer_template  = _load_template(image_dir / "timer.png",       "timer")
        self.exit_template   = _load_template(image_dir / "exit.png",        "exit")
        self.claim_template  = _load_template(image_dir / "claim.png",       "claim")

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Auto Make Coffee Started ===")
        controller = context.tasker.controller
        make_count = 10

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                make_count = int(params.get("count", make_count))
            except Exception:
                pass

        stopping = lambda: context.tasker.stopping

        full_screen = [0, 0, 1280, 720]

        for round_idx in range(make_count):
            if stopping():
                return CustomAction.RunResult(success=False)
            print(f"\n=== 第 {round_idx + 1}/{make_count} 轮 ===")

            # ── 步骤1：检测到店长特供入口后按 F 进入 ─────────────────────────
            print("[coffee] 等待店长特供入口（enter）...")
            if not _wait_for(controller, self.enter_template, full_screen,
                             "enter", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)
            controller.post_key_down(70).wait()   # F
            controller.post_key_up(70).wait()
            print("[coffee] 已按 F 进入店长特供")
            time.sleep(1)

            # ── 步骤2：等待关卡选择界面出现并点击 1-1 新品练习 ───────────────
            print("[coffee] 等待关卡选择界面...")
            if not _wait_and_click(controller, self.level_template, full_screen,
                                   "level_chose", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)
            time.sleep(1)

            # ── 步骤2：点击开始营业 ───────────────────────────────────────
            print("[coffee] 等待开始营业按钮...")
            if not _wait_and_click(controller, self.start_template, START_ROI,
                                   "start", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)

            # 等待时间条出现后再开始计时
            print("[coffee] 等待时间条出现...")
            if not _wait_for(controller, self.timer_template, TIMER_ROI,
                             "timer", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)
            t_start = time.time()

            # ── 步骤3：按时间依次服务 3 个顾客 ───────────────────────────
            for i, food in enumerate(FOOD_SEQUENCES):
                if stopping():
                    return CustomAction.RunResult(success=False)

                wait = food["delay_from_start"] - (time.time() - t_start)
                if wait > 0:
                    print(f"[coffee] food_{i+1}：等待 {wait:.1f}s 后开始")
                    time.sleep(wait)

                print(f"[coffee] food_{i+1}：执行点击序列")
                for pos in food["clicks"]:
                    click_rect(controller, [pos[0], pos[1], 1, 1])
                    time.sleep(CLICK_DELAY)

            # ── 步骤4：点击退出按钮 ───────────────────────────────────────
            print("[coffee] 等待退出按钮...")
            if not _wait_and_click(controller, self.exit_template, EXIT_BUTTON_ROI,
                                   "exit", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)
            time.sleep(1)

            # ── 步骤5：领取奖励 ───────────────────────────────────────────
            print("[coffee] 等待领取奖励按钮...")
            if not _wait_and_click(controller, self.claim_template, CLAIM_ROI,
                                   "claim", stopping_fn=stopping):
                return CustomAction.RunResult(success=False)
            time.sleep(1)

            print(f"[coffee] 第 {round_idx + 1} 轮完成")

        print("=== Auto Make Coffee: 全部轮次完成 ===")
        return CustomAction.RunResult(success=True)
