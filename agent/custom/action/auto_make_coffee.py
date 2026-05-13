import json
import time
from pathlib import Path

import cv2

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from .Common.utils import click_rect, get_image, match_template_in_region


def _load_template(path: Path, name: str):
    if not path.exists():
        print(f"[coffee] 警告：模板不存在 {path.name}，{name} 相关步骤无法执行")
        return None

    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        print(f"[coffee] 警告：无法读取模板 {name}: {path}")
    return template


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

        self.start_template = _load_template(image_dir / "start.png", "start")
        self.star_template = _load_template(image_dir / "star.png", "star")
        self.claim_template = _load_template(image_dir / "claim.png", "claim")

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Auto Make Coffee Started: 敲人/驱赶顾客模式 ===")

        make_count = 10
        check_freq = 0.5

        if argv.custom_action_param:
            try:
                params = json.loads(argv.custom_action_param)
                make_count = int(params.get("count", make_count))
                check_freq = float(params.get("freq", check_freq))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"[coffee] 参数解析失败，使用默认参数: {exc}")

        start_template = self.start_template
        star_template = self.star_template
        claim_template = self.claim_template
        if start_template is None or star_template is None or claim_template is None:
            print("[coffee] 缺少旧版敲人模式必要模板，任务退出")
            return CustomAction.RunResult(success=False)

        return self._run_evict(
            context,
            context.tasker.controller,
            make_count,
            check_freq,
            start_template,
            star_template,
            claim_template,
        )

    def _run_evict(
        self,
        context: Context,
        controller,
        make_count: int,
        check_freq: float,
        start_template,
        star_template,
        claim_template,
    ):

        key_f = 70

        select_level_target = [18, 230, 188, 66]
        click_roi = [28, 272, 65, 56]
        start_roi = [1057, 648, 178, 44]
        star_roi = [1204, 109, 29, 27]
        exit_roi = [11, 12, 38, 37]
        claim_roi = [681, 539, 187, 38]

        for count in range(make_count):
            if context.tasker.stopping:
                return CustomAction.RunResult(success=False)

            print(f"=== Making Coffee {count + 1}/{make_count} ===")

            print("Tapping on select level...")
            click_rect(controller, select_level_target, count=3, hold=0.001)
            time.sleep(1)

            print("Waiting for start business button...")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                match_start, _, match_x, match_y = match_template_in_region(
                    img, start_roi, start_template, 0.8
                )
                if match_start:
                    print("Found 'start.png', clicking...")
                    click_rect(
                        controller,
                        [
                            match_x,
                            match_y,
                            start_template.shape[1],
                            start_template.shape[0],
                        ],
                        count=3,
                        hold=0.001,
                    )
                    time.sleep(3)
                    break

                time.sleep(check_freq)

            print("Waiting for star to reach sales goal...")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                click_rect(controller, click_roi, count=3, hold=0.001)
                img = get_image(controller)
                match_star, _, _, _ = match_template_in_region(img, star_roi, star_template, 0.9)
                if match_star:
                    print("Found 'star.png', clicking target...")
                    click_rect(controller, exit_roi, count=3, hold=0.001)
                    time.sleep(1)
                    break

                time.sleep(2)

            print("Waiting to claim reward...")
            while True:
                if context.tasker.stopping:
                    return CustomAction.RunResult(success=False)

                img = get_image(controller)
                match_claim, _, match_x, match_y = match_template_in_region(
                    img, claim_roi, claim_template, 0.8
                )
                if match_claim:
                    print("Found 'claim.png', clicking...")
                    click_rect(
                        controller,
                        [
                            match_x,
                            match_y,
                            claim_template.shape[1],
                            claim_template.shape[0],
                        ],
                        count=3,
                        hold=0.001,
                    )
                    time.sleep(1)
                    break

                time.sleep(check_freq)

            print("Round finished. Pressing 'F' to continue...")
            controller.post_key_down(key_f).wait()
            time.sleep(0.1)
            controller.post_key_up(key_f).wait()

            time.sleep(2)
            print("Current iteration finished.\n")

        print("All coffee tasks complete.")
        return CustomAction.RunResult(success=True)
