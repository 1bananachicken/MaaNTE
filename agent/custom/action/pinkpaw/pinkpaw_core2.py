from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

try:
    from agent.custom.action.pinkpaw.pinkpaw_reward_logger import notify_pinkpaw_reward
except ImportError:
    from .pinkpaw_reward_logger import notify_pinkpaw_reward

VK = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "Space": 0x20,
    "E": 0x45,
    "F": 0x46,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "Esc": 0x1B,
}

REWARD_OCR_DELAY_MS = 3000
POST_REWARD_DELAY_MS = 7000


class StopActionException(Exception):
    pass


class TaskerStoppedException(Exception):
    pass


def _is_hit(result) -> bool:
    """检查识别节点是否命中（状态 == 成功 0）"""
    if result is None:
        return False
    if result.status.succeeded is False:
        return False
    return True


class ActionHelper:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mx, self.my = 640, 360
        self.last_check_time = 0  # 增加记录上次检测时间的变量
        self.fail_count = 0

    def is_stopping(self) -> bool:
        tasker = getattr(self.ctx, "tasker", None)
        if tasker is None:
            return False
        stopping = getattr(tasker, "stopping", False)
        if callable(stopping):
            stopping = stopping()
        return bool(stopping)

    def raise_if_stopped(self):
        if self.is_stopping():
            raise TaskerStoppedException(
                "PinkPawHeistScheme2Action stopped by Maa tasker."
            )

    def release_controls(self):
        controller = getattr(getattr(self.ctx, "tasker", None), "controller", None)
        if controller is None:
            return

        for key in ("W", "A", "S", "D", "E", "Space"):
            vk = VK.get(key)
            if vk is None:
                continue
            try:
                controller.post_key_up(vk).wait()
            except Exception as exc:
                print(f"[PinkPawHeist] failed to release {key}: {exc}")

        try:
            controller.post_key_up(2).wait()
        except Exception as exc:
            print(f"[PinkPawHeist] failed to release right mouse button: {exc}")

    def run_task(self, task_name, pipeline_override=None):
        self.raise_if_stopped()
        if pipeline_override is None:
            result = self.ctx.run_task(task_name)
        else:
            result = self.ctx.run_task(task_name, pipeline_override=pipeline_override)
        self.raise_if_stopped()
        return result

    # ---------- 按键类操作 ----------
    def _call_key(self, node_type, key_str, extra=None):
        if node_type != "KeyUp":
            self.raise_if_stopped()
        vk = VK.get(key_str)
        if vk is None:
            return False
        param = {"key": vk}
        if extra:
            param.update(extra)
        node_name = f"PinkPawHeist_{node_type}"
        override = {node_name: {"action": {"type": node_type, "param": param}}}
        ret = self.ctx.run_task(node_name, pipeline_override=override) is not None
        if node_type != "KeyUp":
            self.raise_if_stopped()
        return ret

    def click_key(self, key_str):
        return self._call_key("ClickKey", key_str)

    def key_down(self, key_str):
        return self._call_key("KeyDown", key_str)

    def key_up(self, key_str):
        return self._call_key("KeyUp", key_str)

    # ---------- 鼠标操作 ----------
    def move_to(self, x, y, duration_ms=None):
        self.raise_if_stopped()
        dx, dy = x - self.mx, y - self.my
        if dx * dx + dy * dy < 4:
            self.mx, self.my = x, y
            return True
        if duration_ms is None:
            duration_ms = max(int((dx**2 + dy**2) ** 0.5 / 0.5), 50)
        override = {
            "PinkPawHeist_MouseMove": {
                "action": {
                    "type": "Swipe",
                    "param": {
                        "begin": [self.mx, self.my],
                        "end": [x, y],
                        "duration": duration_ms,
                        "only_hover": True,
                    },
                }
            }
        }
        ret = self.ctx.run_task("PinkPawHeist_MouseMove", pipeline_override=override)
        self.raise_if_stopped()
        if ret:
            self.mx, self.my = x, y
        return ret

    def click(self, x, y):
        self.raise_if_stopped()
        self.move_to(x, y)
        override = {
            "PinkPawHeist_Click": {
                "action": {"type": "Click", "param": {"target": [x, y]}}
            }
        }
        ret = (
            self.ctx.run_task("PinkPawHeist_Click", pipeline_override=override)
            is not None
        )
        self.raise_if_stopped()
        return ret

    # ---------- 等待检测（铁门、撤离点） ----------
    def _check_until(self, node_name, timeout_ms):
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout_ms / 1000.0:
            self.raise_if_stopped()
            if _is_hit(self.ctx.run_task(f"PinkPawHeist_{node_name}")):
                return True
            self.delay(200)
        return False

    def wait_gate(self, timeout=10000):
        return self._check_until("CheckGateOnce", timeout)

    def wait_gate2(self, timeout=10000):
        return self._check_until("CheckGate2Once", timeout)

    def wait_door(self, timeout=10000):
        return self._check_until("CheckDoorOnce", timeout)

    def wait_evacuate(self, timeout=15000):
        return self._check_until("CheckEvacuateOnce", timeout)

    # ---------- 怪物检测与战斗 ----------

    def check_monster(self) -> bool:
        """检测当前帧是否有敌方（使用 run_recognition）"""
        self.raise_if_stopped()
        # 获取当前截图
        image = self.ctx.tasker.controller.post_screencap().wait().get()
        self.raise_if_stopped()
        # 运行识别节点（只做识别，不执行动作）
        result = self.ctx.run_recognition("PinkPawHeist_CheckMonsterOnce", image)
        self.raise_if_stopped()
        # 判断是否命中（颜色区域存在）
        return result is not None and result.hit

    def wait_monster(self, timeout=6000) -> bool:
        """等待直到出现敌方，超时返回 False"""
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout / 1000.0:
            self.raise_if_stopped()
            if self.check_monster():
                return True
            self.delay(200)
        return False

    def attack_cycle(self, times=3, loot=False):
        """执行一轮攻击（Space + 鼠标点击）"""
        for _ in range(times):
            self.raise_if_stopped()
            self.ctx.run_task("PinkPawHeist_Core1_Attack_Space")
            self.raise_if_stopped()
        if loot:
            self.click_key("F")

    def fight_until_no_monster(
        self,
        timeout_no_monster: int = 10000,
        wait_for_monster: bool = True,
        role_to_switch_back: str = None,
        loot: bool = False,
        attack_cycles: int = 3,
    ) -> bool:
        """打怪主循环，直到一段时间找不到怪退出"""
        import time

        if wait_for_monster:
            if not self.wait_monster(timeout=timeout_no_monster):
                return False

        no_monster_start = None
        while True:
            self.raise_if_stopped()
            if self.check_monster():
                no_monster_start = None
                self.attack_cycle(times=attack_cycles, loot=loot)
            else:
                now = time.monotonic()
                if no_monster_start is None:
                    no_monster_start = now
                elif now - no_monster_start >= timeout_no_monster / 1000.0:
                    break
                self.delay(50)

        if role_to_switch_back:
            for _ in range(3):
                self.raise_if_stopped()
                self.click_key(role_to_switch_back)
                self.delay(200)
        return True

    def delay(self, ms, check_reward=True):
        import time

        start = time.monotonic()
        end_time = start + (ms / 1000.0)

        while True:
            self.raise_if_stopped()
            now = time.monotonic()
            time_left = end_time - now

            # 1. 时间到了，立刻退出
            if time_left <= 0:
                break

            # 2. 剩余时间不足 0.4 秒时，直接一次性睡完
            if time_left <= 0.4:
                sleep_time = min(0.05, time_left)
                if sleep_time <= 0:
                    break
                time.sleep(sleep_time)
                continue

            # 3. 只有当 check_reward 为 True 时，才进行检测
            if check_reward and (now - self.last_check_time > 2.0):
                self.last_check_time = now

                override = {"PinkPawHeist_CheckReward": {"timeout": 100}}
                result = self.ctx.run_task(
                    "PinkPawHeist_CheckReward", pipeline_override=override
                )
                self.raise_if_stopped()

                # 判断是否命中
                if result is None or result.status.succeeded is False:
                    self.fail_count += 1  # 没找到，失败次数 +1
                    print(
                        f"警告：未检测到 CheckReward，当前连续失败次数: {self.fail_count}"
                    )

                    # 连续失败达到 2 次，才抛出异常终止
                    if self.fail_count >= 2:
                        raise StopActionException(
                            "PinkPawHeist_CheckReward 连续 2 次检测失败，终止主流程"
                        )
                else:
                    self.fail_count = 0  # 找到了，立刻把失败次数清零！

                # 重新计算剩余时间
                now = time.monotonic()
                time_left = end_time - now
                if time_left <= 0:
                    break

            # 4. 睡 50 毫秒
            sleep_time = min(0.05, time_left)
            if sleep_time > 0:
                time.sleep(sleep_time)


@AgentServer.custom_action("PinkPawHeistScheme2Action")
class PinkPawHeistScheme2Action(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        ah = ActionHelper(context)
        try:
            current_ctrl = ah.ctx.tasker.controller
            for _ in range(3):
                ah.click_key("1")
                ah.delay(200)
            ah.key_down("W")
            ah.delay(4500)
            ah.key_down("D")
            ah.delay(3400)
            ah.key_up("D")
            ah.delay(2000)
            ah.key_up("W")
            ah.click_key("F")
            ah.delay(4000)

            ah.key_down("W")
            ah.delay(1500)
            ah.key_down("D")
            ah.delay(300)
            ah.key_up("D")
            for _ in range(20):
                ah.click_key("Space")
                ah.delay(200)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("S")
            ah.delay(700)
            ah.key_up("S")

            # ----- 第一场战斗（怪堆） -----
            ah.run_task("PinkPawHeist_Core1_Log_FightG1")
            for _ in range(3):
                ah.click_key("1")
                ah.delay(200)
            ah.click_key("E")
            ah.delay(200)
            ah.click_key("E")
            ah.delay(200)
            ah.click_key("E")

            if not ah.fight_until_no_monster(
                timeout_no_monster=10000,
                wait_for_monster=True,
                role_to_switch_back="3",
                loot=False,
                attack_cycles=3,
            ):
                self._exit_to_main(ah)
                return CustomAction.RunResult(success=True)

            # 打开铁门
            ah.key_down("D")
            ah.delay(2000)
            ah.key_up("D")
            ah.delay(100)
            ah.key_down("A")
            ah.delay(1100)
            ah.key_up("A")
            ah.delay(100)
            ah.key_down("W")
            ah.delay(2000)
            ah.key_up("W")
            ah.delay(200)

            ah.click_key("F")
            ah.delay(300)
            ah.click_key("F")
            ah.delay(300)
            ah.click_key("F")
            ah.delay(1000)

            # 穿过铁门区域
            ah.delay(3000)
            ah.key_down("W")
            ah.delay(2300)
            ah.key_down("A")
            ah.delay(2000)
            ah.key_up("A")
            ah.delay(1500)
            ah.key_up("W")
            ah.delay(300)
            ah.key_down("A")
            ah.delay(5000)
            ah.key_up("A")
            ah.delay(300)
            ah.key_down("S")
            ah.delay(1800)
            ah.key_up("S")
            ah.delay(300)
            ah.key_down("D")
            ah.delay(2900)
            ah.key_up("D")
            ah.delay(300)
            ah.key_down("S")
            ah.delay(2300)
            ah.key_up("S")
            ah.delay(400)
            ah.key_down("W")
            ah.delay(2000)
            ah.key_up("W")

            # 切换早雾战斗
            ah.click_key("3")
            ah.delay(300)
            ah.key_down("S")
            ah.delay(200)
            ah.key_up("S")
            ah.delay(2000)
            ah.key_down("E")
            ah.delay(1800)
            ah.key_up("E")
            ah.delay(200)
            ah.run_task("PinkPawHeist_Core1_Log_FightG2")

            ah.fight_until_no_monster(
                timeout_no_monster=10000,
                wait_for_monster=True,
                role_to_switch_back="3",
                loot=True,
                attack_cycles=3,
            )

            # ---------- 战斗结束后移动至电梯 ----------
            ah.key_down("W")
            ah.delay(3000)
            ah.key_up("W")
            ah.delay(300)

            ah.key_down("D")
            ah.delay(2000)
            ah.key_down("S")
            ah.delay(3000)
            ah.key_up("S")
            ah.delay(300)
            ah.key_up("D")

            ah.delay(300)
            ah.key_down("A")
            ah.delay(1300)
            ah.key_up("A")
            ah.delay(300)

            ah.key_down("S")
            for _ in range(7):
                ah.click_key("F")
                ah.delay(100)
            ah.key_up("S")
            ah.delay(300)

            ah.key_down("S")
            ah.delay(6000)
            ah.key_up("S")
            ah.delay(300)
            ah.key_down("A")
            ah.delay(100)
            ah.key_down("S")
            ah.delay(2800)
            ah.key_up("S")
            ah.delay(100)
            ah.key_up("A")
            ah.delay(300)

            if not ah.wait_door():
                self._exit_to_main(ah)
                return CustomAction.RunResult(success=True)

            ah.click_key("F")
            ah.delay(1000)
            ah.key_down("D")
            ah.delay(300)
            ah.key_up("D")
            ah.delay(200)
            ah.key_down("S")
            ah.delay(1500)
            ah.key_up("S")

            ah.click_key("F")
            ah.delay(1000, check_reward=False)
            ah.key_down("W")
            ah.delay(14500)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("D")
            ah.delay(13000)
            ah.key_up("D")
            ah.delay(200)

            # ---------- 移动至G1办公层与电梯 ----------
            ah.key_down("D")
            ah.delay(4900)
            ah.key_up("D")
            ah.delay(100)
            ah.key_down("W")
            ah.delay(3000)
            ah.key_up("W")
            # 躲第一道激光
            ah.delay(1400)
            ah.key_down("W")
            ah.delay(2500)
            ah.key_up("W")
            # 躲第二道激光
            ah.delay(300)
            ah.key_down("W")
            ah.delay(10000)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("W")
            ah.delay(16200)
            ah.key_up("W")
            ah.delay(100)

            # ---------- 移动至G1激光层 ----------
            ah.key_down("W")
            ah.delay(7300)
            # 开始躲激光
            ah.key_up("W")
            ah.delay(200)
            ah.key_down("D")
            ah.delay(2500)
            ah.key_down("W")
            for _ in range(4):
                ah.click_key("F")
                ah.delay(200)
            ah.key_up("W")
            ah.delay(200)
            ah.key_up("D")
            ah.delay(200)
            ah.key_down("W")
            ah.delay(3300)
            ah.key_up("W")
            ah.delay(200)
            ah.key_down("D")
            ah.delay(3000)
            ah.key_up("D")
            ah.delay(400)
            ah.key_down("A")
            ah.delay(400)
            ah.key_up("A")
            ah.delay(500)
            ah.key_down("W")
            ah.delay(500)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(200)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(1450)
            ah.key_up("W")
            ah.delay(200)
            ah.key_down("A")
            ah.delay(500)
            ah.key_up("A")
            ah.delay(200)
            ah.key_down("W")
            ah.delay(2000)
            ah.key_up("W")
            ah.delay(200)
            ah.key_down("D")
            ah.delay(1500)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(200)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(2500)
            ah.key_up("D")
            ah.delay(200)
            ah.key_down("W")
            ah.delay(580)
            ah.key_up("W")
            ah.delay(200)
            # 出激光房
            ah.key_down("D")
            ah.delay(3000)
            ah.key_up("D")
            ah.delay(200)
            ah.key_down("W")
            ah.delay(10000)
            ah.key_up("W")
            # 换狼偷渡
            for _ in range(3):
                ah.click_key("4")
                ah.delay(200)

            ah.key_down("W")
            ah.delay(1500)
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(7000)
            current_ctrl.post_key_up(2).wait()
            ah.key_up("W")
            for _ in range(3):
                ah.click_key("3")
                ah.delay(200)
            ah.key_down("D")
            ah.delay(300)
            ah.key_up("D")
            ah.delay(200)
            if not ah.wait_gate2():
                self._exit_to_main(ah)
                return CustomAction.RunResult(success=True)
            ah.click_key("F")
            ah.delay(9000)
            ah.key_down("W")
            ah.delay(100)
            ah.key_down("A")
            ah.delay(1500)
            ah.key_up("A")
            ah.delay(100)
            ah.key_up("W")
            ah.delay(200)
            ah.click_key("F")
            ah.delay(1000, check_reward=False)

            # ---------- 移动至藏品层 ----------
            ah.key_down("W")
            ah.delay(7000)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("A")
            ah.delay(6120)
            ah.key_up("A")
            ah.delay(100)
            # 偷左边展柜藏品
            ah.key_down("S")
            ah.delay(1000)
            ah.key_up("S")
            ah.delay(100)
            ah.key_down("W")
            ah.delay(1000)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("A")
            ah.delay(2500)
            ah.key_up("A")
            ah.delay(100)

            ah.key_down("S")
            ah.delay(1000)
            ah.key_up("S")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("W")
            ah.delay(1000)
            ah.key_up("W")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(1400)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("D")
            ah.delay(1000)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(1000)
            ah.key_up("W")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("S")
            ah.delay(1000)
            ah.key_up("S")
            ah.delay(100)

            # 偷左前边展柜藏品
            ah.key_down("D")
            ah.delay(1500)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(6500)
            ah.key_up("W")
            ah.delay(100)

            ah.key_down("A")
            ah.delay(600)
            ah.key_up("A")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("D")
            ah.delay(600)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            for _ in range(12):
                ah.click_key("F")
                ah.delay(200)
            ah.key_up("W")
            ah.delay(100)

            ah.key_down("A")
            ah.delay(600)
            ah.key_up("A")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("D")
            ah.delay(600)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(500)
            ah.key_up("W")
            ah.delay(100)
            # 准备穿激光
            ah.key_down("D")
            ah.delay(2600)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(500)
            ah.key_up("W")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("S")
            ah.delay(500)
            ah.key_up("S")
            ah.delay(100)

            ah.key_down("D")
            ah.delay(1100)
            ah.key_up("D")
            ah.delay(1700)
            # 穿过第一道激光
            ah.key_down("D")
            ah.delay(500)
            ah.key_up("D")
            ah.delay(1000)
            # 穿过第二道激光
            ah.key_down("D")
            ah.delay(600)
            ah.key_up("D")
            ah.delay(1000)

            # 穿过第一竖激光
            ah.key_down("D")
            ah.delay(1400)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(500)
            ah.key_up("W")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("S")
            ah.delay(500)
            ah.key_up("S")
            ah.delay(1800)

            # 穿过第二道竖激光和第三道和第四道激光
            ah.key_down("D")
            ah.delay(4000)
            ah.key_up("D")
            ah.delay(100)

            ah.key_down("W")
            ah.delay(500)
            ah.key_up("W")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("S")
            ah.delay(500)
            ah.key_up("S")
            ah.delay(100)

            ah.key_down("D")
            ah.delay(2000)
            ah.key_up("D")
            ah.delay(100)
            ah.key_down("S")
            ah.delay(500)
            ah.key_up("S")
            ah.delay(100)

            # 开始吃右前边展柜藏品

            ah.key_down("D")
            ah.delay(600)
            ah.key_up("D")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("A")
            ah.delay(700)
            ah.key_up("A")
            ah.delay(100)

            ah.key_down("S")
            for _ in range(12):
                ah.click_key("F")
                ah.delay(200)
            ah.key_up("S")
            ah.delay(100)

            ah.key_down("D")
            ah.delay(600)
            ah.key_up("D")
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("A")
            ah.delay(600)
            ah.key_up("A")
            ah.delay(100)

            ah.key_down("S")
            ah.delay(9000)
            ah.key_up("S")
            # 开始吃右边展柜藏品
            ah.delay(100)
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("W")
            ah.delay(600)
            ah.key_up("W")
            ah.delay(100)

            ah.key_down("D")
            ah.delay(1300)
            ah.key_up("D")
            ah.delay(100)
            ah.key_down("W")
            ah.delay(2500)
            ah.key_up("W")
            ah.delay(100)
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)
            ah.key_down("S")
            ah.delay(800)
            ah.key_up("S")
            ah.delay(8500)

            ah.key_down("A")
            ah.delay(7500)
            ah.key_up("A")
            # 开始上楼
            ah.key_down("W")
            ah.delay(10000)
            ah.key_up("W")

            ah.key_down("D")
            ah.delay(5900)
            ah.key_up("D")
            ah.delay(100)
            # 开始吃二楼右边
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("S")
            ah.delay(1900)
            ah.key_up("S")
            for _ in range(10):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("S")
            ah.delay(2750)
            ah.key_up("S")
            ah.delay(100)

            ah.key_down("D")
            ah.delay(1000)
            ah.key_up("D")
            # 进入激光藏品房门口
            for _ in range(3):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("D")
            ah.delay(50)
            ah.key_down("S")
            ah.delay(150)
            ah.key_up("S")
            ah.delay(250)
            ah.key_up("D")
            ah.delay(100)
            # 吃激光藏品房门口下面的藏品
            ah.key_down("S")
            ah.delay(1000)
            ah.click_key("Space")
            ah.delay(400)
            ah.key_up("S")
            for _ in range(4):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("W")
            ah.delay(1000)
            ah.click_key("Space")
            ah.delay(400)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("D")
            ah.delay(400)
            ah.key_up("D")
            ah.delay(100)
            # 吃激光藏品房门口上面的藏品
            ah.key_down("W")
            ah.delay(1000)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(100)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(400)
            ah.key_up("W")
            for _ in range(8):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("S")
            ah.delay(1000)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(100)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(500)
            ah.key_up("S")
            ah.delay(100)

            # 出激光藏品房门
            ah.key_down("A")
            ah.delay(1600)
            ah.key_up("A")
            ah.delay(100)

            ah.key_down("S")
            ah.delay(2500)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(200)
            current_ctrl.post_key_up(2).wait()
            ah.delay(500)
            ah.key_up("S")
            ah.delay(100)
            for _ in range(4):
                ah.click_key("F")
                ah.delay(200)

            ah.key_down("W")
            for _ in range(6):
                ah.click_key("F")
                ah.delay(200)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(200)
            current_ctrl.post_key_up(2).wait()
            ah.delay(1000)
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(200)
            current_ctrl.post_key_up(2).wait()
            ah.delay(3400)
            ah.key_up("W")
            ah.delay(100)
            ah.key_down("S")
            ah.delay(200)
            ah.key_up("S")
            ah.delay(100)
            ah.key_down("A")
            ah.delay(500)
            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(200)
            current_ctrl.post_key_up(2).wait()
            ah.delay(8500)
            ah.key_up("A")

            # 开始吃二楼左边
            ah.key_down("S")
            for _ in range(28):
                ah.click_key("F")
                ah.delay(200)

            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_down(2).wait()  # 按下右键
            ah.delay(200)
            current_ctrl.post_key_up(2).wait()
            for _ in range(2):
                ah.click_key("F")
                ah.delay(200)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(200)
            current_ctrl.post_key_down(32).wait()
            ah.delay(200)
            current_ctrl.post_key_up(32).wait()
            ah.delay(200)
            ah.key_up("S")

            # ---------- 最后撤离1 ----------

            ah.key_down("S")
            ah.delay(3000)
            ah.key_up("S")
            ah.delay(200)
            ah.key_down("W")
            ah.delay(6500)
            ah.key_up("W")
            ah.delay(500)
            ah.key_down("A")
            ah.delay(150)
            ah.key_up("A")
            ah.delay(500)
            for _ in range(4):
                ah.click_key("F")
                ah.delay(200)
            ah.delay(1500, check_reward=False)
            evac_result = ah.run_task("PinkPawHeist_EvacuateOnce")
            if evac_result.status.succeeded:
                ah.delay(REWARD_OCR_DELAY_MS, check_reward=False)
                notify_pinkpaw_reward(ah.ctx, success=True)
                ah.delay(POST_REWARD_DELAY_MS, check_reward=False)
            else:
                # ---------- 最后撤离2 ----------

                ah.key_down("D")
                ah.delay(6100)
                ah.key_up("D")

                ah.key_down("W")
                ah.delay(7700)
                ah.key_up("W")

                ah.key_down("D")
                ah.delay(3000)
                ah.key_up("D")

                ah.key_down("W")
                ah.delay(5800)
                ah.key_up("W")

                ah.key_down("A")
                ah.delay(2800)
                ah.key_up("A")

                ah.key_down("W")
                ah.delay(9000)
                ah.key_up("W")

                for _ in range(4):
                    ah.click_key("F")
                    ah.delay(200)
                ah.delay(7000)
                ah.key_down("W")
                ah.delay(2800)
                ah.key_up("W")

                ah.key_down("D")
                ah.delay(3000)
                ah.key_up("D")

                for _ in range(4):
                    ah.click_key("F")
                    ah.delay(200)
                ah.delay(1500, check_reward=False)
                evac_result = ah.run_task("PinkPawHeist_EvacuateOnce")
                if evac_result.status.succeeded:
                    ah.delay(REWARD_OCR_DELAY_MS, check_reward=False)
                    notify_pinkpaw_reward(ah.ctx, success=True)
                    ah.delay(POST_REWARD_DELAY_MS, check_reward=False)
                else:
                    # ---------- 最后撤离3 ----------
                    ah.delay(500)
                    ah.key_down("A")
                    ah.delay(3000)
                    ah.key_up("A")

                    ah.key_down("S")
                    ah.delay(10000)
                    ah.key_up("S")
                    ah.delay(100)

                    ah.key_down("D")
                    ah.delay(2000)
                    ah.key_up("D")
                    ah.key_down("S")
                    ah.delay(3100)
                    ah.key_up("S")

                    ah.key_down("D")
                    ah.delay(14000)
                    ah.key_up("D")

                    ah.key_down("S")
                    ah.delay(1000)
                    ah.key_up("S")
                    ah.delay(500)
                    ah.key_down("W")
                    ah.delay(400)
                    ah.key_up("W")
                    ah.delay(500)
                    for _ in range(4):
                        ah.click_key("F")
                        ah.delay(200)

                    ah.key_down("D")
                    ah.delay(5400)
                    ah.key_up("D")

                    for _ in range(4):
                        ah.click_key("F")
                        ah.delay(200)

                    ah.delay(1500, check_reward=False)
                    evac_result = ah.run_task("PinkPawHeist_EvacuateOnce")
                    if evac_result.status.succeeded:
                        ah.delay(REWARD_OCR_DELAY_MS, check_reward=False)
                        notify_pinkpaw_reward(ah.ctx, success=True)
                        ah.delay(POST_REWARD_DELAY_MS, check_reward=False)
                    else:
                        notify_pinkpaw_reward(ah.ctx, success=False)
                        self._exit_to_main(ah)
                        return CustomAction.RunResult(success=True)
                    return CustomAction.RunResult(success=True)
                return CustomAction.RunResult(success=True)
            return CustomAction.RunResult(success=True)
        except TaskerStoppedException as e:
            print(f"[PinkPawHeist] stopped by tasker: {e}")
            ah.release_controls()
            return CustomAction.RunResult(success=False)
        except StopActionException as e:
            # 捕获到终止异常，直接结束
            print(f"[PinkPawHeist] 流程提前终止: {e}")
            # --- 安全垫：强制松开所有方向键 ---
            ah.key_up("W")
            ah.delay(50)
            ah.key_up("A")
            ah.delay(50)
            ah.key_up("S")
            ah.delay(50)
            ah.key_up("D")
            ah.delay(50)

            current_ctrl = ah.ctx.tasker.controller
            current_ctrl.post_key_up(2).wait()  # 松开鼠标右键
            ah.delay(50)
            current_ctrl.post_key_up(32).wait()  # 松开空格键
            for _ in range(4):
                ah.click_key("Esc")
                ah.delay(2000, check_reward=False)
            evac_result = ah.run_task("PinkPawHeist_Once")
            ah.delay(10000, check_reward=False)
            notify_pinkpaw_reward(ah.ctx, success=False)

            return CustomAction.RunResult(success=True)

    def _exit_to_main(self, ah: ActionHelper):
        for _ in range(3):
            ah.click_key("Esc")
            ah.delay(1000, check_reward=False)
        ah.delay(1500, check_reward=False)
        ah.click(775, 473)
        ah.delay(500, check_reward=False)
        ah.click(775, 473)
        ah.delay(10000, check_reward=False)
