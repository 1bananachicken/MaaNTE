from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import Status
from datetime import datetime

VK = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "Space": 0x20,
    "F": 0x46,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "Esc": 0x1B,
}


def _is_hit(result) -> bool:
    """检查识别节点是否命中（状态 == 成功 0）"""
    if result is None:
        return False
    return result.status == 0  # 0 = MaaStatus.Success


class ActionHelper:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mx, self.my = 640, 360

    # ---------- 按键类操作 ----------
    def _call_key(self, node_type, key_str, extra=None):
        vk = VK.get(key_str)
        if vk is None:
            return False
        param = {"key": vk}
        if extra:
            param.update(extra)
        node_name = f"PinkPawHeist_{node_type}"
        override = {node_name: {"action": {"type": node_type, "param": param}}}
        return self.ctx.run_task(node_name, pipeline_override=override) is not None

    def click_key(self, key_str):
        return self._call_key("ClickKey", key_str)

    def key_down(self, key_str):
        return self._call_key("KeyDown", key_str)

    def key_up(self, key_str):
        return self._call_key("KeyUp", key_str)

    # ---------- 鼠标操作 ----------
    def move_to(self, x, y, duration_ms=None):
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
        if ret:
            self.mx, self.my = x, y
        return ret

    def click(self, x, y):
        self.move_to(x, y)
        override = {
            "PinkPawHeist_Click": {
                "action": {"type": "Click", "param": {"target": [x, y]}}
            }
        }
        return (
            self.ctx.run_task("PinkPawHeist_Click", pipeline_override=override)
            is not None
        )

    # ---------- 等待检测（铁门、撤离点） ----------
    def _check_until(self, node_name, timeout_ms):
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout_ms / 1000.0:
            if _is_hit(self.ctx.run_task(f"PinkPawHeist_{node_name}")):
                return True
            self.delay(200)
        return False

    def wait_gate(self, timeout=10000):
        return self._check_until("CheckGateOnce", timeout)

    def wait_evacuate(self, timeout=15000):
        return self._check_until("CheckEvacuateOnce", timeout)

    # ---------- 怪物检测与战斗 ----------

    def check_monster(self) -> bool:
        """检测当前帧是否有敌方（使用 run_recognition）"""
        # 获取当前截图
        image = self.ctx.tasker.controller.post_screencap().wait().get()
        # 运行识别节点（只做识别，不执行动作）
        result = self.ctx.run_recognition("PinkPawHeist_CheckMonsterOnce", image)
        # 判断是否命中（颜色区域存在）
        return result is not None and result.hit

    def wait_monster(self, timeout=5000) -> bool:
        """等待直到出现敌方，超时返回 False"""
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout / 1000.0:
            if self.check_monster():
                return True
            self.delay(100)
        return False

    def attack_cycle(self, times=3, loot=False):
        """执行一轮攻击（Space + 鼠标点击）"""
        for _ in range(times):
            self.ctx.run_task("PinkPawHeist_Core1_Attack_Space")
        if loot:
            self.click_key("F")

    def fight_until_no_monster(
        self,
        timeout_no_monster: int = 5000,
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
                self.click_key(role_to_switch_back)
                self.delay(200)
        return True

    def delay(self, ms):
        import time

        time.sleep(ms / 1000.0)


@AgentServer.custom_action("PinkPawHeistScheme1Action")
class PinkPawHeistScheme1Action(CustomAction):
    # 类变量：跨轮次保持的自适应超时（首轮测算后更新）
    _adaptive_timeout_ms: float = 120000  # 首轮默认最大等待 2 分钟
    _calibrated: bool = False

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        ah = ActionHelper(context)

        # ----- 切换3号，前往铁门 -----
        for _ in range(3):
            ah.click_key("3")
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
        ah.ctx.run_task("PinkPawHeist_Core1_Log_FightG1")
        for _ in range(3):
            ah.click_key("1")
            ah.delay(200)

        if not ah.fight_until_no_monster(
            timeout_no_monster=5000,
            wait_for_monster=True,
            role_to_switch_back="3",
            loot=False,
            attack_cycles=3,
        ):
            self._log_to_frontend(ah, "⚠️ 第一场战斗未检测到怪物，退出重试")
            self._exit_to_main(ah)
            return CustomAction.RunResult(success=True)

        # ----- 移动到房间右上角 -----
        ah.key_down("W")
        ah.delay(4000)
        ah.key_up("W")
        ah.delay(500)
        ah.key_down("S")
        ah.delay(600)
        ah.key_up("S")
        ah.delay(200)
        ah.key_down("D")
        ah.delay(3400)
        ah.key_up("D")
        ah.delay(200)
        ah.key_down("A")
        ah.delay(200)
        ah.key_up("A")
        ah.delay(200)
        # 后退
        ah.key_down("S")
        for _ in range(22):
            ah.click_key("F")
            ah.delay(300)
        ah.key_up("S")
        ah.delay(500)
        # 前进
        ah.key_down("A")
        ah.delay(700)
        ah.key_up("A")
        ah.delay(200)
        ah.key_down("W")
        for _ in range(13):
            ah.click_key("F")
            ah.delay(300)
        ah.key_up("W")
        ah.delay(500)
        # 继续向前
        ah.key_down("W")
        ah.delay(2500)
        ah.key_up("W")

        # 检测铁门
        """
        if ah.wait_gate():
            ah.click_key("Space")
            for _ in range(5):
                ah.click_key("F")
                ah.delay(300)
        else:
            self._exit_to_main(ah)
            return CustomAction.RunResult(success=False)
        """

        ah.delay(200)
        ah.click_key("Space")
        ah.delay(500)

        """
        # ----- 第二次战斗 -----
        for _ in range(3):
            ah.click_key("1")
            ah.delay(200)

        ah.fight_until_no_monster(
            timeout_no_monster=10000,
            wait_for_monster=True,
            role_to_switch_back="3",
            loot=True,
            attack_cycles=3,
        )
        """

        # 向左
        ah.key_down("S")
        ah.delay(1000)
        ah.key_up("S")
        ah.key_down("A")
        ah.delay(500)
        for _ in range(40):
            ah.click_key("F")
            ah.delay(100)
        ah.key_up("A")

        """
        # 第三次战斗
        for _ in range(3):
            ah.click_key("1")
            ah.delay(200)

        ah.fight_until_no_monster(
            timeout_no_monster=10000,
            wait_for_monster=True,
            role_to_switch_back="3",
            loot=True,
            attack_cycles=3,
        )
        """

        # 向后
        ah.key_down("S")
        ah.delay(500)
        ah.key_down("D")
        ah.delay(60)
        ah.key_up("D")
        for _ in range(12):
            ah.click_key("F")
            ah.delay(300)
        ah.key_up("S")
        ah.delay(500)
        ah.key_down("D")
        ah.delay(750)
        ah.key_up("D")

        # 前进
        ah.key_down("W")
        for _ in range(22):
            ah.click_key("F")
            ah.delay(300)
        ah.key_up("W")
        ah.delay(500)

        # 向右
        ah.key_down("D")
        for _ in range(16):
            ah.click_key("F")
            ah.delay(300)
        ah.key_up("D")
        ah.delay(500)

        # 打开铁门
        ah.click_key("F")
        ah.delay(300)
        ah.click_key("F")
        ah.delay(1000)
        """
        if not ah.wait_gate():
            self._exit_to_main(ah)
            return CustomAction.RunResult(success=False)
        """

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
        ah.delay(1500)
        ah.key_up("S")
        ah.delay(300)
        ah.key_down("D")
        ah.delay(2900)
        ah.key_up("D")
        ah.delay(300)
        ah.key_down("S")
        ah.delay(2000)
        ah.key_up("S")
        ah.delay(400)
        ah.key_down("W")
        ah.delay(2000)
        ah.key_up("W")

        # 切换4号战斗
        ah.click_key("4")
        ah.delay(300)
        ah.key_down("S")
        ah.delay(200)
        ah.key_up("S")
        ah.delay(2000)
        ah.ctx.run_task("PinkPawHeist_Core1_Log_FightG2")

        ah.fight_until_no_monster(
            timeout_no_monster=5000,
            wait_for_monster=True,
            role_to_switch_back="3",
            loot=True,
            attack_cycles=3,
        )

        # ---------- 战斗结束后移动至电梯与撤离点 ----------
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

        ah.key_down("D")
        for _ in range(15):
            ah.click_key("F")
            ah.delay(100)
        ah.key_up("D")
        ah.delay(300)

        ah.key_down("A")
        for _ in range(22):
            ah.click_key("F")
            ah.delay(100)
        ah.key_up("A")
        ah.delay(300)

        ah.key_down("D")
        ah.delay(1600)
        ah.key_up("D")
        ah.delay(200)

        ah.key_down("S")
        ah.delay(6000)
        ah.key_up("S")
        ah.delay(300)
        ah.key_down("A")
        ah.delay(3000)
        ah.key_up("A")
        ah.delay(300)

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
        ah.delay(1000)
        ah.key_down("W")
        ah.delay(2500)
        ah.key_up("W")
        ah.delay(100)
        ah.key_down("A")
        ah.delay(2300)
        ah.key_up("A")
        ah.delay(200)

        ah.key_down("W")
        for _ in range(19):
            ah.click_key("F")
            ah.delay(200)
        ah.key_up("W")
        ah.delay(300)
        ah.key_down("S")
        ah.delay(500)
        ah.key_up("S")

        # 最终撤离点检测
        ah.click_key("F")
        ah.delay(1500)

        evac_result = ah.ctx.run_task("PinkPawHeist_EvacuateOnce")
        if evac_result.status.succeeded:
            ah.delay(10000)
            # === 自适应等待：撤离成功后等待回到小吱旁边 ===
            self._wait_for_xiaozhi_adaptive(ah)
        else:
            # 撤离失败，退出到主界面，但返回 success=True 让 pipeline 继续循环重试
            self._log_to_frontend(ah, "⚠️ 撤离失败，退出到主界面等待重试")
            self._exit_to_main(ah)
        return CustomAction.RunResult(success=True)

    def _wait_for_xiaozhi_adaptive(self, ah: ActionHelper):
        """撤离成功后，等待角色回到小吱旁边。首轮测算耗时，后续自适应。"""
        import time

        timeout_ms = PinkPawHeistScheme1Action._adaptive_timeout_ms
        calibrated = PinkPawHeistScheme1Action._calibrated

        print(f"[PinkPawHeist] 等待回到小吱... (超时: {timeout_ms/1000:.1f}s, 已校准: {calibrated})")

        start = time.monotonic()
        found = False

        while time.monotonic() - start < timeout_ms / 1000.0:
            if ah.ctx.tasker.stopping:
                return
            # 尝试识别小吱
            image = ah.ctx.tasker.controller.post_screencap().wait().get()
            result = ah.ctx.run_recognition("PinkPawHeist_DetectXiaoZhi", image)
            if result is not None and result.hit:
                found = True
                break
            ah.delay(1000)  # 每秒检测一次

        elapsed_ms = (time.monotonic() - start) * 1000

        if found:
            print(f"[PinkPawHeist] ✅ 找到小吱! 耗时: {elapsed_ms/1000:.1f}s")
            if not calibrated:
                # 首轮校准：实际耗时 × 1.2 作为后续超时（至少20秒）
                new_timeout = max(20000, elapsed_ms * 1.2)
                PinkPawHeistScheme1Action._adaptive_timeout_ms = new_timeout
                PinkPawHeistScheme1Action._calibrated = True
                print(f"[PinkPawHeist] 📐 首轮校准完成!")
                print(f"[PinkPawHeist]    实测等待时间: {elapsed_ms/1000:.1f}s")
                print(f"[PinkPawHeist]    后续超时设为: {new_timeout/1000:.1f}s (实测×1.2)")
                # 推送到前端
                self._log_to_frontend(ah, f"📐 校准完成: 实测{elapsed_ms/1000:.1f}s, 后续超时{new_timeout/1000:.1f}s")
            else:
                self._log_to_frontend(ah, f"✅ 找到小吱 ({elapsed_ms/1000:.1f}s)")
        else:
            print(f"[PinkPawHeist] ⚠️ 等待超时 ({timeout_ms/1000:.1f}s)，未找到小吱")
            self._log_to_frontend(ah, f"⚠️ 等待小吱超时 ({timeout_ms/1000:.1f}s)")

    def _log_to_frontend(self, ah: ActionHelper, message: str):
        """通过 pipeline focus 向 MXU 前端推送消息"""
        ah.ctx.run_task("PinkPawHeist_LogMessage", pipeline_override={
            "PinkPawHeist_LogMessage": {
                "focus": {
                    "Node.Action.Starting": {
                        "content": message,
                        "display": ["log", "toast"]
                    }
                }
            }
        })

    def _exit_to_main(self, ah: ActionHelper):
        for _ in range(3):
            ah.click_key("Esc")
            ah.delay(1000)
        ah.delay(1500)
        ah.click(775, 473)
        ah.delay(500)
        ah.click(775, 473)
        ah.delay(10000)
