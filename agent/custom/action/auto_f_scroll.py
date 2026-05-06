import time
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import ctypes


@AgentServer.custom_action("auto_f_scroll")
class AutoFScroll(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        controller = context.tasker.controller

        KEY_F = 70  # 给 MAA 控制器发送按键用的 F 键代码
        VK_F = 0x46  # Windows 底层 API 监听物理键盘用的 F 键代码

        # 调用 Windows 底层 API 检测你手上的物理 F 键是否被按下
        # 0x8000 是一个掩码，如果结果不为 0，说明按键当前正被按住

        # Windows API 鼠标滚轮常数
        MOUSEEVENTF_WHEEL = 0x0800

        while not context.tasker.stopping:
            is_f_pressed = ctypes.windll.user32.GetAsyncKeyState(VK_F) & 0x8000
            if is_f_pressed:
                # 只有检测到你长按了 F 键，才会触发极速连点和滚轮
                controller.post_key_down(KEY_F)
                time.sleep(0.1)
                controller.post_key_up(KEY_F)

                try:
                    # 参数: dwFlags, dx, dy, dwData(滚轮幅度), dwExtraInfo
                    # -120 代表向下滚动一格
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
                except Exception as e:
                    pass

                time.sleep(0.1)

        return CustomAction.RunResult(success=True)
