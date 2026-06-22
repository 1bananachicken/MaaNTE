# -*- coding: utf-8 -*-
# python -m pip install maafw
import os
import time
import sys

from maa.tasker import Tasker
from maa.toolkit import Toolkit
from maa.context import Context
from maa.resource import Resource
from maa.controller import AdbController, Win32Controller
from maa.define import MaaWin32InputMethodEnum, MaaWin32ScreencapMethodEnum
import json

Agent_FLAG = False

from maa.resource import Resource

resource = Resource()

# from agent_resource import resource

# resource = Resource()
# resource.use_cpu()

resource.use_directml()

from custom.autofight import *


def main():
    # from utils.logger import change_console_level
    change_console_level("DEBUG")

    user_path = "../"
    resource_path = r"E:\PyCharm\Project\GitMALostWord\MaaNTE\assets\resource\base"

    Toolkit.init_option(user_path)

    res_job = resource.post_bundle(resource_path)
    res_job.wait()

    # If not found on Windows, try running as administrator
    windows_list = Toolkit.find_desktop_windows()
    exists = any(window.window_name.strip() == '异环' for window in windows_list)
    if not exists:
        print("No ADB device found.")
        exit()

    def find_window_by_name(windows_list, target_name):
        for window in windows_list:
            if window.window_name.strip() == target_name:
                return window.hwnd
        return None

    pipeline_override = merge_json_files_recursive(resource_path + r'\pipeline')

    hwnd = find_window_by_name(windows_list, '异环')
    controller = Win32Controller(
        hWnd=hwnd,
        screencap_method=MaaWin32ScreencapMethodEnum.Background,
        mouse_method=MaaWin32InputMethodEnum.Seize,
        keyboard_method=MaaWin32InputMethodEnum.Seize,
    )

    controller.post_connection().wait()
    tasker = Tasker()
    # tasker = Tasker(notification_handler=MyNotificationHandler())
    tasker.bind(resource, controller)

    if not tasker.inited:
        print("Failed to init Windows.")
        exit()

    # just an example, use it in json

    start_time = time.time()

    # tasker.controller.post_click_key(27)
    # tasker.post_task("UF_GetImage", pipeline_override).wait().get()

    controller_one = tasker.controller


    controller2 = Win32Controller(
        hWnd=hwnd,
        screencap_method=MaaWin32ScreencapMethodEnum.Background,
        mouse_method=MaaWin32InputMethodEnum.PostMessage,
        keyboard_method=MaaWin32InputMethodEnum.PostMessage,
    )
    controller2.post_connection().wait()
    tasker2 = Tasker()
    # tasker = Tasker(notification_handler=MyNotificationHandler())
    tasker2.bind(resource, controller2)
    controller_two = tasker2.controller

    time.sleep(2)
    controller_one.post_key_down(18).wait()
    time.sleep(1)
    controller_one.post_key_down(87).wait()
    controller_one.post_key_down(65).wait()
    time.sleep(1)
    controller_one.post_key_up(87).wait()
    controller_one.post_key_up(65).wait()


    # tasker.post_action(
    #         "__CharacterControllerDeltaSwipeAction",
    #         override, "", override
    #     )

    # def run_action(
    #         self,
    #         entry: str,
    #         box: RectType = (0, 0, 0, 0),
    #         reco_detail: str = "",
    #         pipeline_override: Dict = {},

    end_time = time.time()  # 记录结束时间

    execution_time = end_time - start_time
    print(f"代码运行时间：{execution_time} 秒")


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img


"""
视角旋转控制模块
对应 charactercontroller 包中的 rotateView 及相关 Action
"""


class CharacterController:
    """角色视角控制器"""

    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    SCREEN_CENTER_X = SCREEN_WIDTH // 2  # 640
    SCREEN_CENTER_Y = SCREEN_HEIGHT // 2  # 360
    ROTATION_SPEED = 2  # 角度转换系数

    def __init__(self, maa_context):
        """
        初始化控制器
        :param maa_context: MaaFramework 上下文对象
        """
        self.ctx = maa_context
        self.target_not_found_counter = 0
        self.max_search_attempts = 15

    # ==================== 核心旋转函数 ====================

    def rotate_view(self, dx: int, dy: int):
        """
        旋转视角
        :param dx: 水平偏移量（正=右转，负=左转）
        :param dy: 垂直偏移量（正=下看，负=上看）
        """
        override = {
            "__CharacterControllerDeltaSwipeAction": {
                "begin": [self.SCREEN_CENTER_X, self.SCREEN_CENTER_Y, 4, 4],
                "end": [self.SCREEN_CENTER_X + dx, self.SCREEN_CENTER_Y + dy, 4, 4],
                "custom_action_param": {
                    "dx": dx,
                    "dy": dy
                }
            }
        }

        # 第一步：执行滑动
        self.ctx.run_action(
            "__CharacterControllerDeltaSwipeAction",
            [0, 0, 0, 0], "", override
        )
        # 第二步：按下 Alt 键
        self.ctx.run_action(
            "__CharacterControllerDeltaAltKeyDownAction",
            [0, 0, 0, 0], "", None
        )
        # 第三步：点击屏幕中心（Alt 按下状态旋转视角）
        self.ctx.run_action(
            "__CharacterControllerDeltaClickCenterAction",
            [0, 0, 0, 0], "", None
        )
        # 第四步：松开 Alt 键
        self.ctx.run_action(
            "__CharacterControllerDeltaAltKeyUpAction",
            [0, 0, 0, 0], "", None
        )

    # ==================== 水平旋转 ====================

    def yaw_delta(self, delta: int):
        """
        水平旋转视角（左右看）
        :param delta: 旋转角度（度）
        """
        delta = delta % 360
        dx = delta * self.ROTATION_SPEED
        self.rotate_view(dx, 0)

    # ==================== 垂直旋转 ====================

    def pitch_delta(self, delta: int):
        """
        垂直旋转视角（上下看）
        :param delta: 旋转角度（度）
        """
        delta = delta % 360
        dy = delta * self.ROTATION_SPEED
        self.rotate_view(0, dy)

    # ==================== 前后移动 ====================

    def move_axis(self, duration: int):
        """
        前后移动
        :param duration: 正=前进，负=后退，绝对值=持续时间
        """
        if duration > 0:
            override = {
                "__CharacterControllerAxisLongPressForwardAction": {
                    "duration": duration
                }
            }
            self.ctx.run_action(
                "__CharacterControllerAxisLongPressForwardAction",
                [0, 0, 0, 0], "", override
            )
        elif duration < 0:
            override = {
                "__CharacterControllerAxisLongPressBackwardAction": {
                    "duration": -duration
                }
            }
            self.ctx.run_action(
                "__CharacterControllerAxisLongPressBackwardAction",
                [0, 0, 0, 0], "", override
            )

    def forward_axis(self, axis: int):
        """
        按轴移动
        :param axis: 轴值，乘以 100 得到持续时间
        """
        self.move_axis(100 * axis)

    # ==================== 移动到目标 ====================

    def move_to_target(self, target_box: list, align_threshold: int = 120):
        """
        根据目标位置自动调整视角并移动
        :param target_box: 目标边界框 [x, y, w, h]
        :param align_threshold: 水平居中对齐阈值（像素）
        """
        lower_threshold = 480

        x, y, w, h = target_box
        target_center_x = x + w // 2
        target_center_y = y + h // 2
        offset_x = target_center_x - self.SCREEN_CENTER_X

        if offset_x < -align_threshold:
            # 目标在左边 → 左转
            dx = offset_x // 3
            self.rotate_view(dx, 0)
            print(f"[左转] offsetX={offset_x}, dx={dx}")

        elif offset_x > align_threshold:
            # 目标在右边 → 右转
            dx = offset_x // 3
            self.rotate_view(dx, 0)
            print(f"[右转] offsetX={offset_x}, dx={dx}")

        elif target_center_y > lower_threshold:
            # 目标在下方且已居中 → 已经走过，后退
            self.move_axis(-200)
            print(f"[后退] targetCenterY={target_center_y}")

        else:
            # 目标在上方且已居中 → 前进
            self.move_axis(200)
            print(f"[前进] offsetX={offset_x}, targetCenterY={target_center_y}")

    # ==================== 找不到目标时的搜索旋转 ====================

    def move_to_target_not_found(self, delta: int):
        """
        找不到目标时旋转视角搜索
        :param delta: 每次旋转的角度
        :return: 是否继续搜索
        """
        self.target_not_found_counter += 1

        if self.target_not_found_counter > self.max_search_attempts:
            print(f"[警告] 已旋转 {self.max_search_attempts} 次仍未找到目标，停止搜索")
            self.target_not_found_counter = 0
            return False

        delta = delta % 360
        dx = delta * self.ROTATION_SPEED
        self.rotate_view(dx, 0)

        print(f"[搜索] 第 {self.target_not_found_counter} 次旋转, delta={delta}, dx={dx}")
        return True

    def reset_search_counter(self):
        """重置搜索计数器"""
        self.target_not_found_counter = 0


# ==================== 使用示例 ====================

def example_usage(ctx):
    """使用示例"""
    controller = CharacterController(ctx)

    # 1. 水平旋转 90 度（右转）
    controller.yaw_delta(90)

    # 2. 垂直旋转 -30 度（上看）
    controller.pitch_delta(-30)

    # 3. 前进 2 秒
    controller.forward_axis(2)

    # 4. 根据目标位置自动导航
    target_box = [500, 300, 100, 100]  # 假设识别到的目标位置
    controller.move_to_target(target_box)

    # 5. 找不到目标时搜索
    found = False
    while not found:
        result = controller.move_to_target_not_found(45)  # 每次旋转 45 度
        if not result:
            print("目标丢失")
            break
        # 这里应该有目标检测逻辑来更新 found 状态


# ==================== 纯数学计算（不依赖框架） ====================

def calculate_rotation_params(delta_degrees: int, rotation_speed: int = 2):
    """
    计算旋转参数（纯数学，不依赖框架）
    :param delta_degrees: 旋转角度（度）
    :param rotation_speed: 旋转速度系数
    :return: (dx, dy) 偏移量
    """
    delta = delta_degrees % 360
    return delta * rotation_speed, 0  # 水平旋转


def calculate_click_position(start_box: list, end_box: list,
                             target: int, max_quantity: int):
    """
    计算滑块精确点击位置（对应 BetterSliding 的 handleFindEnd 逻辑）
    :param start_box: 起点边界框 [x, y, w, h]
    :param end_box: 终点边界框 [x, y, w, h]
    :param target: 目标数量
    :param max_quantity: 最大数量
    :return: (click_x, click_y) 点击坐标
    """
    start_x = start_box[0] + start_box[2] // 2
    start_y = start_box[1] + start_box[3] // 2
    end_x = end_box[0] + end_box[2] // 2
    end_y = end_box[1] + end_box[3] // 2

    numerator = target - 1
    denominator = max_quantity - 1

    if denominator == 0:
        raise ValueError("max_quantity 不能为 1")

    click_x = start_x + (end_x - start_x) * numerator // denominator
    click_y = start_y + (end_y - start_y) * numerator // denominator

    return click_x, click_y

def merge_json_files_recursive(folder_path):
    """
    递归读取文件夹及其所有子文件夹中的所有JSON文件，
    并将其内容合并为一个字典

    参数:
        folder_path: 根文件夹路径

    返回:
        合并后的字典
    """
    merged_dict = {}
    file_count = 0  # 统计处理的JSON文件数量

    # 递归遍历所有文件夹和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 检查文件是否为JSON文件
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                file_count += 1

                try:
                    # 打开并读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)

                        # 确保读取的数据是字典类型
                        if isinstance(json_data, dict):
                            # 将当前JSON文件的内容合并到总字典中
                            # 如果有重复的键，后面的会覆盖前面的
                            merged_dict.update(json_data)
                        else:
                            print(f"警告: {file_path} 中的数据不是字典类型，已跳过")

                except json.JSONDecodeError:
                    print(f"错误: 无法解析 {file_path}，文件可能不是有效的JSON")
                except Exception as e:
                    print(f"处理 {file_path} 时出错: {str(e)}")

    print(f"共处理了 {file_count} 个JSON文件")
    return merged_dict


if __name__ == "__main__":
    main()
