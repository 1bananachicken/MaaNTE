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
resource.use_directml()
from custom.autofight import *

key_codes = {
    'alt': 18,  # 左 Alt (VK_MENU)
    'lalt': 164,  # 左 Alt (VK_LMENU)
    'ralt': 165,  # 右 Alt (VK_RMENU)
    'key_w': 87,
    'key_a': 65,
    'key_s': 83,
    'key_d': 68,
    'key_q': 81,
    'key_e': 69,
    'key_r': 82,

}


def main():
    # from utils.logger import change_console_level
    change_console_level("DEBUG")

    user_path = "./"
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

    hwnd = find_window_by_name(windows_list, '异环')
    controller = Win32Controller(
        hWnd=hwnd,
        screencap_method=MaaWin32ScreencapMethodEnum.Background,
        mouse_method=MaaWin32InputMethodEnum.Seize,
        keyboard_method=MaaWin32InputMethodEnum.PostMessage,
    )
    controller.post_connection().wait()
    tasker = Tasker()
    # tasker = Tasker(notification_handler=MyNotificationHandler())
    tasker.bind(resource, controller)
    if not tasker.inited:
        print("Failed to init Windows.")
        exit()

    pipeline_override = merge_json_files_recursive(resource_path + r'\pipeline')
    start_time = time.time()

    # 实时截图
    # tasker.post_task("UF_GetImage", pipeline_override).wait().get()

    # 自动打本
    #
    # 进入副本
    normal_dict = [
        "合订本",
        "万花筒",
        "硬币记",

        "小心鸽子",
        "扑克茶会",
        "惊喜派对",
        "心电感应",
        "越狱艺术",

        "苹果核",
        "螺旋乐",
        "液态梦",
        "冷甜点",
        "戏剧芯",

        "钟表把戏",
        "雕塑展馆",
        "纬线织机",
        "守卫萝卜",
        "精神图谱",
        "轨道之夜",

    ]
    # # # 选择要战斗的目标
    # pipeline_override["AF_ChooseNormalInstance"]["recognition"]["param"][
    #     "custom_recognition_param"] = {"target_instance": "戏剧芯", "repeat_time": 3}
    # # 螺旋乐
    # # pipeline_override["AF_ChooseNormalInstance"]["recognition"]["param"][
    # #     "custom_recognition_param"] = {"target_instance": "硬币记", "repeat_time": 3}
    #
    # # # # 前往怪面前
    # pipeline_override["AF_Move_W"]["action"]["param"]["duration"] = 4000
    #
    # pipeline_override["AF_GetInInstance_3_1"]["next"] = ["AF_ChooseNormalInstance"]
    # pipeline_override["AF_GetInInstanceMove_10_1"]["next"] = ["AF_Move_W"]
    # pipeline_override["AF_Move_W"]["next"] = ["AF_AutoFightCls"]
    # pipeline_override["AF_AutoFightCls"]["next"] = ["AF_Action_FindBox"]
    # pipeline_override["AF_FindBox_3_1"]["next"] = ["AF_GetReward_1_1"]

    pipeline_override.update({
        "DF_TargetNumber_clean": {
            "recognition": {
                "param": {
                    "custom_recognition_param": {"clean_node_name": "DF_TargetNumber",
                                                 "target_number": 5}
                }
            }
        },
        "AF_ChooseNormalInstance": {
            "recognition": {
                "param": {
                    "custom_recognition_param": {
                        "target_instance": "纬线织机"
                    }
                }
            }
        }
    })

    # pipeline_override["DF_Action"]["recognition"]["param"][
    #     "custom_recognition_param"] = {"target_instance": "戏剧芯", "repeat_instance": 3}

    tasker.post_task("DF_Action", pipeline_override).wait().get()


    # tasker.post_task("AF_GetInInstanceMove_10_1", pipeline_override).wait().get()

    # tasker.post_task("AF_FindBox_3_1", pipeline_override).wait().get()

    # for i in range(3):
    #
    #     # # 进入副本界面
    #     tasker.post_task("AF_Action_GetInInstance", pipeline_override).wait().get()
    # # # 选择要战斗的目标
    # tasker.post_task("AF_ChooseNormalInstance", pipeline_override).wait().get()
    # # #
    # # # # 前往怪面前
    # tasker.post_task("AF_Move_W", pipeline_override).wait().get()
    #
    # # 刷取副本
    # tasker.post_task("AF_AutoFightCls", pipeline_override).wait().get()
    # # controller.post_click(360, 103, 2).wait()
    #
    # # 领取奖励
    # tasker.post_task("AF_Action_FindBox", pipeline_override).wait().get()

    # controller.post_touch_down(1026,300,0)
    # time.sleep(1)
    # controller.post_touch_up(0)
    # time.sleep(1)
    # controller.post_touch_down(360, 103, 0)
    # controller.post_touch_up(0)

    key_codes = {
        'alt': 18,  # 左 Alt (VK_MENU)
        'lalt': 164,  # 左 Alt (VK_LMENU)
        'ralt': 165,  # 右 Alt (VK_RMENU)
    }

    time.sleep(1)
    # tasker.post_task("AF_ChooseNormalInstance_1_2", pipeline_override).wait().get()
    # tasker.controller.post_click_key(27)

    end_time = time.time()  # 记录结束时间

    execution_time = end_time - start_time
    print(f"代码运行时间：{execution_time} 秒")


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img


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
