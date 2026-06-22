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
    # change_console_level("DEBUG")

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
        mouse_method=MaaWin32InputMethodEnum.PostMessage,
        keyboard_method=MaaWin32InputMethodEnum.PostMessage,
    )

    controller.post_connection().wait()

    tasker = Tasker()
    # tasker = Tasker(notification_handler=MyNotificationHandler())
    tasker.bind(resource, controller)

    if not tasker.inited:
        print("Failed to init Windows.")
        exit()

    # just an example, use it in json
    pipeline_override = merge_json_files_recursive(resource_path + r'\pipeline')
    start_time = time.time()

    tasker.post_task("UF_GetImage", pipeline_override).wait().get()



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
