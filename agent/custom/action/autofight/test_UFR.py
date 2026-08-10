# python -m pip install maafw
import os
import time
import sys
import json
import cv2
import os
from datetime import datetime
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from .logger import logger


class UF_Logger(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        info_logger = json.loads(argv.custom_recognition_param).get("info")
        debug_logger = json.loads(argv.custom_recognition_param).get("debug")

        if info_logger:
            if isinstance(info_logger, str):
                info_logger = [info_logger]
            for info_one in info_logger:
                logger.info(f'{info_one}')
        if debug_logger:
            if isinstance(debug_logger, str):
                debug_logger = [debug_logger]
            for debug_one in debug_logger:
                logger.debug(f'{debug_one}')
        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img


def save_image(img, save_dir="screenshots"):
    """保存单张图片（自动递增）"""

    os.makedirs(save_dir, exist_ok=True)

    # 找到当前最大编号
    existing = [f for f in os.listdir(save_dir) if f.endswith('.png')]
    max_num = 0
    for f in existing:
        try:
            num = int(f.split('.')[0])
            if num > max_num:
                max_num = num
        except:
            pass

    # 新文件名为递增值
    new_num = max_num + 1
    filepath = os.path.join(save_dir, f"{new_num:06d}.png")
    print(f"save at :{filepath}")

    cv2.imwrite(filepath, img)
    return filepath


class UF_GetImage(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        # target_instance = json.loads(argv.custom_recognition_param).get("target_instance")
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        controller = context.tasker.controller
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_dir = os.path.join("debug/autofight_frames", f"{timestamp}")
        while True:
            if context.tasker.stopping:
                CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")
            img = get_image(controller)
            # print(datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
            save_image(img,save_dir )

            # time.sleep(0.5)  # 可选延迟

        img = get_image(controller)

        normal_dict = {
            "合订本": ["经验及甲硬币", "胡迪尼的魔术舞台", 1],
            "万花筒": ["经验及甲硬币", "胡迪尼的魔术舞台", 2],
            "硬币记": ["经验及甲硬币", "胡迪尼的魔术舞台", 3]
        }

        dict_dir = [[60, 100, 90, 30],
                    [60, 180, 90, 30],
                    [60, 260, 90, 30],
                    [60, 320, 90, 30],
                    [60, 400, 90, 30],
                    [60, 480, 90, 30],
                    ]

        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


class UF_CountClean(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        clean_node_name = json.loads(argv.custom_recognition_param).get("clean_node_name")
        target_number = json.loads(argv.custom_recognition_param).get("target_number")
        if not target_number or target_number == "set_in_code" or type(target_number) == str:
            logger.warning(f'{argv.node_name}节点无目标循环次数，设置为默认值5')
            target_number = 5
        if not clean_node_name or clean_node_name == "UF_Count":
            logger.warning(f'{argv.node_name}节点无清除对象，暂不进行清除操作')
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")

        context.override_pipeline({
            clean_node_name: {
                "custom_recognition_param": {
                    "current_number": 0,
                    "target_number": target_number

                }}})
        logger.debug(f'已将节点{clean_node_name}计数清零，循环次数设置为{target_number}')
        logger.debug(f'节点{argv.node_name}已完成运行')
        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


# @resource.custom_recognition("UF_Count")
class UF_Count(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        current_number = json.loads(argv.custom_recognition_param).get("current_number")
        target_number = json.loads(argv.custom_recognition_param).get("target_number")
        if not target_number or type(target_number) == str:
            logger.warning(f'{argv.node_name}节点无目标循环次数')
            return None
        target_number = int(target_number)
        if type(target_number) != int:
            logger.warning(f'{argv.node_name}节点无目标循环次数')
            return None
        if target_number < 0:
            logger.debug(f'运行次数为{str(current_number + 1)}，目标次数为无限')
            context.override_pipeline({
                argv.node_name: {
                    "custom_recognition_param": {
                        "current_number": current_number + 1,
                        "target_number": target_number
                    }}})
            return None
        if current_number < target_number:
            logger.debug(f'运行次数为{str(current_number + 1)}，目标次数为{target_number}')
            context.override_pipeline({
                argv.node_name: {
                    "custom_recognition_param": {
                        "current_number": current_number + 1,
                        "target_number": target_number
                    }}})
            return None
        else:
            logger.debug(f'已达到循环上限，循环终止，目标循环次数为{str(current_number)}')
            logger.debug(f'节点{argv.node_name}已完成运行')
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


# @resource.custom_recognition("UF_ChangePipeline")
class UF_ChangePipeline(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        target_pipeline = json.loads(argv.custom_recognition_param).get("target_pipeline")
        change_str = json.loads(argv.custom_recognition_param).get("change_str")
        context.override_pipeline({target_pipeline: change_str})
        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


