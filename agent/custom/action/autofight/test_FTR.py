# -*- coding: utf-8 -*-


import json
import time
from pathlib import Path
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from .logger import logger

class FT_StopFight(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')

        start_FT_time = json.loads(argv.custom_recognition_param).get("start_time" , -1)
        fight_spend_time = json.loads(argv.custom_recognition_param).get("fight_spend_time", 50)

        if start_FT_time == -1:
            current_time = time.time()
            current_node = argv.node_name
            context.override_pipeline({
                current_node: {
                    "custom_recognition_param": {
                        "start_time": str(current_time),
                        "fight_spend_time": int(fight_spend_time)

                    }}})
            logger.debug(f'开始时间:{current_time},运行{fight_spend_time}秒')
            return None
        elif time.time() > float(start_FT_time) + float(fight_spend_time):
            logger.debug(f'到达运行时间,已运行运行{time.time() - float(start_FT_time)}秒')
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")
        else:
            return None


class FT_GetTeamRole(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        ##########_##########_##########
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        current_image = argv.image.copy()
        ##########_##########_##########
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param.get("model")

        ##########_##########_##########

        pipeline_override_one = {
            "FT_GetTeamRole_NeuralNetworkDetect": {"recognition": "NeuralNetworkDetect",
                                              "model": model,
                                              "roi": [0, 0, 0, 0],
                                              "labels": ['attack', 'none'],
                                              "expected": [0]
                                              }
        }

        reco_detail = context.run_recognition(
            "FT_GetTeamRole_NeuralNetworkDetect",
            new_image,
            pipeline_override=pipeline_override_one,
        )
        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


class FT_ReadTxt(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        ##########_##########_##########
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        ##########_##########_##########
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        txt_dir = custom_recognition_param.get("txt_dir")
        ##########_##########_##########
        if not txt_dir:
            logger.error("txt_dir 参数为空")
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")

        txt_path = Path(txt_dir)
        if not txt_path.is_file() or txt_path.suffix != ".txt":
            logger.error(f"路径不是有效的txt文件: {txt_dir}")
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")

        content = txt_path.read_text(encoding="utf-8")
        logger.debug(f"成功读取txt文件: {txt_dir}")




        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")

