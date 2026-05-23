# -*- coding: utf-8 -*-

import json
import time
import numpy as np
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR
from .logger import logger




class DF_Action(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        target_instance = custom_recognition_param.get("target_instance")
        repeat_instance = custom_recognition_param.get("repeat_instance")
        print(1)
        # if target_instance is None:
        #     logger.error(f'未选择刷取目标')
        #     return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100),detail="Finish")
        # if repeat_instance is None:
        #     repeat_instance = 1
        ppover = {

            # "DF_TargetNumber_clean": {
            #     "recognition": {
            #         "param": {
            #             "custom_recognition_param": {"clean_node_name":"DF_TargetNumber","target_number":repeat_instance}
            #         }
            #     }
            # },


            "DF_Action_2_2": {
                "next": ["AF_Action_GetInInstance"]
            },


            # "AF_ChooseNormalInstance": {
            #     "recognition": {
            #         "param": {
            #             "custom_recognition_param": {"target_instance": target_instance}
            #         }
            #     }
            # },
            "AF_GetInInstance_3_1": {
                "next": ["AF_ChooseNormalInstance"]
            },
            "AF_GetInInstanceMove_10_1": {
                "next": ["AF_Move_W"]
            },
            "AF_Move_W": {
                "action": {
                    "param": {"duration": 5000}
                },
                "next": ["AF_AutoFightCls"]
            },
            "AF_AutoFightCls": {
                "next": ["AF_Action_FindBox"]
            },



        }

        a = context.override_pipeline(ppover)
        print()

        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")