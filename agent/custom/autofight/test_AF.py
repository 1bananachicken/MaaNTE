# -*- coding: utf-8 -*-


import json
import time
import numpy as np
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR
from .logger import logger


class AF_ChooseNormalInstance(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        custom_recognition_param = json.loads(argv.custom_recognition_param)

        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        target_instance = custom_recognition_param.get("target_instance")
        normal_dict = {
            "合订本": ["经验及甲硬币", 0],
            "万花筒": ["经验及甲硬币", 1],
            "硬币记": ["经验及甲硬币", 2],
            "小心鸽子": ["异能升级材料", 0],
            "扑克茶会": ["异能升级材料", 1],
            "惊喜派对": ["异能升级材料", 2],
            "心电感应": ["异能升级材料", 3],
            "越狱艺术": ["异能升级材料", 4],
            "苹果核": ["弧盘突破材料", 0],
            "螺旋乐": ["弧盘突破材料", 1],
            "液态梦": ["弧盘突破材料", 2],
            "冷甜点": ["弧盘突破材料", 3],
            "戏剧芯": ["弧盘突破材料", 4],
            "钟表把戏": ["空幕", 0],
            "雕塑展馆": ["空幕", 1],
            "纬线织机": ["空幕", 2],
            "守卫萝卜": ["空幕", 3],
            "精神图谱": ["空幕", 4],
            "轨道之夜": ["空幕", 5],

        }
        a = normal_dict.get(target_instance)
        if normal_dict.get(target_instance) is None:
            context.override_next("AF_ChooseNormalInstance",[])
            return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")
        instance_list = [[60, 120, 90, 30],
                    [60, 180, 90, 30],
                    [60, 280, 90, 30],
                    [60, 340, 90, 30],
                    [60, 420, 90, 30],
                    [60, 500, 90, 30],
                    ]

        target_instance_final = [normal_dict[target_instance][0]]+[instance_list[normal_dict[target_instance][1]]]

        ppover = {
            "AF_ChooseNormalInstance_1_1": {"expected": target_instance_final[0]},
            "AF_ChooseNormalInstance_1_2": {"expected": target_instance_final[0]},
            "AF_GetInInstanceMove_8_1" : {"expected": [target_instance]},
            "AF_GetInInstanceMove_8_2" : {"target": target_instance_final[1]},

                  }
        context.override_pipeline(ppover)
        # aaaa = context.get_node_data("AF_GetInInstanceMove_8_1")
        # aaaa1 = context.get_node_data("AF_GetInInstanceMove_9_1")
        # context.run_task("AF_GetInInstanceMove_8_1")

        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")



class AF_AutoFightCls(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param.get("model")
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')

        index = 0
        time_after_attack = 0.70
        finish_ocr_pipeline = JOCR()
        finish_ocr_pipeline.expected =["挑战成功"]
        finish_ocr_pipeline.roi =[567,140,140,40]
        finish_ocr_pipeline.threshold = 0.7
        finish_ocr_pipeline.only_rec = True
        t1 = time.time()

        pipeline_override_one ={
                    "MyCustomNeuralNetworkClassify": {"recognition": "NeuralNetworkClassify",
                                                      "model":model,
                                                      "roi": [0,0,0,0],
                                                      "labels":['attack', 'none'],
                                                      "expected":[0]
                                                      }
                }

        post_image =    None
        click_middle = True

        context.tasker.controller.post_key_down(83).wait()
        use_key_81 = True
        time_key_81 = None
        time_key_69 = None
        wait_time = 10
        while True:
            if context.tasker.stopping:
                return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")
            if click_middle:
                context.tasker.controller.post_click(360, 103, 2)
            new_image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                "MyCustomNeuralNetworkClassify",
                new_image,
                pipeline_override=pipeline_override_one ,
            )
            # print(reco_detail.all_results[0].label)
            if use_key_81:
                if time_key_81 is None:
                    context.tasker.controller.post_click_key(81).wait()
                    time_key_81 = time.time()
                elif (time.time() - time_key_81) >= wait_time:
                    context.tasker.controller.post_click_key(81).wait()
                    time_key_81 = time.time()  # 重置时间
            else:
                if time_key_69 is None:
                    context.tasker.controller.post_click_key(69).wait()
                    time_key_69 = time.time()
                elif (time.time() - time_key_69) >= wait_time:
                    context.tasker.controller.post_click_key(69).wait()
                    time_key_69 = time.time()  # 重置时间

            use_key_81 = not use_key_81  # 切换标志

            # context.tasker.controller.post_click_key(83).wait()



            # time.sleep(0.05)
            if reco_detail.all_results[0].label == "attack":
                print(f"attack_index = {index}")

                index += 1
                t1 = time.time()

                # context.tasker.controller.post_click(360, 103, 0)
                context.tasker.controller.post_click_key(16).wait()
                tt1 = time.time()
                # context.run_recognition_direct(
                #     "OCR",
                #     finish_ocr_pipeline,
                #     image=new_image,
                #                                )
                tt2 = time.time()
                if (tt2 -tt1)<=time_after_attack:
                    time.sleep(time_after_attack-tt2+tt1)
                context.tasker.controller.post_click(360, 103, 0).wait()
                time.sleep(0.1)
                continue

            context.tasker.controller.post_click(360, 103, 0).wait()

            finsh_index = context.run_recognition_direct(
                "OCR",
                finish_ocr_pipeline,
                image=new_image,
            )
            t2 = time.time()

            # print(t2-t1)
            # print(finsh_index.best_result)

            # if post_image is None:
            #     post_image = new_image
            #
            # _,mse = psnr_similarity(post_image,new_image)
            # # mse = np.mean((new_image.astype(float) - post_image.astype(float)) ** 2)
            # # if mse <=350:
            # #     click_middle = False
            #
            # print(mse)
            # post_image = new_image

            if finsh_index.best_result is not None:
                print(finsh_index.best_result.text)
                while  time.time()-time_key_69<=7 and time.time()-time_key_81<=7:
                    time.sleep(1)
                break
            if t2 - t1 > 40:
                break

        context.tasker.controller.post_key_up(83).wait()

        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


def psnr_similarity(img1, img2, threshold=30):
    """基于PSNR判断相似度，PSNR越大越相似（通常>30dB表示质量较好）"""
    if img1.shape != img2.shape:
        return False

    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    if mse == 0:
        return True, float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr > threshold, psnr

class AF_AutoFightClsTest(CustomRecognition):
    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param.get("model")
        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')

        index = 0
        time_after_attack = 0.30
        finish_ocr_pipeline = JOCR()
        finish_ocr_pipeline.expected =["挑战成功"]
        finish_ocr_pipeline.roi =[567,140,140,40]
        finish_ocr_pipeline.threshold = 0.5
        finish_ocr_pipeline.only_rec = True
        t1 = time.time()
        while True:

            tt1 = time.time()
            new_image = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                "MyCustomNeuralNetworkClassify",
                new_image,
                pipeline_override={
                    "MyCustomNeuralNetworkClassify": {"recognition": "NeuralNetworkClassify",
                                                      "model":model,
                                                      "roi": [0,0,0,0],
                                                      "labels":['attack', 'none'],
                                                      "expected":[0]
                                                      }
                },
            )
            print(reco_detail.all_results[0].label)
            # context.tasker.controller.post_click_key(81).wait()
            # context.tasker.controller.post_click_key(69).wait()
            # context.tasker.controller.post_click(360, 103, 0)
            # context.tasker.controller.post_click(360, 103, 2)
            time.sleep(0.05)
            if reco_detail.all_results[0].label == "attack":
                print(f"attack_index = {index}")
                index += 1
                tt2 = time.time()
                if (tt2 -tt1)<=time_after_attack:
                    time.sleep(time_after_attack-tt2+tt1)
                # context.tasker.controller.post_click(360, 103, 0)
                context.tasker.controller.post_click_key(16).wait()
                print("shift")

                context.run_recognition_direct(
                    "OCR",
                    finish_ocr_pipeline,
                    image=new_image,
                                               )
                t1 = time.time()


                time.sleep(0.5)

            finsh_index = context.run_recognition_direct(
                "OCR",
                finish_ocr_pipeline,
                image=new_image,
            )
            t2 = time.time()
            print(t2-t1)
            print(finsh_index.best_result)

            if finsh_index.best_result is not None:
                print(finsh_index.best_result.text)
                break
            if t2 - t1 > 20:
                break





        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")