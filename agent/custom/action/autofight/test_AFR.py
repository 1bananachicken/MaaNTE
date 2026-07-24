# -*- coding: utf-8 -*-


import json
import time
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR
from .logger import logger
from pathlib import Path

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
        ##########_##########_##########
        logger.debug("")
        logger.debug(f'正在运行节点{argv.node_name}')
        ##########_##########_##########
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param.get("model")

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
            if t2 - t1 > 10:
                break

        context.tasker.controller.post_key_up(83).wait()

        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")




class AF_AutoFightClsTest(CustomRecognition):
    # 互斥字典初始化模板, 包含常用按键及鼠标左/右/中键
    DEFAULT_MUTEX_DICT = {
        'q': 0,  # Q键上次使用时间戳, 键码: 81
        'e': 0,  # E键上次使用时间戳, 键码: 69
        'r': 0,  # R键上次使用时间戳, 键码: 82
        'f': 0,  # F键上次使用时间戳, 键码: 70
        'g': 0,  # G键上次使用时间戳, 键码: 71
        '1': 0,  # 数字1键上次使用时间戳, 键码: 49
        '2': 0,  # 数字2键上次使用时间戳, 键码: 50
        '3': 0,  # 数字3键上次使用时间戳, 键码: 51
        '4': 0,  # 数字4键上次使用时间戳, 键码: 52
        'shift': 0,  # Shift键上次使用时间戳, 键码: 16
        'space': 0,  # 空格键上次使用时间戳, 键码: 32
        'left_click': 0,  # 鼠标左键按下上次时间戳 (鼠标事件, 无键码)
        'right_click': 0,  # 鼠标右键按下上次时间戳 (鼠标事件, 无键码)
        'middle_click': 0,  # 鼠标中键按下上次时间戳 (鼠标事件, 无键码)
    }

    @staticmethod
    def _can_cast_skill(current_time, mutex_dict, skill_config):
        """检查技能是否可以释放, 并在通过判定时自动更新对应时间

        Args:
            current_time: 当前时间戳
            mutex_dict:   互斥字典, 包含所有操作的上次执行时间,
                          格式: {'q': timestamp, 'e': timestamp,
                                 'left_click': timestamp, 'r': timestamp, ...}
                          DEFAULT_MUTEX_DICT 提供了包含常用按键及鼠标左/右/中键的初始化模板
            skill_config: 当前执行内容字典, 包含需要互斥的值以及其互斥时间,
                          格式: {'name': str,   # 技能名称
                                 'cd': float,    # 技能自身CD, 也作为一个互斥时间(秒), 0=无冷却
                                 'mutex': {      # 互斥映射: 对每个关联技能指定独立的互斥间隔
                                     'e': 2.0,           # E技能放完后2秒内不能放当前技能
                                     'left_click': 0.5,  # 鼠标左键后0.5秒内不能放当前技能
                                     'r': 3.0,           # R技能放完后3秒内不能放当前技能
                                 }}                      # 无关联的技能在mutex中不出现, 自动跳过

        Returns:
            bool: 是否可以释放该技能 (返回True时已自动更新mutex_dict中对应技能的时间戳)
        """
        skill_name = skill_config['name']
        cd = skill_config.get('cd', 0)
        mutex_map = skill_config.get('mutex', {})

        # 自身CD检查
        last_cast_time = mutex_dict.get(skill_name, 0)
        if cd != 0 and current_time - last_cast_time < cd:
            return False


        # 逐技能互斥检查: 遍历mutex中所有关联的key, 从mutex_dict取对应上次运行时间, 校验间隔
        for target_skill, interval in mutex_map.items():
            if interval == 0:
                continue
            target_last_time = mutex_dict.get(target_skill, 0)
            if current_time - target_last_time < interval:
                return False

        # 通过所有检查, 更新对应技能的上次执行时间
        mutex_dict[skill_name] = current_time
        return True

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        logger.debug("##########_##########_##########")
        logger.debug(f'正在运行节点{argv.node_name}')
        ##########_##########_##########
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param.get("model",)
        e_interval = custom_recognition_param.get("e_interval", 20)  # 按下E键的间隔秒数, 0表示每轮都按
        q_interval = custom_recognition_param.get("q_interval", 20)  # 按下Q键的间隔秒数, 0表示每轮都按
        e_delay = custom_recognition_param.get("e_delay", 2)  # E键按下后延迟秒数(技能施法时间)
        q_delay = custom_recognition_param.get("q_delay", 4)  # Q键按下后延迟秒数(技能施法时间)
        skill_mutex_time = custom_recognition_param.get("skill_mutex_time", 2)  # 互斥时间: 放完一个技能后多少秒内不能放另一个
        skill_order = custom_recognition_param.get("skill_order", ["step_1", "step_2"])  # 技能尝试顺序, 默认先Q后E
        during_time = custom_recognition_param.get("during_time", -1 )  # 技能尝试顺序, 默认先Q后E

        node_start_time = time.time()


        # 技能动作字典: key为步骤标识, value为动作配置
        # 每个元素: {'config': 技能互斥配置, 'key_code': 按键键码, 'delay': 按下后等待秒数}
        skill_actions = {
            'step_1': {'config': {'name': 'q', 'cd': q_interval,
                                  'mutex': {'e': skill_mutex_time,
                                            # 'left_click': skill_mutex_time,
                                            'r': skill_mutex_time}},
                       'key_code': 81, 'delay': q_delay},
            'step_2': {'config': {'name': 'e', 'cd': e_interval,
                                  'mutex': {'q': skill_mutex_time,
                                            # 'left_click': skill_mutex_time,
                                            'r': skill_mutex_time}},
                       'key_code': 69, 'delay': e_delay},
        }

        index = 0
        time_after_attack = 0.30
        shift_attack_delay = 0.30  # Shift闪避后到鼠标左键攻击的固定等待秒数
        loop_min_interval = 0.1  # 每轮循环最小间隔(秒)
        mutex_dict = self.DEFAULT_MUTEX_DICT.copy()
        # finish_ocr_pipeline = JOCR()
        # finish_ocr_pipeline.expected = ["挑战成功"]
        # finish_ocr_pipeline.roi = [567, 140, 140, 40]
        # finish_ocr_pipeline.threshold = 0.5
        # finish_ocr_pipeline.only_rec = True
        t1 = time.time()
        jump_time = time.time()+ 4
        start_time = time.time()

        current_file = Path(__file__).resolve()

        project_root = current_file.parents[4]  # 索引从0开始，所以第5层是索引4
        model_path = project_root / "assets" / "resource" / "base" / "model" / "classify" / "0710_1.onnx"


        while not context.tasker.stopping:
            loop_start = time.time()
            t2 = time.time()
            # print(t2 - t1)


            if during_time >=0:
                if t2-start_time > during_time:
                    break


            if t2 - t1 > 30:
                break


            if model and model_path.is_file():
                print(model)
                print(print(__file__))
                new_image = context.tasker.controller.post_screencap().wait().get()
                reco_detail = context.run_recognition(
                    "MyCustomNeuralNetworkClassify",
                    new_image,
                    pipeline_override={
                        "MyCustomNeuralNetworkClassify": {"recognition": "NeuralNetworkClassify",
                                                          "model": model,
                                                          "roi": [0, 0, 0, 0],
                                                          "labels": ['attack', 'none'],
                                                          "expected": [0]
                                                          }
                    },
                )
                # print(reco_detail.all_results[0].label)
                current_time = time.time()
                reco_detail_label = reco_detail.all_results[0].label
            else:
                current_time = time.time()
                reco_detail_label = "none"

            click_right = True
            if reco_detail_label == "attack":
                # print(f"attack_index = {index}")
                index += 1
                tt2 = time.time()
                if (tt2 - current_time) <= time_after_attack:
                    time.sleep(time_after_attack - tt2 + current_time)
                # Shift闪避
                context.tasker.controller.post_click_key(16).wait()
                # Shift闪避后等待固定间隔, 再鼠标左键攻击
                time.sleep(shift_attack_delay)
                context.tasker.controller.post_click(360, 103, 0)
                t1 = time.time()
            else:
                # 按custom_recognition_param中指定的顺序依次尝试技能动作, 命中一个后break不再尝试后续

                if time.time() >= jump_time :


                    for step_key in skill_order:
                        action = skill_actions.get(step_key)
                        if action is None:
                            continue
                        if self._can_cast_skill(current_time, mutex_dict, action['config']):
                            context.tasker.controller.post_click_key(action['key_code'])
                            jump_time = time.time()+ action['delay']
                            # if action['delay'] > 0:
                            #     time.sleep(action['delay'])
                            logger.debug(f'按键 ： {action["config"]["name"]}')
                            click_right = False
                            break  # 命中一个后跳出, 不再尝试后续技能
                    if click_right:
                        pass
                        context.tasker.controller.post_click(360, 103, 0)
                        logger.debug(f'点击鼠标左键')
                else:
                    pass
                    context.tasker.controller.post_click(360, 103, 0)
                    logger.debug(f'点击鼠标左键')
            # print(mutex_dict)
            elapsed = time.time() - loop_start
            if elapsed < loop_min_interval:
                time.sleep(loop_min_interval - elapsed)
            continue

        logger.debug(f'节点运行时间: {time.time() - node_start_time}')
        logger.debug("##########_##########_##########")
        return CustomRecognition.AnalyzeResult(box=(0, 0, 100, 100), detail="Finish")


