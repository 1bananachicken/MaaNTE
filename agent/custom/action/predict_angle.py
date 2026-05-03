import cv2
import time
import json
import math
import numpy as np

from pathlib import Path
from .Common.utils import get_image, match_template_in_region

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context


@AgentServer.custom_action("predict_angle")
class PredictAngle(CustomAction):
    abs_path = Path(__file__).parents[4]
    if Path.exists(abs_path / "assets"):
        image_dir = abs_path / "assets/resource/base/image/Fish"
    else:
        image_dir = abs_path / "resource/base/image/Fish"

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        pointer_roi = [86, 76, 35, 34] # x, y, w, h
        controller = context.tasker.controller
        
        # 提示：按 Q 键可以退出这个死循环
        print("启动实时角度预测... (在弹出的窗口上按 'Q' 键退出)")

        while True:
            # 1. 获取画面
            img = get_image(controller)
            if img is None:
                time.sleep(0.1)
                continue
            
            # --- 应用区域限制 ---
            rx, ry, rw, rh = pointer_roi
            img = img[ry:ry+rh, rx:rx+rw] 
            # --------------------------------------------------------

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([0, 0, 255])
            upper_hsv = np.array([37, 255, 255])

            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # === 安全校验：防止画面中没有符合HSV的目标导致代码报错崩溃 ===
            if not contours:
                print("当前画面未检测到箭头特征")
                cv2.imshow("Arrow Tracker", img)
                cv2.imshow("Mask Debug", mask)
                # 等待1秒，按 'q' 键退出
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
                continue
            # ========================================================

            arrow_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(arrow_contour)
            
            # 同样需要避免除以 0 的情况，改为 continue 跳过当前帧
            if M["m00"] == 0:
                continue
                
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 寻找距离重心最远的点（也就是箭头的尖端）
            max_dist = 0
            tipX, tipY = cX, cY
            for point in arrow_contour:
                px, py = point[0]
                dist = math.sqrt((px - cX)**2 + (py - cY)**2)
                if dist > max_dist:
                    max_dist = dist
                    tipX, tipY = px, py
                    
            # 计算 360 度角度
            delta_x = tipX - cX
            delta_y = tipY - cY
            
            angle_rad = math.atan2(delta_y, delta_x)
            angle_deg = math.degrees(angle_rad)
            
            if angle_deg < 0:
                angle_deg += 360
                
            final_angle = (angle_deg + 90) % 360
            
            print(f"箭头质心:({cX}, {cY}), 尖端:({tipX}, {tipY}) | 计算角度: {final_angle:.2f}°")

            # =========================================
            # 可视化绘制部分
            # =========================================
            display_img = img.copy()
            
            # 画出轮廓外边框 (绿色)
            cv2.drawContours(display_img, [arrow_contour], -1, (0, 255, 0), 1)
            
            # 画出质心 (蓝色，因为 OpenCV 是 BGR 格式，所以 (255, 0, 0) 是蓝色)
            cv2.circle(display_img, (cX, cY), 2, (255, 0, 0), -1)
            
            # 画出尖端 (红色，(0, 0, 255))
            cv2.circle(display_img, (tipX, tipY), 2, (0, 0, 255), -1)
            
            # 画出连线 (黄色)
            cv2.line(display_img, (cX, cY), (tipX, tipY), (0, 255, 255), 1)
            
            # 在旁边标上角度数字 (因为你画面可能很小，字体我调到了 0.4)
            cv2.putText(display_img, f"{final_angle:.1f}", (cX + 5, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 如果你画面太小(比如 24x24)，为了方便看，可以把它放大显示
            display_img = cv2.resize(display_img, (0,0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (0,0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

            # 显示窗口
            cv2.imshow("Arrow Tracker (Press 'q' to quit)", display_img)
            cv2.imshow("Mask Debug", mask) # 顺便看看你的 HSV 过滤得干不干净

            # 核心替换：替代原来的 time.sleep(1)
            # 等待 1000 毫秒(1秒)，如果在等待期间检测到按下了 'q' 键，则结束死循环
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                print("收到退出指令，关闭可视化监控")
                break

        # 循环结束后关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        
        # MAA 动作要求返回一个结果
        return CustomAction.RunResult(success=True)
