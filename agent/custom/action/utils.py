import cv2
import time
import numpy as np


def get_image(controller):
    job = controller.post_screencap()
    job.wait()
    img = controller.cached_image
    return img

def click_rect(controller, rect):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    controller.post_touch_down(cx, cy).wait()
    time.sleep(0.001)
    controller.post_touch_up().wait()

def match_template_in_region(img, region, template, min_similarity=0.8):
    if img is None or not isinstance(img, np.ndarray):
        return False, 0.0, 0, 0
    
    x1, y1, w, h = region
    x2, y2 = x1 + w, y1 + h
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0, 0, 0
        
    roi = img[y1:y2, x1:x2]
    
    if len(roi.shape) == 3 and roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val >= min_similarity:
        return True, max_val, x1 + max_loc[0], y1 + max_loc[1]
    return False, max_val, 0, 0
