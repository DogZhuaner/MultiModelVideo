# capture_logic.py
import time
import cv2

from MultiModelVideo.image import image_matcher
from cameraManager import camera
from handDetection import is_hand_present
from MultiModelVideo.image import image_matcher


last_no_hand_time = None
has_captured = False
capture_delay = 0.5

def check_and_capture_once():
    global last_no_hand_time, has_captured
    if is_hand_present():
        # 检测到手，重置计时和拍照标志
        last_no_hand_time = None
        has_captured = False
    else:
        if has_captured:
            return  # 已拍过照，不重复拍
        if last_no_hand_time is None:
            last_no_hand_time = time.time()
        elif time.time() - last_no_hand_time > capture_delay:
            ret, frame = camera.get_frame()
            if ret:
                cv2.imwrite(f"image/live_capture.jpg", frame)
                print("📸 拍照完成")
                has_captured = True  # 标记为已拍照
                #下一步送入image_matcher切分
                image_matcher.main()


