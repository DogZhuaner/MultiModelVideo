# main.py
import time
from cameraManager import camera
from capture_logic import check_and_capture_once
def start_handDetecton():
    try:
        while True:
            check_and_capture_once()
            time.sleep(0.05)  # 避免过度占用CPU
    except KeyboardInterrupt:
        print("退出")
    finally:
        camera.release()

if __name__ == '__main__':
    start_handDetecton()