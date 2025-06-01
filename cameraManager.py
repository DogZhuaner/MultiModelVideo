# camera_manager.py
import cv2
import threading

class CameraManager:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def get_frame(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# 全局实例
camera = CameraManager()
