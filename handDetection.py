# detector.py
import mediapipe as mp
import cv2
from cameraManager import camera

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

def is_hand_present():
    ret, frame = camera.get_frame()
    if not ret:
        return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    return bool(result.multi_hand_landmarks)
