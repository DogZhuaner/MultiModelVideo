# hand_capture_system.py - 集成手掌检测拍照系统
import time
import cv2
import os
import threading
from datetime import datetime
from cameraManager import camera

# 尝试导入 mediapipe，如果失败则使用备用检测方法
try:
    import mediapipe as mp

    HAS_MEDIAPIPE = True
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    print("✅ 使用 MediaPipe 手掌检测")
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ MediaPipe 未安装，使用简化检测方法")


class HandCaptureSystem:
    """集成手掌检测拍照系统"""

    def __init__(self, save_dir="image", capture_delay=0.5):
        # 保存配置
        self.save_dir = save_dir
        self.capture_delay = capture_delay
        os.makedirs(save_dir, exist_ok=True)

        # 状态变量
        self.last_no_hand_time = None
        self.has_captured = False
        self.running = False
        self.detection_thread = None

        # 回调函数
        self.on_hand_detected_callback = None
        self.on_hand_disappeared_callback = None
        self.on_photo_captured_callback = None

        # 简化检测参数（当没有MediaPipe时使用）
        if not HAS_MEDIAPIPE:
            self._init_simple_detection()

        print(f"✅ 手掌检测拍照系统初始化完成")
        print(f"📁 保存目录: {os.path.abspath(save_dir)}")
        print(f"⏱️ 捕获延迟: {capture_delay}秒")

    def _init_simple_detection(self):
        """初始化简化检测方法"""
        import numpy as np
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.min_hand_area = 2000

    def is_hand_present(self):
        """检测是否有手掌存在"""
        ret, frame = camera.get_frame()
        if not ret or frame is None:
            return False

        if HAS_MEDIAPIPE:
            return self._mediapipe_detection(frame)
        else:
            return self._simple_detection(frame)

    def _mediapipe_detection(self, frame):
        """使用MediaPipe检测手掌"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            return bool(result.multi_hand_landmarks)
        except Exception as e:
            print(f"MediaPipe检测异常: {e}")
            return False

    def _simple_detection(self, frame):
        """简化的手掌检测方法"""
        try:
            import numpy as np

            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 肤色检测
            mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 检查是否有足够大的轮廓
            for contour in contours:
                if cv2.contourArea(contour) > self.min_hand_area:
                    return True

            return False

        except Exception as e:
            print(f"简化检测异常: {e}")
            return False

    def check_and_capture_once(self):
        """检查手掌状态并执行拍照逻辑"""
        hand_present = self.is_hand_present()

        if hand_present:
            # 检测到手，重置计时和拍照标志
            if self.last_no_hand_time is not None:
                # 手掌重新出现
                if self.on_hand_detected_callback:
                    self.on_hand_detected_callback()

            self.last_no_hand_time = None
            self.has_captured = False

        else:
            # 没有检测到手
            if self.has_captured:
                return  # 已拍过照，不重复拍

            if self.last_no_hand_time is None:
                # 手掌刚消失
                self.last_no_hand_time = time.time()
                print("👋 手掌消失，开始倒计时...")

                if self.on_hand_disappeared_callback:
                    self.on_hand_disappeared_callback()

            elif time.time() - self.last_no_hand_time > self.capture_delay:
                # 时间到，执行拍照
                self._capture_photo()

    def _capture_photo(self):
        """执行拍照"""
        ret, frame = camera.get_frame()
        print("📸 准备拍照...")

        if ret and frame is not None:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_capture_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, filename)

            # 保存图片
            success = cv2.imwrite(filepath, frame)

            if success:
                print(f"📸 拍照完成: {filename}")
                self.has_captured = True

                # 执行回调
                if self.on_photo_captured_callback:
                    self.on_photo_captured_callback(filepath, frame)

                return filepath
            else:
                print("❌ 图片保存失败")
                return None
        else:
            print("❌ 无法获取摄像头画面")
            return None

    def start_detection(self):
        """启动检测（非阻塞）"""
        if self.running:
            print("⚠️ 检测已在运行中")
            return

        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("🚀 手掌检测已启动（后台运行）")

    def stop_detection(self):
        """停止检测"""
        if not self.running:
            print("⚠️ 检测未在运行")
            return

        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1)
        print("🛑 手掌检测已停止")

    def _detection_loop(self):
        """检测循环（在后台线程中运行）"""
        try:
            while self.running:
                self.check_and_capture_once()
                time.sleep(0.05)  # 避免过度占用CPU
        except Exception as e:
            print(f"检测循环异常: {e}")
        finally:
            print("检测循环结束")

    def start_detection_blocking(self):
        """启动检测（阻塞模式）"""
        print("🚀 开始手掌检测...")
        print("📝 使用说明:")
        print("   - 将手掌放入摄像头画面")
        print("   - 移开手掌后会自动拍照")
        print("   - 按 Ctrl+C 退出")
        print("-" * 40)

        try:
            self.running = True
            while self.running:
                self.check_and_capture_once()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n👋 用户退出")
        except Exception as e:
            print(f"\n❌ 程序异常: {e}")
        finally:
            self.running = False
            print("检测结束")

    def set_callbacks(self, on_hand_detected=None, on_hand_disappeared=None, on_photo_captured=None):
        """设置回调函数"""
        self.on_hand_detected_callback = on_hand_detected
        self.on_hand_disappeared_callback = on_hand_disappeared
        self.on_photo_captured_callback = on_photo_captured

    def manual_capture(self):
        """手动拍照"""
        return self._capture_photo()

    def reset_state(self):
        """重置检测状态"""
        self.last_no_hand_time = None
        self.has_captured = False
        print("🔄 检测状态已重置")

    def get_status(self):
        """获取当前状态"""
        hand_present = self.is_hand_present()
        return {
            'running': self.running,
            'hand_present': hand_present,
            'has_captured': self.has_captured,
            'waiting_time': time.time() - self.last_no_hand_time if self.last_no_hand_time else 0
        }

    def release(self):
        """释放资源"""
        self.stop_detection()
        # 注意：这里不释放camera，因为可能有其他程序在使用


# ===== 全局实例和便捷接口 =====
_global_detector = None


def get_detector(save_dir="image", capture_delay=0.5):
    """获取全局检测器实例"""
    global _global_detector
    if _global_detector is None:
        _global_detector = HandCaptureSystem(save_dir, capture_delay)
    return _global_detector


def start_hand_detection(save_dir="image", capture_delay=0.5, blocking=True):
    """启动手掌检测（便捷接口）"""
    detector = get_detector(save_dir, capture_delay)
    if blocking:
        detector.start_detection_blocking()
    else:
        detector.start_detection()
    return detector


def stop_hand_detection():
    """停止手掌检测（便捷接口）"""
    global _global_detector
    if _global_detector:
        _global_detector.stop_detection()


def is_hand_present():
    """检测是否有手掌（便捷接口）"""
    detector = get_detector()
    return detector.is_hand_present()


def manual_capture(save_dir="image"):
    """手动拍照（便捷接口）"""
    detector = get_detector(save_dir)
    return detector.manual_capture()


def get_detection_status():
    """获取检测状态（便捷接口）"""
    global _global_detector
    if _global_detector:
        return _global_detector.get_status()
    return {'running': False, 'hand_present': False, 'has_captured': False, 'waiting_time': 0}


# ===== 使用示例 =====
def example_usage():
    """使用示例"""
    print("=== 手掌检测系统使用示例 ===")

    # 方式1: 简单使用
    print("\n1. 简单使用:")
    start_hand_detection(save_dir="captures", blocking=False)
    time.sleep(10)  # 运行10秒
    stop_hand_detection()

    # 方式2: 带回调函数
    print("\n2. 带回调函数:")

    def on_hand_detected():
        print("🖐️ 回调: 检测到手掌")

    def on_hand_disappeared():
        print("👋 回调: 手掌消失")

    def on_photo_captured(filepath, frame):
        print(f"📸 回调: 照片已保存到 {filepath}")
        # 这里可以调用 image_matcher.main() 等后续处理

    detector = get_detector()
    detector.set_callbacks(on_hand_detected, on_hand_disappeared, on_photo_captured)
    detector.start_detection()

    # 方式3: 手动控制
    print("\n3. 手动控制:")
    detector = HandCaptureSystem()

    for i in range(10):
        status = detector.get_status()
        print(f"状态: {status}")

        if status['hand_present']:
            print("有手掌")
        else:
            print("无手掌")

        time.sleep(1)


# ===== 主程序入口 =====
def main():
    """主程序"""
    print("选择运行模式:")
    print("1. 阻塞模式检测（主程序）")
    print("2. 非阻塞模式检测（后台运行）")
    print("3. 使用示例")

    choice = input("请选择 (1-3): ").strip()

    if choice == "1":
        start_hand_detection(blocking=True)
    elif choice == "2":
        start_hand_detection(blocking=False)
        print("检测在后台运行，按 Enter 停止...")
        input()
        stop_hand_detection()
    elif choice == "3":
        example_usage()
    else:
        print("直接启动阻塞模式...")
        start_hand_detection(blocking=True)


if __name__ == "__main__":
    main()