# camera_manager.py - 4K超高质量版本
import cv2
import threading
import time
import os
import socket
import numpy as np
from pathlib import Path


class UltraHQ4KCameraManager:
    """4K超高质量摄像头管理器"""

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.is_master = False
        self.lock_socket = None

        # 线程锁
        self.frame_lock = threading.Lock()

        # 4K质量参数
        self.target_width = 3840  # 4K宽度
        self.target_height = 2160  # 4K高度
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0

        # 共享文件路径 - 4K需要更大存储
        self.share_dir = Path.home() / ".camera_share_4k"
        self.share_dir.mkdir(exist_ok=True)
        self.frame_file = self.share_dir / f"camera_{src}_frame_4k.png"
        self.status_file = self.share_dir / f"camera_{src}_status.txt"

        # 性能优化参数
        self.save_quality = 95  # PNG压缩质量
        self.target_fps = 15  # 目标帧率（4K下保持流畅）
        self.frame_interval = 1.0 / self.target_fps  # 帧间隔
        self.last_save_time = 0

        # 尝试成为主进程
        self._try_become_master()

        if self.is_master:
            print(f"🎥 成为4K摄像头主进程 (PID: {os.getpid()})")
            self._start_camera_4k()
        else:
            print(f"📱 连接到4K摄像头主进程 (PID: {os.getpid()})")
            self._start_client()

    def _try_become_master(self):
        """使用端口锁定机制判断是否为主进程"""
        try:
            self.lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.lock_socket.bind(('127.0.0.1', 9000 + self.src))
            self.is_master = True
        except OSError:
            self.is_master = False
            if self.lock_socket:
                self.lock_socket.close()
                self.lock_socket = None

    def _start_camera_4k(self):
        """启动4K摄像头（仅主进程）"""
        try:
            print("🔍 初始化4K摄像头...")

            # 尝试不同的后端，DirectShow通常对高分辨率支持更好
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "MediaFoundation"),
                (cv2.CAP_ANY, "自动选择")
            ]

            for backend, name in backends:
                print(f"  尝试后端: {name}")
                self.cap = cv2.VideoCapture(self.src, backend)
                if self.cap.isOpened():
                    print(f"  ✅ {name} 后端成功")
                    break
            else:
                raise Exception("❌ 所有后端都无法打开摄像头")

            # ===== 4K分辨率设置 =====
            print("📐 设置4K分辨率...")

            # 首先尝试4K分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            # 检查实际分辨率
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"  目标分辨率: {self.target_width}x{self.target_height}")
            print(f"  实际分辨率: {self.actual_width}x{self.actual_height}")

            # 如果不支持4K，尝试其他高分辨率
            if self.actual_width < 3840:
                print("  📏 4K不支持，尝试其他高分辨率...")

                # 尝试的分辨率列表（从高到低）
                resolutions = [
                    (2560, 1440, "2K QHD"),
                    (1920, 1080, "1080p Full HD"),
                    (1280, 720, "720p HD")
                ]

                for width, height, name in resolutions:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                    actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if actual_w >= width * 0.9 and actual_h >= height * 0.9:  # 允许10%误差
                        self.actual_width = actual_w
                        self.actual_height = actual_h
                        print(f"  ✅ 使用 {name}: {actual_w}x{actual_h}")
                        break
                else:
                    print("  ⚠️ 使用默认分辨率")

            # ===== 帧率设置 =====
            print("🎬 优化帧率设置...")

            # 根据分辨率调整目标帧率
            if self.actual_width >= 3840:  # 4K
                self.target_fps = 15
            elif self.actual_width >= 2560:  # 2K
                self.target_fps = 20
            elif self.actual_width >= 1920:  # 1080p
                self.target_fps = 25
            else:  # 720p或更低
                self.target_fps = 30

            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_interval = 1.0 / self.target_fps

            print(f"  目标帧率: {self.target_fps} FPS")
            print(f"  实际帧率: {self.actual_fps} FPS")

            # ===== 图像质量参数设置 =====
            print("🎨 优化图像质量参数...")

            try:
                # 基础图像参数
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # 亮度
                self.cap.set(cv2.CAP_PROP_CONTRAST, 140)  # 对比度稍微提高
                self.cap.set(cv2.CAP_PROP_SATURATION, 130)  # 饱和度稍微提高

                # 高级参数（如果支持）
                if hasattr(cv2, 'CAP_PROP_SHARPNESS'):
                    self.cap.set(cv2.CAP_PROP_SHARPNESS, 140)  # 锐度

                if hasattr(cv2, 'CAP_PROP_AUTO_WB'):
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 自动白平衡

                if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 自动曝光

                # 关闭自动增益（如果支持）以保持图像质量稳定
                if hasattr(cv2, 'CAP_PROP_GAIN'):
                    self.cap.set(cv2.CAP_PROP_GAIN, 0)

                print("  ✅ 图像质量参数设置完成")

            except Exception as e:
                print(f"  ⚠️ 部分参数设置失败: {e}")

            # ===== 缓冲区优化 =====
            print("💾 优化缓冲区设置...")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区，减少延迟

            # ===== 摄像头预热 =====
            print("🔥 摄像头预热...")
            for i in range(10):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"  预热帧 {i + 1}/10: {frame.shape}")
                time.sleep(0.1)

            # 启动捕获线程
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop_4k, daemon=True)
            self.update_thread.start()

            print("🚀 4K超高质量摄像头启动成功！")
            print(f"📊 最终配置: {self.actual_width}x{self.actual_height} @ {self.target_fps}FPS")

        except Exception as e:
            print(f"❌ 4K摄像头启动失败: {e}")
            self.is_master = False

    def _start_client(self):
        """启动客户端模式"""
        self.running = True
        self.client_thread = threading.Thread(target=self._client_loop, daemon=True)
        self.client_thread.start()

    def _update_loop_4k(self):
        """4K摄像头更新循环（仅主进程）"""
        frame_count = 0
        last_fps_time = time.time()
        last_frame_time = time.time()

        while self.running:
            if self.cap and self.cap.isOpened():
                current_time = time.time()

                # 帧率控制 - 确保不超过目标帧率
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # 短暂休息
                    continue

                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # ===== 4K图像质量增强 =====
                    frame = self._enhance_4k_quality(frame)

                    with self.frame_lock:
                        self.ret = True
                        self.frame = frame.copy()

                    # 控制保存频率以保持性能
                    if current_time - self.last_save_time >= 1.0 / 10:  # 最多10FPS保存
                        self._save_shared_frame_4k(frame)
                        self.last_save_time = current_time

                    frame_count += 1
                    last_frame_time = current_time

                    # 每5秒输出一次性能统计
                    if current_time - last_fps_time >= 5.0:
                        actual_fps = frame_count / (current_time - last_fps_time)
                        print(f"📊 4K性能: {actual_fps:.1f} FPS | 已处理 {frame_count} 帧")
                        last_fps_time = current_time
                        frame_count = 0

                else:
                    with self.frame_lock:
                        self.ret = False
                        self.frame = None
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def _enhance_4k_quality(self, frame):
        """4K图像质量增强"""
        try:
            # 对于4K图像，使用更精细的增强算法

            # 1. 轻微降噪（保持细节）
            frame = cv2.bilateralFilter(frame, 3, 20, 20)

            # 2. 自适应锐化
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 根据图像清晰度调整锐化强度
            if laplacian_var < 500:  # 图像模糊，增强锐化
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]]) * 0.3
            else:  # 图像清晰，轻微锐化
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]]) * 0.1

            kernel[1, 1] += 1  # 确保权重和为1
            frame = cv2.filter2D(frame, -1, kernel)

            # 3. 色彩空间优化
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.05)  # 轻微提升饱和度
            hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.02)  # 轻微提升明度
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 4. 确保像素值在有效范围内
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            return frame

        except Exception as e:
            print(f"4K图像增强失败: {e}")
            return frame

    def _client_loop(self):
        """客户端更新循环"""
        while self.running:
            frame = self._load_shared_frame_4k()

            with self.frame_lock:
                if frame is not None:
                    self.ret = True
                    self.frame = frame
                else:
                    self.ret = False
                    self.frame = None

            time.sleep(0.1)  # 客户端更新频率可以低一些

    def _save_shared_frame_4k(self, frame):
        """4K高质量保存帧到共享文件"""
        try:
            # 使用PNG格式，最高质量保存
            success = cv2.imwrite(str(self.frame_file), frame, [
                cv2.IMWRITE_PNG_COMPRESSION, 1,  # 最快压缩（文件较大但速度快）
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
            ])

            if success:
                # 更新状态文件
                with open(self.status_file, 'w') as f:
                    f.write(f"{time.time()},{os.getpid()},{self.actual_width},{self.actual_height}")

        except Exception as e:
            print(f"保存4K帧失败: {e}")

    def _load_shared_frame_4k(self):
        """4K高质量加载共享帧"""
        try:
            # 检查状态文件
            if not self.status_file.exists():
                return None

            with open(self.status_file, 'r') as f:
                content = f.read().strip()
                parts = content.split(',')
                if len(parts) >= 2:
                    timestamp = float(parts[0])

                    # 检查是否过期（3秒内的帧才有效，4K处理可能慢一些）
                    if time.time() - timestamp > 3.0:
                        return None

            # 加载PNG图像
            if self.frame_file.exists():
                frame = cv2.imread(str(self.frame_file), cv2.IMREAD_COLOR)
                return frame

            return None

        except Exception as e:
            return None

    def get_frame(self):
        """获取当前帧"""
        with self.frame_lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            else:
                return False, None

    def get_4k_info(self):
        """获取4K摄像头详细信息"""
        if not self.is_master or not self.cap:
            return "非主进程或摄像头未初始化"

        info = {
            'resolution': f"{self.actual_width}x{self.actual_height}",
            'target_fps': self.target_fps,
            'actual_fps': self.actual_fps,
            'is_4k': self.actual_width >= 3840,
            'is_2k': self.actual_width >= 2560,
            'is_fhd': self.actual_width >= 1920,
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'share_dir': str(self.share_dir),
            'frame_file_size': self.frame_file.stat().st_size if self.frame_file.exists() else 0
        }
        return info

    def save_4k_sample(self, filename="4k_sample.png"):
        """保存4K样本图片"""
        ret, frame = self.get_frame()
        if ret and frame is not None:
            success = cv2.imwrite(filename, frame, [
                cv2.IMWRITE_PNG_COMPRESSION, 0,  # 无压缩，最高质量
            ])
            if success:
                file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
                print(f"✅ 4K样本已保存: {filename} ({file_size:.1f} MB)")
                return filename
            else:
                print(f"❌ 保存失败: {filename}")
                return None
        else:
            print("❌ 无法获取帧")
            return None

    def release(self):
        """释放资源"""
        self.running = False

        if self.is_master:
            if self.cap:
                self.cap.release()

            # 清理共享文件
            try:
                if self.frame_file.exists():
                    self.frame_file.unlink()
                if self.status_file.exists():
                    self.status_file.unlink()
            except:
                pass

        if self.lock_socket:
            self.lock_socket.close()


# ===== 全局实例和兼容接口 =====
camera = UltraHQ4KCameraManager()


def get_frame():
    """全局函数接口"""
    return camera.get_frame()


def release():
    """全局释放接口"""
    return camera.release()


def get_4k_info():
    """获取4K摄像头信息"""
    return camera.get_4k_info()


def save_4k_sample(filename="4k_sample.png"):
    """保存4K样本"""
    return camera.save_4k_sample(filename)


# ===== 测试代码 =====
if __name__ == "__main__":
    print("=== 4K超高质量摄像头管理器测试 ===")
    print(f"进程类型: {'主进程' if camera.is_master else '客户端进程'}")
    print(f"进程PID: {os.getpid()}")

    if camera.is_master:
        info = get_4k_info()
        print(f"\n📊 4K摄像头信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 测试获取帧质量
    print("\n🎬 开始4K质量测试:")
    for i in range(3):
        ret, frame = camera.get_frame()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            megapixels = (height * width) / 1000000
            print(f"✅ 帧 {i + 1}: {width}x{height} ({megapixels:.1f}MP)")

            # 保存测试图片
            test_filename = f"4k_test_{i + 1}.png"
            success = cv2.imwrite(test_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if success:
                file_size = os.path.getsize(test_filename) / (1024 * 1024)
                print(f"  💾 保存: {test_filename} ({file_size:.1f} MB)")
        else:
            print(f"❌ 帧 {i + 1}: 失败")
        time.sleep(2)

    # 保存高质量样本
    if camera.is_master:
        print("\n📸 保存4K样本图片...")
        save_4k_sample("ultra_hq_4k_sample.png")

    print("\n🔍 质量检查建议:")
    print("1. 查看保存的PNG文件分辨率和文件大小")
    print("2. 检查图像清晰度和色彩饱和度")
    print("3. 确认帧率是否流畅")

    camera.release()
    print("测试完成")