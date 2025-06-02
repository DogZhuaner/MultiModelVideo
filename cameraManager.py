# camera_manager.py - 防闪烁版本
import cv2
import threading
import time
import os
import tempfile
import json
import atexit
import numpy as np
from pathlib import Path
import pickle
import struct

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class AntiFlickerCameraManager:
    """防闪烁的摄像头管理器"""

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.is_master = False

        # 防闪烁相关
        self.frame_buffer = []  # 帧缓冲区
        self.buffer_size = 3  # 缓冲帧数
        self.last_frame = None  # 上一帧缓存
        self.frame_lock = threading.RLock()  # 可重入锁

        # 文件路径
        self.temp_dir = Path(tempfile.gettempdir()) / "camera_manager"
        self.temp_dir.mkdir(exist_ok=True)

        self.lock_file = self.temp_dir / f"camera_{src}.lock"
        self.frame_file_a = self.temp_dir / f"camera_{src}_frame_a.dat"
        self.frame_file_b = self.temp_dir / f"camera_{src}_frame_b.dat"
        self.status_file = self.temp_dir / f"camera_{src}_status.json"

        # 双缓冲控制
        self.current_buffer = 'a'
        self.frame_counter = 0

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化摄像头管理器"""
        if self._check_existing_master():
            self.is_master = False
            print(f"连接到现有摄像头主进程")
            self._start_client_monitor()
        else:
            if self._become_master():
                self.is_master = True
                print(f"成为摄像头主进程")
                self._start_camera_capture()

    def _check_existing_master(self):
        """检查是否已有主进程"""
        if not self.lock_file.exists():
            return False

        try:
            with open(self.lock_file, 'r') as f:
                pid = int(f.read().strip())

            if HAS_PSUTIL:
                return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
            else:
                return self._simple_pid_check(pid)

        except (ValueError, FileNotFoundError, Exception):
            self.lock_file.unlink(missing_ok=True)
            return False

    def _simple_pid_check(self, pid):
        """简化的进程检查"""
        try:
            if os.name == 'nt':
                import subprocess
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                                        capture_output=True, text=True)
                return str(pid) in result.stdout
            else:
                os.kill(pid, 0)
                return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _become_master(self):
        """成为主进程"""
        try:
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))
            atexit.register(self._cleanup_master)
            return True
        except Exception as e:
            print(f"无法创建主进程锁: {e}")
            return False

    def _cleanup_master(self):
        """清理主进程资源"""
        if self.is_master:
            self.running = False
            for file_path in [self.lock_file, self.frame_file_a, self.frame_file_b, self.status_file]:
                try:
                    file_path.unlink(missing_ok=True)
                except:
                    pass

    def _start_camera_capture(self):
        """启动摄像头捕获"""
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 {self.src}")

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 设置缓冲区大小（减少延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.running = True

            # 启动捕获线程
            self.update_thread = threading.Thread(target=self.update, daemon=True)
            self.update_thread.start()

            print("摄像头初始化成功")

        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            self.is_master = False

    def update(self):
        """主进程更新循环 - 使用双缓冲防闪烁"""
        consecutive_failures = 0

        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    consecutive_failures = 0

                    with self.frame_lock:
                        # 更新内存帧
                        self.ret = True
                        self.frame = frame.copy()

                        # 更新帧缓冲区
                        self.frame_buffer.append(frame.copy())
                        if len(self.frame_buffer) > self.buffer_size:
                            self.frame_buffer.pop(0)

                    # 保存到文件 - 使用双缓冲
                    self._save_frame_with_double_buffer(frame)

                    self.frame_counter += 1

                else:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print("摄像头连续读取失败，尝试重新初始化...")
                        self._reinitialize_camera()
                        consecutive_failures = 0

            time.sleep(0.033)  # 约30FPS

    def _save_frame_with_double_buffer(self, frame):
        """使用双缓冲保存帧"""
        try:
            # 选择当前要写入的缓冲区
            if self.current_buffer == 'a':
                write_file = self.frame_file_a
                next_buffer = 'b'
            else:
                write_file = self.frame_file_b
                next_buffer = 'a'

            # 序列化帧数据
            frame_data = {
                'frame': frame,
                'timestamp': time.time(),
                'counter': self.frame_counter,
                'shape': frame.shape
            }

            # 写入临时文件，然后原子性移动
            temp_file = write_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(frame_data, f)

            # 原子性移动（避免读取时文件不完整）
            temp_file.replace(write_file)

            # 更新状态文件
            status = {
                'ret': True,
                'timestamp': time.time(),
                'current_buffer': self.current_buffer,
                'frame_counter': self.frame_counter,
                'pid': os.getpid()
            }

            status_temp = self.status_file.with_suffix('.tmp')
            with open(status_temp, 'w') as f:
                json.dump(status, f)
            status_temp.replace(self.status_file)

            # 切换缓冲区
            self.current_buffer = next_buffer

        except Exception as e:
            print(f"保存帧失败: {e}")

    def _reinitialize_camera(self):
        """重新初始化摄像头"""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(self.src)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("摄像头重新初始化成功")
        except Exception as e:
            print(f"摄像头重新初始化失败: {e}")

    def _start_client_monitor(self):
        """启动客户端监控线程"""
        self.client_thread = threading.Thread(target=self._client_monitor_loop, daemon=True)
        self.client_thread.start()

    def _client_monitor_loop(self):
        """客户端监控循环 - 预加载帧"""
        last_counter = -1

        while True:
            try:
                frame_data = self._load_latest_frame()
                if frame_data and frame_data.get('counter', -1) > last_counter:
                    with self.frame_lock:
                        self.ret = True
                        self.frame = frame_data['frame'].copy()
                        self.last_frame = frame_data['frame'].copy()

                    last_counter = frame_data['counter']

            except Exception as e:
                # 静默处理错误
                pass

            time.sleep(0.020)  # 50Hz监控频率

    def get_frame(self):
        """获取当前帧 - 防闪烁版本"""
        if self.is_master:
            # 主进程：直接返回最新帧
            with self.frame_lock:
                if self.frame is not None:
                    return self.ret, self.frame.copy()
                else:
                    return False, None
        else:
            # 客户端：返回缓存的帧，避免实时读取造成闪烁
            with self.frame_lock:
                if self.frame is not None:
                    return True, self.frame.copy()
                elif self.last_frame is not None:
                    return True, self.last_frame.copy()
                else:
                    # 备用：尝试读取一次
                    return self._get_shared_frame_safe()

    def _load_latest_frame(self):
        """加载最新帧数据"""
        try:
            # 读取状态文件
            if not self.status_file.exists():
                return None

            with open(self.status_file, 'r') as f:
                status = json.load(f)

            if not status.get('ret', False):
                return None

            # 检查时效性
            if time.time() - status.get('timestamp', 0) > 2.0:
                return None

            # 确定要读取的缓冲区文件
            current_buffer = status.get('current_buffer', 'a')
            if current_buffer == 'a':
                read_file = self.frame_file_b  # 读取另一个缓冲区
            else:
                read_file = self.frame_file_a

            # 读取帧数据
            if read_file.exists():
                with open(read_file, 'rb') as f:
                    frame_data = pickle.load(f)
                return frame_data

            return None

        except Exception as e:
            return None

    def _get_shared_frame_safe(self):
        """安全地获取共享帧（备用方法）"""
        try:
            frame_data = self._load_latest_frame()
            if frame_data and 'frame' in frame_data:
                return True, frame_data['frame'].copy()
            return False, None
        except:
            return False, None

    def release(self):
        """释放资源"""
        if self.is_master:
            self.running = False
            if self.cap is not None:
                self.cap.release()
        self._cleanup_master()


# ===== 兼容性保持 =====
camera = AntiFlickerCameraManager()


def get_frame():
    return camera.get_frame()


def release():
    return camera.release()


# ===== 测试代码 =====
if __name__ == "__main__":
    print("=== 防闪烁摄像头管理器测试 ===")
    print(f"进程 PID: {os.getpid()}")
    print(f"是否主进程: {camera.is_master}")

    # 连续获取帧测试
    print("\n连续帧测试（检查闪烁）:")
    prev_frame = None
    flicker_count = 0

    for i in range(50):
        ret, frame = camera.get_frame()
        if ret and frame is not None:
            # 检测帧差异（简单的闪烁检测）
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                mean_diff = np.mean(diff)
                if mean_diff > 50:  # 阈值可调整
                    flicker_count += 1
                    print(f"  帧 {i + 1}: 可能闪烁 (差异: {mean_diff:.1f})")
                else:
                    print(f"  帧 {i + 1}: 正常 (差异: {mean_diff:.1f})")
            else:
                print(f"  帧 {i + 1}: 首帧 - 尺寸 {frame.shape}")

            prev_frame = frame.copy()

            # 显示帧（如果有显示环境）
            try:
                cv2.imshow('Anti-Flicker Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
        else:
            print(f"  帧 {i + 1}: 获取失败")

        time.sleep(0.1)

    print(f"\n闪烁检测结果: {flicker_count}/50 帧可能有闪烁")
    if flicker_count < 5:
        print("✅ 闪烁控制良好")
    else:
        print("⚠️ 可能需要进一步优化")

    camera.release()
    cv2.destroyAllWindows()