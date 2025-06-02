# 原始调用方式分析：
#
# 其他文件可能的调用方式：
# 1. from cameraManager import camera
#    ret, frame = camera.get_frame()
#
# 2. import cameraManager
#    ret, frame = cameraManager.camera.get_frame()
#
# 3. from cameraManager import get_frame  # 如果有这个函数
#    ret, frame = get_frame()

# ===== 完全向后兼容的 camera_manager.py =====

import cv2
import threading
import time
import os
import tempfile
import json
import atexit
from pathlib import Path

# 尝试导入 psutil，如果没有则使用替代方案
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: 未安装psutil，使用简化的进程检测")


class BackwardCompatibleCameraManager:
    """完全向后兼容的摄像头管理器"""

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = False
        self.is_master = False

        # 创建临时目录
        self.temp_dir = Path(tempfile.gettempdir()) / "camera_manager"
        self.temp_dir.mkdir(exist_ok=True)

        self.lock_file = self.temp_dir / f"camera_{src}.lock"
        self.frame_file = self.temp_dir / f"camera_{src}_frame.jpg"
        self.status_file = self.temp_dir / f"camera_{src}_status.json"

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化摄像头管理器"""
        # 检查是否已有主进程
        if self._check_existing_master():
            self.is_master = False
            print(f"连接到现有摄像头主进程 (PID: {self._get_master_pid()})")
        else:
            # 尝试成为主进程
            if self._become_master():
                self.is_master = True
                print(f"成为摄像头主进程 (PID: {os.getpid()})")
                self._start_camera_capture()
            else:
                print("无法成为主进程，使用客户端模式")

    def _check_existing_master(self):
        """检查是否已有主进程在运行"""
        if not self.lock_file.exists():
            return False

        try:
            with open(self.lock_file, 'r') as f:
                pid = int(f.read().strip())

            # 检查进程是否还在运行
            if HAS_PSUTIL:
                return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
            else:
                # 简化的进程检测（Windows兼容）
                return self._simple_pid_check(pid)

        except (ValueError, FileNotFoundError, Exception):
            # 锁文件损坏，删除它
            self.lock_file.unlink(missing_ok=True)
            return False

    def _simple_pid_check(self, pid):
        """简化的进程存在检查"""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                                        capture_output=True, text=True)
                return str(pid) in result.stdout
            else:  # Unix-like
                os.kill(pid, 0)  # 发送信号0检查进程是否存在
                return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _get_master_pid(self):
        """获取主进程PID"""
        try:
            with open(self.lock_file, 'r') as f:
                return int(f.read().strip())
        except:
            return None

    def _become_master(self):
        """尝试成为主进程"""
        try:
            # 创建锁文件
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))

            # 注册退出清理
            atexit.register(self._cleanup_master)
            return True

        except Exception as e:
            print(f"无法创建主进程锁: {e}")
            return False

    def _cleanup_master(self):
        """清理主进程资源"""
        if self.is_master:
            self.running = False

            # 清理文件
            for file_path in [self.lock_file, self.frame_file, self.status_file]:
                try:
                    file_path.unlink(missing_ok=True)
                except:
                    pass

    def _start_camera_capture(self):
        """启动摄像头捕获（仅主进程）"""
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 {self.src}")

            # 设置摄像头参数 - 保持与原版本相同
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.running = True

            # 启动捕获线程
            self.update_thread = threading.Thread(target=self.update, daemon=True)
            self.update_thread.start()

            print("摄像头初始化成功")

        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            self.is_master = False

    def update(self):
        """更新循环 - 保持与原版本相同的方法名"""
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()

                # 更新内存中的帧（保持原有行为）
                self.ret = ret
                self.frame = frame

                if ret and frame is not None:
                    try:
                        # 保存到文件供其他进程使用
                        cv2.imwrite(str(self.frame_file), frame)

                        # 更新状态
                        status = {
                            'ret': True,
                            'timestamp': time.time(),
                            'frame_shape': frame.shape,
                            'pid': os.getpid()
                        }

                        with open(self.status_file, 'w') as f:
                            json.dump(status, f)

                    except Exception as e:
                        print(f"保存共享帧失败: {e}")

            time.sleep(0.033)  # 保持原有的更新频率

    def get_frame(self):
        """获取当前帧 - 保持与原版本完全相同的接口"""
        if self.is_master:
            # 主进程直接返回内存中的帧（保持原有性能）
            return self.ret, self.frame
        else:
            # 客户端进程从文件读取帧
            return self._get_shared_frame()

    def _get_shared_frame(self):
        """从共享文件获取帧"""
        try:
            # 检查状态文件
            if not self.status_file.exists():
                return False, None

            with open(self.status_file, 'r') as f:
                status = json.load(f)

            # 检查状态和时效性
            if not status.get('ret', False):
                return False, None

            if time.time() - status.get('timestamp', 0) > 2.0:  # 2秒超时
                return False, None

            # 读取帧文件
            if self.frame_file.exists():
                frame = cv2.imread(str(self.frame_file))
                if frame is not None:
                    return True, frame

            return False, None

        except Exception as e:
            # 静默处理错误，避免影响其他进程
            return False, None

    def release(self):
        """释放资源 - 保持与原版本相同的接口"""
        if self.is_master:
            self.running = False
            if self.cap is not None:
                self.cap.release()

        # 清理资源
        self._cleanup_master()


# ===== 保持完全向后兼容的全局实例和接口 =====

# 创建全局实例（与原版本完全相同）
camera = BackwardCompatibleCameraManager()


# 如果原来有这些函数，保持它们（可选的兼容性函数）
def get_frame():
    """全局函数接口（如果有的话）"""
    return camera.get_frame()


def release():
    """全局释放函数接口（如果有的话）"""
    return camera.release()


# ===== 测试和验证代码 =====
if __name__ == "__main__":
    print("=== 摄像头管理器兼容性测试 ===")
    print(f"当前进程 PID: {os.getpid()}")
    print(f"是否为主进程: {camera.is_master}")
    print(f"临时文件目录: {camera.temp_dir}")

    # 测试原有接口
    print("\n测试 camera.get_frame() 接口:")
    for i in range(10):
        ret, frame = camera.get_frame()
        if ret:
            print(f"  帧 {i + 1}: 成功 - 尺寸 {frame.shape}")
        else:
            print(f"  帧 {i + 1}: 失败")
        time.sleep(0.5)

    # 测试全局函数接口
    print("\n测试 get_frame() 全局函数接口:")
    ret, frame = get_frame()
    if ret:
        print(f"  全局函数测试: 成功 - 尺寸 {frame.shape}")
    else:
        print("  全局函数测试: 失败")

    print("\n=== 兼容性测试完成 ===")
    print("可以同时运行多个此脚本来测试多进程功能")

    # 释放资源
    camera.release()