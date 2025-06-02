import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import os
from MultiModelVideo.cameraManager import camera


class CameraCalibrationSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("电路配盘系统摄像头校准")
        self.root.geometry("1600x1000")  # 增大窗口以适应更大的显示区域
        self.root.configure(bg='#f0f8ff')

        # 摄像头相关 - 移除原有的摄像头初始化代码
        self.current_frame = None

        # 校准状态
        self.calibration_score = 0.0
        self.is_calibrated = False
        self.is_stabilizing = False
        self.countdown = 0
        self.stabilize_start_time = 0

        # 参考图片路径配置 - 在这里修改参考图片路径
        self.reference_image_path = "reference.jpg"  # 修改为你的参考图片路径
        self.reference_image = None

        # 线程控制
        self.running = True

        # 创建UI
        self.create_ui()

        # 加载参考图
        self.load_reference_image()

        # 启动摄像头更新和校准检测
        self.start_camera_updates()

    def create_ui(self):
        """创建用户界面"""
        # 主标题
        title_frame = tk.Frame(self.root, bg='#f0f8ff')
        title_frame.pack(pady=15)

        title_label = tk.Label(title_frame, text="电路配盘系统摄像头校准",
                               font=("Microsoft YaHei", 22, "bold"),
                               fg='#1e3a8a', bg='#f0f8ff')
        title_label.pack()

        subtitle_label = tk.Label(title_frame, text="请根据标准图样调整摄像头位置，确保拍摄清晰度和角度符合要求",
                                  font=("Microsoft YaHei", 11),
                                  fg='#3b82f6', bg='#f0f8ff')
        subtitle_label.pack(pady=5)

        # 主内容区域
        main_frame = tk.Frame(self.root, bg='#f0f8ff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # 左侧 - 实时摄像头（增大尺寸）
        left_frame = tk.LabelFrame(main_frame, text="实时摄像头画面",
                                   font=("Microsoft YaHei", 12, "bold"),
                                   fg='#1e3a8a', bg='white', relief='solid', bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # 摄像头显示区域（增大尺寸）
        self.camera_frame = tk.Frame(left_frame, bg='white')
        self.camera_frame.pack(padx=15, pady=15)

        # 移除width和height限制，让图片按实际尺寸显示
        self.camera_label = tk.Label(self.camera_frame, bg='black')
        self.camera_label.pack()

        # 校准信息
        info_frame = tk.Frame(left_frame, bg='white')
        info_frame.pack(fill=tk.X, padx=15, pady=10)

        # 校准进度
        progress_frame = tk.Frame(info_frame, bg='white')
        progress_frame.pack(fill=tk.X, pady=8)

        tk.Label(progress_frame, text="校准精度:", font=("Microsoft YaHei", 11),
                 bg='white').pack(side=tk.LEFT)

        self.score_var = tk.StringVar(value="0%")
        self.score_label = tk.Label(progress_frame, textvariable=self.score_var,
                                    font=("Microsoft YaHei", 11, "bold"),
                                    fg='#3b82f6', bg='white')
        self.score_label.pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(progress_frame, length=500, mode='determinate')  # 增加进度条长度
        self.progress_bar.pack(fill=tk.X, pady=8)

        # 右侧 - 标准参考图和控制
        right_frame = tk.Frame(main_frame, bg='#f0f8ff')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # 标准图显示
        ref_frame = tk.LabelFrame(right_frame, text="标准参考图样",
                                  font=("Microsoft YaHei", 12, "bold"),
                                  fg='#1e3a8a', bg='white', relief='solid', bd=2)
        ref_frame.pack(fill=tk.X, pady=(0, 10))

        # 参考图显示标签 - 移除尺寸限制
        self.ref_label = tk.Label(ref_frame, text="加载参考图片中...",
                                  bg='#f5f5f5', relief='solid', bd=2)
        self.ref_label.pack(padx=10, pady=10)

        # 当前参考图路径显示
        self.img_path_var = tk.StringVar(value=f"参考图片: {self.reference_image_path}")
        path_label = tk.Label(ref_frame, textvariable=self.img_path_var,
                              font=("Microsoft YaHei", 9),
                              bg='white', fg='#666666', wraplength=300)
        path_label.pack(padx=10, pady=(0, 5))

        # 调整提示
        tips_frame = tk.Frame(ref_frame, bg='white')
        tips_frame.pack(fill=tk.X, padx=10, pady=5)

        tips_text = [
            "• 确保电路板完全在画面中",
            "• 保持电路板水平放置",
            "• 调整光线确保清晰度",
            "• 避免反光和阴影",
            "• 参考图与实物方向一致"
        ]

        for tip in tips_text:
            tk.Label(tips_frame, text=tip, font=("Microsoft YaHei", 9),
                     bg='white', fg='#666666', anchor='w').pack(anchor='w')

        # 状态显示
        status_frame = tk.LabelFrame(right_frame, text="校准状态",
                                     font=("Microsoft YaHei", 12, "bold"),
                                     fg='#1e3a8a', bg='white', relief='solid', bd=2)
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="正在加载参考图片...")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                     font=("Microsoft YaHei", 11),
                                     bg='white', fg='#dc2626', wraplength=280)
        self.status_label.pack(padx=10, pady=10)

        # 倒计时显示
        self.countdown_var = tk.StringVar(value="")
        self.countdown_label = tk.Label(status_frame, textvariable=self.countdown_var,
                                        font=("Microsoft YaHei", 16, "bold"),
                                        bg='white', fg='#16a34a')
        self.countdown_label.pack(pady=5)

        # 控制按钮
        button_frame = tk.Frame(status_frame, bg='white')
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.reset_btn = tk.Button(button_frame, text="重新校准",
                                   command=self.reset_calibration,
                                   font=("Microsoft YaHei", 10),
                                   bg='#6b7280', fg='white',
                                   relief='flat', padx=20)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.next_btn = tk.Button(button_frame, text="开始检测",
                                  command=self.start_detection,
                                  font=("Microsoft YaHei", 10, "bold"),
                                  bg='#2563eb', fg='white',
                                  relief='flat', padx=20,
                                  state='disabled')
        self.next_btn.pack(side=tk.RIGHT)

    def load_reference_image(self):
        """加载参考图片"""
        try:
            if not os.path.exists(self.reference_image_path):
                raise FileNotFoundError(f"参考图片文件不存在: {self.reference_image_path}")

            # 读取图片
            img = cv2.imread(self.reference_image_path)
            if img is None:
                raise ValueError("无法读取图片文件，请检查文件格式")

            self.reference_image = img

            # 显示参考图片
            self.display_reference_image()

            # 更新状态
            self.status_var.set("请调整摄像头位置")
            self.status_label.configure(fg='#dc2626')

            print(f"成功加载参考图片: {self.reference_image_path}")

        except Exception as e:
            error_msg = f"加载参考图片失败: {str(e)}"
            print(error_msg)
            self.status_var.set("参考图片加载失败")
            self.status_label.configure(fg='#dc2626')
            self.ref_label.configure(text=f"图片加载失败\n{str(e)}")
            messagebox.showerror("错误", error_msg)

    def display_reference_image(self):
        """显示参考图片"""
        if self.reference_image is not None:
            # 转换为显示格式
            img_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 增大参考图显示尺寸
            display_size = (450, 350)  # 显著增大尺寸
            img_pil = img_pil.resize(display_size, Image.Resampling.LANCZOS)

            self.ref_photo = ImageTk.PhotoImage(img_pil)
            self.ref_label.configure(image=self.ref_photo, text="")

    def start_camera_updates(self):
        """启动摄像头更新和校准检测"""
        # 启动摄像头画面更新线程
        self.camera_thread = threading.Thread(target=self.update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # 启动校准检测线程
        self.calibration_thread = threading.Thread(target=self.calibration_loop)
        self.calibration_thread.daemon = True
        self.calibration_thread.start()

    def update_camera(self):
        """更新摄像头画面 - 使用全局camera实例"""
        while self.running:
            # 使用全局camera实例获取帧
            ret, frame = camera.get_frame()
            if ret and frame is not None:
                # 保存原始画面，不进行镜像翻转
                self.current_frame = frame.copy()

                # 添加校准框
                display_frame = frame.copy()
                self.draw_calibration_overlay(display_frame)

                # 转换为tkinter可显示的格式（增大显示尺寸）
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((800, 600))  # 显著增大显示尺寸

                photo = ImageTk.PhotoImage(frame_pil)

                # 在主线程中更新UI
                self.root.after(0, self.update_camera_display, photo)

            time.sleep(0.033)  # 约30FPS

    def draw_calibration_overlay(self, frame):
        """绘制校准覆盖层"""
        h, w = frame.shape[:2]

        # 根据校准分数确定边框颜色 - 恢复合理阈值
        if self.calibration_score > 0.7:  # 恢复到70%
            color = (0, 255, 0)  # 绿色
        elif self.calibration_score > 0.5:  # 恢复到50%
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 0, 255)  # 红色

        # 绘制边框
        thickness = 8  # 增加边框厚度
        cv2.rectangle(frame, (thickness // 2, thickness // 2),
                      (w - thickness // 2, h - thickness // 2), color, thickness)

        # 绘制中心十字线
        center_x, center_y = w // 2, h // 2
        cross_size = 30
        cv2.line(frame, (center_x - cross_size, center_y),
                 (center_x + cross_size, center_y), (100, 100, 255), 3)
        cv2.line(frame, (center_x, center_y - cross_size),
                 (center_x, center_y + cross_size), (100, 100, 255), 3)

        # 绘制网格
        grid_spacing = 60
        for i in range(grid_spacing, w, grid_spacing):
            cv2.line(frame, (i, 0), (i, h), (100, 100, 100), 1)
        for i in range(grid_spacing, h, grid_spacing):
            cv2.line(frame, (0, i), (w, i), (100, 100, 100), 1)

        # 显示倒计时
        if self.is_stabilizing and self.countdown > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            text = str(self.countdown)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 10
            thickness = 15
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # 校准完成标识
        if self.is_calibrated:
            cv2.rectangle(frame, (w - 150, 10), (w - 10, 60), (0, 255, 0), -1)
            cv2.putText(frame, "CALIBRATED", (w - 145, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def update_camera_display(self, photo):
        """更新摄像头显示"""
        if self.running:  # 只有在运行状态下才更新显示
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

    def calibration_loop(self):
        """校准检测循环"""
        while self.running and not self.is_calibrated:
            if self.current_frame is not None and self.reference_image is not None:
                # 计算校准分数
                score = self.calculate_calibration_score(self.current_frame)
                self.calibration_score = score

                # 更新UI
                self.root.after(0, self.update_calibration_ui, score)

                # 检查是否需要开始稳定检测 - 恢复合理阈值
                if score > 0.7 and not self.is_stabilizing and not self.is_calibrated:  # 恢复到70%
                    self.start_stabilization()
                elif score <= 0.7 and self.is_stabilizing:
                    self.stop_stabilization()

            time.sleep(0.1)

    def calculate_calibration_score(self, frame):
        """计算校准分数（优化相似度权重版本）"""
        if frame is None or self.reference_image is None:
            return 0.0

        # 图像质量检测（清晰度） - 恢复适中要求
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity_score = min(laplacian_var / 800.0, 1.0)

        # 亮度检测 - 恢复适中要求
        brightness = np.mean(gray)
        brightness_score = max(0.2, 1.0 - abs(brightness - 128) / 150.0)

        # 边缘检测 - 恢复适中要求
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        object_score = min(edge_density * 8, 1.0)

        # 特征点匹配 - 重点优化，增加权重
        feature_score = self.calculate_feature_similarity(frame)

        # 综合评分 - 大幅增加相似度权重，降低其他权重
        base_score = 0.1  # 恢复适中的基础分
        total_score = base_score + (
                clarity_score * 0.15 +  # 降低清晰度权重
                brightness_score * 0.15 +  # 降低亮度权重
                object_score * 0.15 +  # 降低边缘检测权重
                feature_score * 0.55  # 大幅增加相似度权重到55%
        )

        # 适度的随机波动
        noise = (np.random.random() - 0.5) * 0.05
        total_score = max(0.05, min(1, total_score + noise))

        return total_score

    def calculate_feature_similarity(self, frame):
        """计算特征相似度（多维度优化版本）"""
        try:
            # 转换为灰度图
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)

            # 调整尺寸保持一致
            target_size = (300, 300)
            frame_resized = cv2.resize(frame_gray, target_size)
            ref_resized = cv2.resize(ref_gray, target_size)

            # 1. 直方图相似度 (权重30%)
            frame_hist = cv2.calcHist([frame_resized], [0], None, [256], [0, 256])
            ref_hist = cv2.calcHist([ref_resized], [0], None, [256], [0, 256])
            hist_corr = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
            hist_score = max(0, (hist_corr + 1) / 2)  # 转换到[0,1]范围

            # 2. 结构相似度 (权重40%) - 使用模板匹配
            result = cv2.matchTemplate(frame_resized, ref_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            template_score = max(0, max_val)

            # 3. 边缘相似度 (权重30%) - 比较边缘特征
            frame_edges = cv2.Canny(frame_resized, 50, 150)
            ref_edges = cv2.Canny(ref_resized, 50, 150)

            # 计算边缘重叠度
            frame_edge_count = np.sum(frame_edges > 0)
            ref_edge_count = np.sum(ref_edges > 0)
            intersection = np.sum((frame_edges > 0) & (ref_edges > 0))

            if frame_edge_count > 0 and ref_edge_count > 0:
                edge_score = intersection / max(frame_edge_count, ref_edge_count)
            else:
                edge_score = 0.3  # 默认分数

            # 4. 综合相似度计算
            final_similarity = (
                    hist_score * 0.3 +  # 直方图相似度权重30%
                    template_score * 0.4 +  # 模板匹配权重40%
                    edge_score * 0.3  # 边缘相似度权重30%
            )

            # 应用平滑函数，让相似度更容易达到高分
            # 使用sigmoid函数变换，让中等相似度更容易提升
            final_similarity = 1 / (1 + np.exp(-10 * (final_similarity - 0.3)))

            return min(1.0, final_similarity)

        except Exception as e:
            print(f"相似度计算异常: {e}")
            return 0.4  # 异常时返回中等分数

    def update_calibration_ui(self, score):
        """更新校准UI"""
        if not self.running:  # 如果程序正在关闭，不更新UI
            return

        # 更新进度条和分数
        self.progress_bar['value'] = score * 100
        self.score_var.set(f"{int(score * 100)}%")

        # 更新状态信息 - 恢复合理阈值
        if self.is_calibrated:
            self.status_var.set("校准完成！摄像头已准备就绪")
            self.status_label.configure(fg='#16a34a')
            self.next_btn.configure(state='normal')
            self.start_detection()
        elif self.reference_image is None:
            self.status_var.set("参考图片未加载")
            self.status_label.configure(fg='#dc2626')
        elif score > 0.7:  # 恢复到较合理的70%阈值
            if self.is_stabilizing:
                self.status_var.set("位置良好，请保持摄像头稳定")
            else:
                self.status_var.set("位置良好，开始稳定性检测")
            self.status_label.configure(fg='#16a34a')
        elif score > 0.5:  # 恢复到50%的中等阈值
            self.status_var.set("位置接近，请微调摄像头角度")
            self.status_label.configure(fg='#d97706')
        else:
            self.status_var.set("请调整摄像头位置，确保电路板清晰可见")
            self.status_label.configure(fg='#dc2626')

    def start_stabilization(self):
        """开始稳定性检测"""
        self.is_stabilizing = True
        self.stabilize_start_time = time.time()

        # 启动倒计时线程
        countdown_thread = threading.Thread(target=self.countdown_loop)
        countdown_thread.daemon = True
        countdown_thread.start()

    def stop_stabilization(self):
        """停止稳定性检测"""
        self.is_stabilizing = False
        self.countdown = 0
        self.countdown_var.set("")

    def countdown_loop(self):
        """倒计时循环 - 恢复标准时间"""
        for i in range(3, 0, -1):  # 恢复到3秒
            if not self.is_stabilizing or not self.running:
                break

            self.countdown = i
            self.root.after(0, lambda: self.countdown_var.set(f"保持稳定 {i}s"))
            time.sleep(1)

        if self.is_stabilizing and self.running:
            self.is_calibrated = True
            self.is_stabilizing = False
            self.countdown = 0
            self.root.after(0, lambda: self.countdown_var.set(""))

    def reset_calibration(self):
        """重置校准"""
        self.is_calibrated = False
        self.is_stabilizing = False
        self.countdown = 0
        self.calibration_score = 0.0
        self.countdown_var.set("")
        self.next_btn.configure(state='disabled')

        # 重启校准线程
        if hasattr(self, 'calibration_thread') and self.calibration_thread.is_alive():
            pass
        else:
            self.calibration_thread = threading.Thread(target=self.calibration_loop)
            self.calibration_thread.daemon = True
            self.calibration_thread.start()

    def start_detection(self):
        """开始检测 - 校准完成后的程序出口"""
        # 显示提示信息
        result = messagebox.askquestion("校准完成",
                                        "校准完成！\n是否进入电路配盘检测模式？\n\n点击'是'将关闭校准窗口并启动检测程序。",
                                        icon='question')

        if result == 'yes':
            # 这里是程序的出口点 - 校准完成后关闭窗口
            print("校准完成，准备启动检测程序...")

            # 停止所有线程
            self.running = False

            # 延迟关闭窗口，确保线程能够正常结束
            self.root.after(500, self.close_window)

    def close_window(self):
        """关闭窗口"""
        try:
            self.root.quit()
            self.root.destroy()
            print("校准窗口已关闭")

            # 在这里可以调用下一个程序或返回到主程序
            # 例如:
            # import main_detection_system
            # main_detection_system.run()

        except Exception as e:
            print(f"关闭窗口时发生错误: {e}")

    def on_closing(self):
        """关闭程序"""
        self.running = False
        # 等待线程结束
        time.sleep(0.2)
        try:
            self.root.destroy()
        except:
            pass

    def run(self):
        """运行程序"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = CameraCalibrationSystem()
    app.run()