"""
图像区域选择器（修改版）
功能：在标准图上手动选择多个区域，提取特征并保存
优化：图像放大显示，尽量占满UI界面
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

base_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = os.path.join(base_dir, "standard.jpg")

class RegionSelector:
    def __init__(self, reference_image_path):
        self.reference_image_path = reference_image_path
        self.reference_image = None
        self.display_image = None
        self.current_region = []
        self.regions = []
        self.region_features = []
        self.drawing = False
        self.scale_factor = 1.0

        # GUI相关
        self.root = None
        self.canvas = None
        self.canvas_image = None

        # 加载参考图像
        self.load_reference_image()

    def load_reference_image(self):
        """加载参考图像"""
        try:
            self.reference_image = cv2.imread(self.reference_image_path)
            if self.reference_image is None:
                raise ValueError(f"无法读取图像: {self.reference_image_path}")
            print(f"成功加载参考图像: {self.reference_image_path}")
            print(f"图像尺寸: {self.reference_image.shape}")
        except Exception as e:
            print(f"加载图像失败: {e}")
            raise

    def extract_region_features(self, region_coords):
        """提取区域特征"""
        try:
            # 创建掩码
            mask = np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
            pts = np.array(region_coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # 获取边界框
            x, y, w, h = cv2.boundingRect(pts)
            region_roi = self.reference_image[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]

            features = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'polygon': [[int(p[0]), int(p[1])] for p in region_coords],
                'area': float(cv2.contourArea(pts))
            }

            # 1. 颜色特征 - HSV直方图
            hsv_roi = cv2.cvtColor(region_roi, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv_roi], [0], mask_roi, [50], [0, 180])
            hist_s = cv2.calcHist([hsv_roi], [1], mask_roi, [60], [0, 256])
            hist_v = cv2.calcHist([hsv_roi], [2], mask_roi, [60], [0, 256])

            features['color_hist'] = {
                'h': hist_h.flatten().tolist(),
                's': hist_s.flatten().tolist(),
                'v': hist_v.flatten().tolist()
            }

            # 2. 纹理特征 - LBP (简化版)
            gray_roi = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
            lbp_features = self.calculate_lbp_features(gray_roi, mask_roi)
            features['texture'] = lbp_features

            # 3. 形状特征
            contour = pts.reshape(-1, 1, 2)
            features['shape'] = {
                'perimeter': float(cv2.arcLength(contour, True)),
                'aspect_ratio': float(w / h) if h > 0 else 0,
                'extent': float(features['area'] / (w * h)) if w * h > 0 else 0,
                'solidity': float(features['area'] / cv2.contourArea(cv2.convexHull(contour))) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
            }

            # 4. SIFT关键点特征
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_roi, mask_roi)

            if keypoints and descriptors is not None:
                features['sift'] = {
                    'keypoints': len(keypoints),
                    'descriptors': descriptors.tolist()
                }
            else:
                features['sift'] = {'keypoints': 0, 'descriptors': []}

            # 5. 统计特征
            features['statistics'] = {
                'mean_color': [float(np.mean(region_roi[:,:,i])) for i in range(3)],
                'std_color': [float(np.std(region_roi[:,:,i])) for i in range(3)],
                'mean_gray': float(np.mean(gray_roi)),
                'std_gray': float(np.std(gray_roi))
            }

            return features

        except Exception as e:
            print(f"特征提取失败: {e}")
            return None

    def calculate_lbp_features(self, gray_image, mask):
        """计算LBP纹理特征"""
        try:
            height, width = gray_image.shape
            lbp = np.zeros_like(gray_image)

            # 简化的LBP计算
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if mask[i, j] == 0:
                        continue

                    center = gray_image[i, j]
                    code = 0

                    # 8邻域LBP
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]

                    for idx, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << idx)

                    lbp[i, j] = code

            # 计算LBP直方图
            hist = cv2.calcHist([lbp], [0], mask, [256], [0, 256])
            return hist.flatten().tolist()

        except Exception as e:
            print(f"LBP计算失败: {e}")
            return [0] * 256

    def create_gui(self):
        """创建图形界面"""
        self.root = tk.Tk()
        self.root.title("区域选择器 - 点击选择区域")
        self.root.geometry("1500x1100")  # 进一步增大窗口尺寸以容纳大图像

        # 控制面板 - 紧凑布局
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        tk.Label(control_frame, text="操作：左键选择顶点，右键完成区域",
                font=("Arial", 9)).pack(side=tk.LEFT)

        tk.Button(control_frame, text="保存", command=self.save_regions,
                 bg='#4CAF50', fg='white', font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=2)

        tk.Button(control_frame, text="撤销", command=self.undo_last_point,
                 bg='#FF9800', fg='white', font=("Arial", 9)).pack(side=tk.RIGHT, padx=2)

        tk.Button(control_frame, text="清除", command=self.clear_current_region,
                 bg='#f44336', fg='white', font=("Arial", 9)).pack(side=tk.RIGHT, padx=2)

        # 信息面板 - 紧凑布局
        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)

        self.info_label = tk.Label(info_frame, text=f"已选择区域数量: 0", font=("Arial", 8))
        self.info_label.pack(side=tk.LEFT)

        # 图像显示区域 - 增大显示区域
        self.canvas = tk.Canvas(self.root, bg='white', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # 显示图像
        self.display_image_on_canvas()

    def display_image_on_canvas(self):
        """在画布上显示图像"""
        # 获取实际画布尺寸，尽量占满界面
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 如果画布还没有渲染完成，使用默认大尺寸
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 1150  # 增大默认宽度
            canvas_height = 750  # 增大默认高度

        # 留一些边距
        canvas_width -= 20
        canvas_height -= 20

        h, w = self.reference_image.shape[:2]
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        self.scale_factor = min(scale_w, scale_h)  # 移除1.0限制，允许放大

        new_width = int(w * self.scale_factor)
        new_height = int(h * self.scale_factor)

        # 调整图像尺寸
        display_img = cv2.resize(self.reference_image, (new_width, new_height))
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        pil_img = Image.fromarray(display_img_rgb)
        self.canvas_image = ImageTk.PhotoImage(pil_img)

        # 在画布中央显示图像，尽量占满空间
        self.canvas.delete("all")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas_img_id = self.canvas.create_image(
            (canvas_width + 20)//2, (canvas_height + 20)//2,
            image=self.canvas_image, anchor="center"
        )

        # 重新绘制已有区域
        self.redraw_regions()

    def on_left_click(self, event):
        """左键点击事件"""
        # 转换画布坐标到图像坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # 获取图像在画布中的位置
        img_x = canvas_x - self.canvas.coords(self.canvas_img_id)[0] + self.canvas_image.width()//2
        img_y = canvas_y - self.canvas.coords(self.canvas_img_id)[1] + self.canvas_image.height()//2

        # 转换到原始图像坐标
        orig_x = int(img_x / self.scale_factor)
        orig_y = int(img_y / self.scale_factor)

        # 检查坐标是否在图像范围内
        h, w = self.reference_image.shape[:2]
        if 0 <= orig_x < w and 0 <= orig_y < h:
            self.current_region.append([orig_x, orig_y])
            self.redraw_regions()
            print(f"添加点: ({orig_x}, {orig_y})")

    def on_right_click(self, event):
        """右键点击事件 - 完成当前区域"""
        if len(self.current_region) >= 3:
            self.complete_current_region()
        else:
            messagebox.showwarning("警告", "至少需要3个点才能形成一个区域")

    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if len(self.current_region) > 0:
            self.redraw_regions(mouse_pos=(event.x, event.y))

    def complete_current_region(self):
        """完成当前区域选择"""
        if len(self.current_region) >= 3:
            # 请求区域名称
            region_name = simpledialog.askstring("区域名称", "请输入区域名称:")
            if not region_name:
                region_name = f"区域_{len(self.regions) + 1}"

            # 提取特征
            features = self.extract_region_features(self.current_region)
            if features:
                features['name'] = region_name
                features['id'] = len(self.regions)

                self.regions.append(self.current_region.copy())
                self.region_features.append(features)

                print(f"完成区域选择: {region_name}, 顶点数: {len(self.current_region)}")
                self.update_info_label()

                # 清除当前区域
                self.current_region = []
                self.redraw_regions()
            else:
                messagebox.showerror("错误", "特征提取失败")

    def undo_last_point(self):
        """撤销最后一个点"""
        if self.current_region:
            self.current_region.pop()
            self.redraw_regions()
            print("撤销最后一个点")

    def clear_current_region(self):
        """清除当前区域"""
        self.current_region = []
        self.redraw_regions()
        print("清除当前区域")

    def redraw_regions(self, mouse_pos=None):
        """重新绘制所有区域"""
        # 清除之前的绘制（保留图像）
        for item in self.canvas.find_all():
            if item != self.canvas_img_id:
                self.canvas.delete(item)

        # 获取图像在画布中的位置
        img_center_x, img_center_y = self.canvas.coords(self.canvas_img_id)
        img_left = img_center_x - self.canvas_image.width()//2
        img_top = img_center_y - self.canvas_image.height()//2

        # 绘制已完成的区域
        for i, region in enumerate(self.regions):
            canvas_points = []
            for point in region:
                canvas_x = img_left + point[0] * self.scale_factor
                canvas_y = img_top + point[1] * self.scale_factor
                canvas_points.extend([canvas_x, canvas_y])

            if len(canvas_points) >= 6:  # 至少3个点
                self.canvas.create_polygon(canvas_points, outline='red', fill='', width=2, tags='completed')

                # 绘制区域标签
                center_x = sum(canvas_points[::2]) / len(canvas_points[::2])
                center_y = sum(canvas_points[1::2]) / len(canvas_points[1::2])
                self.canvas.create_text(center_x, center_y, text=f"区域{i+1}",
                                      fill='red', font=("Arial", 12, "bold"), tags='completed')

        # 绘制当前正在选择的区域
        if self.current_region:
            canvas_points = []
            for point in self.current_region:
                canvas_x = img_left + point[0] * self.scale_factor
                canvas_y = img_top + point[1] * self.scale_factor
                canvas_points.extend([canvas_x, canvas_y])

                # 绘制顶点
                self.canvas.create_oval(canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3,
                                      fill='blue', outline='blue', tags='current')

            # 绘制线段
            if len(canvas_points) >= 4:
                for i in range(0, len(canvas_points)-2, 2):
                    self.canvas.create_line(canvas_points[i], canvas_points[i+1],
                                          canvas_points[i+2], canvas_points[i+3],
                                          fill='blue', width=2, tags='current')

            # 绘制到鼠标的预览线
            if mouse_pos and len(canvas_points) >= 2:
                self.canvas.create_line(canvas_points[-2], canvas_points[-1],
                                      mouse_pos[0], mouse_pos[1],
                                      fill='gray', width=1, dash=(5, 5), tags='current')

    def update_info_label(self):
        """更新信息标签"""
        self.info_label.config(text=f"已选择区域数量: {len(self.regions)}")

    def save_regions(self):
        """保存所有区域特征"""
        if not self.region_features:
            messagebox.showwarning("警告", "没有选择任何区域")
            return

        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.reference_image_path))[0]
            save_path = f"regions_rules.json"

            # 保存数据
            save_data = {
                'reference_image': self.reference_image_path,
                'image_shape': self.reference_image.shape,
                'regions_count': len(self.region_features),
                'regions': self.region_features,
                'created_time': timestamp
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("成功", f"区域特征已保存到: {save_path}")
            print(f"保存成功: {save_path}")
            print(f"共保存 {len(self.region_features)} 个区域")

        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def run(self):
        """运行区域选择器"""
        self.create_gui()
        self.root.mainloop()

def main():
    # 配置参考图像路径
    REFERENCE_IMAGE_PATH = reference_path   # 修改为你的参考图像路径

    try:
        selector = RegionSelector(REFERENCE_IMAGE_PATH)
        selector.run()
    except Exception as e:
        print(f"程序运行失败: {e}")

if __name__ == "__main__":
    main()