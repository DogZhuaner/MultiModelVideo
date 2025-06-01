"""
图像配准和区域截取器
功能：读取图片文件，按照标准图进行配准，截取相同区域
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk


class ImageMatcher:
    def __init__(self):
        self.reference_image = None
        self.target_image = None
        self.regions_data = None
        self.reference_image_path = ""
        self.target_image_path = ""
        self.regions_file_path = ""

        # 配准相关
        self.transform_matrix = None
        self.matching_result = None

        # GUI相关
        self.root = None
        self.canvas_ref = None
        self.canvas_target = None
        self.progress_var = None

    def load_regions_file(self, regions_file_path):
        """加载区域特征文件"""
        try:
            with open(regions_file_path, 'r', encoding='utf-8') as f:
                self.regions_data = json.load(f)

            self.regions_file_path = regions_file_path
            self.reference_image_path = self.regions_data['reference_image']

            # 加载参考图像
            self.reference_image = cv2.imread(self.reference_image_path)
            if self.reference_image is None:
                raise ValueError(f"无法读取参考图像: {self.reference_image_path}")

            print(f"成功加载区域文件: {regions_file_path}")
            print(f"参考图像: {self.reference_image_path}")
            print(f"区域数量: {self.regions_data['regions_count']}")

            return True

        except Exception as e:
            print(f"加载区域文件失败: {e}")
            return False

    def load_target_image(self, target_image_path):
        """加载目标图像"""
        try:
            self.target_image = cv2.imread(target_image_path)
            if self.target_image is None:
                raise ValueError(f"无法读取目标图像: {target_image_path}")

            self.target_image_path = target_image_path
            print(f"成功加载目标图像: {target_image_path}")
            print(f"图像尺寸: {self.target_image.shape}")

            return True

        except Exception as e:
            print(f"加载目标图像失败: {e}")
            return False

    def register_images(self):
        """图像配准"""
        try:
            print("开始图像配准...")

            # 转换为灰度图
            ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

            # 使用SIFT检测特征点
            sift = cv2.SIFT_create()

            # 检测关键点和描述符
            kp1, des1 = sift.detectAndCompute(ref_gray, None)
            kp2, des2 = sift.detectAndCompute(target_gray, None)

            print(f"参考图像特征点: {len(kp1)}")
            print(f"目标图像特征点: {len(kp2)}")

            if len(kp1) < 10 or len(kp2) < 10:
                raise ValueError("特征点太少，无法进行配准")

            # 特征匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 应用Lowe比率测试筛选好的匹配
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            print(f"好的匹配点数量: {len(good_matches)}")

            if len(good_matches) < 10:
                raise ValueError("匹配点太少，无法进行配准")

            # 提取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算单应性矩阵
            self.transform_matrix, mask = cv2.findHomography(src_pts, dst_pts,
                                                             cv2.RANSAC, 5.0)

            if self.transform_matrix is None:
                raise ValueError("无法计算变换矩阵")

            # 保存匹配结果用于可视化
            self.matching_result = {
                'keypoints_ref': kp1,
                'keypoints_target': kp2,
                'good_matches': good_matches,
                'inliers': mask.ravel().tolist() if mask is not None else []
            }

            print("图像配准完成")
            print(f"变换矩阵:\n{self.transform_matrix}")

            return True

        except Exception as e:
            print(f"图像配准失败: {e}")
            return False

    def extract_regions_from_target(self):
        """从目标图像中截取区域"""
        if self.transform_matrix is None:
            print("请先进行图像配准")
            return None

        try:
            extracted_regions = []

            for region_data in self.regions_data['regions']:
                region_id = region_data['id']
                region_name = region_data['name']
                polygon = region_data['polygon']

                print(f"处理区域 {region_id}: {region_name}")

                # 将参考图像中的多边形坐标转换到目标图像坐标
                src_polygon = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                dst_polygon = cv2.perspectiveTransform(src_polygon, self.transform_matrix)
                dst_polygon = dst_polygon.reshape(-1, 2).astype(np.int32)

                # 获取边界框
                x, y, w, h = cv2.boundingRect(dst_polygon)

                # 确保边界框在图像范围内
                h_img, w_img = self.target_image.shape[:2]
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                if w <= 0 or h <= 0:
                    print(f"区域 {region_id} 超出图像边界，跳过")
                    continue

                # 创建掩码
                mask = np.zeros(self.target_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [dst_polygon], 255)

                # 截取区域
                region_roi = self.target_image[y:y + h, x:x + w].copy()
                mask_roi = mask[y:y + h, x:x + w]

                # 应用掩码（将区域外的像素设为黑色）
                region_roi[mask_roi == 0] = [0, 0, 0]

                # 保存区域信息
                region_info = {
                    'id': region_id,
                    'name': region_name,
                    'original_polygon': polygon,
                    'transformed_polygon': dst_polygon.tolist(),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'roi_image': region_roi,
                    'mask': mask_roi
                }

                extracted_regions.append(region_info)
                print(f"成功截取区域 {region_id}: 尺寸 {w}x{h}")

            return extracted_regions

        except Exception as e:
            print(f"区域截取失败: {e}")
            return None

    def save_extracted_regions(self, extracted_regions, save_dir=None):
        """保存截取的区域"""
        try:
            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_name = os.path.splitext(os.path.basename(self.target_image_path))[0]
                save_dir = f"extracted_regions_{target_name}_{timestamp}"

            os.makedirs(save_dir, exist_ok=True)

            # 保存截取规则和信息
            rules_data = {
                'source_target_image': self.target_image_path,
                'source_regions_file': self.regions_file_path,
                'reference_image': self.reference_image_path,
                'transform_matrix': self.transform_matrix.tolist(),
                'extracted_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'regions_info': []
            }

            # 保存每个区域
            for region_info in extracted_regions:
                region_id = region_info['id']
                region_name = region_info['name']
                roi_image = region_info['roi_image']

                # 保存区域图像
                roi_filename = f"region_{region_id:02d}_{region_name}.jpg"
                roi_path = os.path.join(save_dir, roi_filename)
                cv2.imwrite(roi_path, roi_image)

                # 保存掩码
                mask_filename = f"mask_{region_id:02d}_{region_name}.jpg"
                mask_path = os.path.join(save_dir, mask_filename)
                cv2.imwrite(mask_path, region_info['mask'])

                # 记录区域信息
                region_record = {
                    'id': region_id,
                    'name': region_name,
                    'roi_file': roi_filename,
                    'mask_file': mask_filename,
                    'bbox': region_info['bbox'],
                    'original_polygon': region_info['original_polygon'],
                    'transformed_polygon': region_info['transformed_polygon']
                }

                rules_data['regions_info'].append(region_record)

            # 保存截取规则文件
            rules_file = os.path.join(save_dir, 'extraction_rules.json')
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, ensure_ascii=False, indent=2)

            # 创建可视化结果
            self.create_visualization(extracted_regions, save_dir)

            print(f"截取结果已保存到: {save_dir}")
            print(f"共保存 {len(extracted_regions)} 个区域")

            return save_dir

        except Exception as e:
            print(f"保存失败: {e}")
            return None

    def create_visualization(self, extracted_regions, save_dir):
        """创建可视化结果"""
        try:
            # 创建配准结果可视化
            if self.matching_result:
                vis_matches = cv2.drawMatches(
                    self.reference_image, self.matching_result['keypoints_ref'],
                    self.target_image, self.matching_result['keypoints_target'],
                    self.matching_result['good_matches'], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(os.path.join(save_dir, 'registration_matches.jpg'), vis_matches)

            # 创建区域标注图
            target_annotated = self.target_image.copy()

            for region_info in extracted_regions:
                polygon = np.array(region_info['transformed_polygon'], dtype=np.int32)
                cv2.polylines(target_annotated, [polygon], True, (0, 255, 0), 3)

                # 添加区域标签
                center = np.mean(polygon, axis=0).astype(int)
                cv2.putText(target_annotated, f"{region_info['id']}:{region_info['name']}",
                            tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(save_dir, 'annotated_result.jpg'), target_annotated)

            print("可视化结果已保存")

        except Exception as e:
            print(f"创建可视化失败: {e}")

    def create_gui(self):
        """创建图形界面"""
        self.root = tk.Tk()
        self.root.title("图像配准和区域截取器")
        self.root.geometry("1400x900")

        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 文件选择按钮
        tk.Button(control_frame, text="选择区域文件", command=self.select_regions_file,
                  bg='#2196F3', fg='white', font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="选择目标图像", command=self.select_target_image,
                  bg='#2196F3', fg='white', font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="开始配准", command=self.start_registration,
                  bg='#FF9800', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="截取区域", command=self.start_extraction,
                  bg='#4CAF50', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_label = tk.Label(control_frame, text="请选择文件", font=("Arial", 10))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 进度条
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # 图像显示区域
        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 参考图像显示
        ref_frame = tk.LabelFrame(image_frame, text="参考图像", font=("Arial", 12, "bold"))
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas_ref = tk.Canvas(ref_frame, bg='lightgray')
        self.canvas_ref.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 目标图像显示
        target_frame = tk.LabelFrame(image_frame, text="目标图像", font=("Arial", 12, "bold"))
        target_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.canvas_target = tk.Canvas(target_frame, bg='lightgray')
        self.canvas_target.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_regions_file(self):
        """选择区域文件"""
        file_path = filedialog.askopenfilename(
            title="选择区域特征文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )

        if file_path:
            if self.load_regions_file(file_path):
                self.status_label.config(text="区域文件已加载")
                self.display_reference_image()
            else:
                messagebox.showerror("错误", "加载区域文件失败")

    def select_target_image(self):
        """选择目标图像"""
        file_path = filedialog.askopenfilename(
            title="选择目标图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )

        if file_path:
            if self.load_target_image(file_path):
                self.status_label.config(text="目标图像已加载")
                self.display_target_image()
            else:
                messagebox.showerror("错误", "加载目标图像失败")

    def start_registration(self):
        """开始配准"""
        if self.reference_image is None or self.target_image is None:
            messagebox.showwarning("警告", "请先加载参考图像和目标图像")
            return

        self.status_label.config(text="正在进行图像配准...")
        self.progress_var.set(30)
        self.root.update()

        if self.register_images():
            self.progress_var.set(100)
            self.status_label.config(text="图像配准完成")
            messagebox.showinfo("成功", "图像配准完成！")
        else:
            self.progress_var.set(0)
            self.status_label.config(text="图像配准失败")
            messagebox.showerror("错误", "图像配准失败")

    def start_extraction(self):
        """开始截取区域"""
        if self.transform_matrix is None:
            messagebox.showwarning("警告", "请先进行图像配准")
            return

        self.status_label.config(text="正在截取区域...")
        self.progress_var.set(50)
        self.root.update()

        extracted_regions = self.extract_regions_from_target()

        if extracted_regions:
            self.progress_var.set(80)
            self.root.update()

            save_dir = self.save_extracted_regions(extracted_regions)

            if save_dir:
                self.progress_var.set(100)
                self.status_label.config(text="截取完成")
                messagebox.showinfo("成功", f"区域截取完成！\n结果保存在: {save_dir}")
            else:
                self.progress_var.set(0)
                self.status_label.config(text="保存失败")
                messagebox.showerror("错误", "保存截取结果失败")
        else:
            self.progress_var.set(0)
            self.status_label.config(text="截取失败")
            messagebox.showerror("错误", "区域截取失败")

    def display_reference_image(self):
        """显示参考图像"""
        if self.reference_image is not None:
            self.display_image_on_canvas(self.reference_image, self.canvas_ref, show_regions=True)

    def display_target_image(self):
        """显示目标图像"""
        if self.target_image is not None:
            self.display_image_on_canvas(self.target_image, self.canvas_target)

    def display_image_on_canvas(self, image, canvas, show_regions=False):
        """在画布上显示图像"""
        try:
            # 获取画布尺寸
            canvas.update()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 400, 300

            # 计算缩放比例
            h, w = image.shape[:2]
            scale_w = (canvas_width - 20) / w
            scale_h = (canvas_height - 20) / h
            scale = min(scale_w, scale_h, 1.0)

            new_width = int(w * scale)
            new_height = int(h * scale)

            # 调整图像尺寸
            display_img = cv2.resize(image, (new_width, new_height))

            # 绘制区域标注
            if show_regions and self.regions_data:
                for region_data in self.regions_data['regions']:
                    polygon = np.array(region_data['polygon'], dtype=np.float32)
                    polygon *= scale  # 缩放坐标
                    polygon = polygon.astype(np.int32)

                    cv2.polylines(display_img, [polygon], True, (0, 255, 0), 2)

                    # 添加标签
                    center = np.mean(polygon, axis=0).astype(int)
                    cv2.putText(display_img, f"{region_data['id']}",
                                tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 转换为RGB
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_img_rgb)
            photo = ImageTk.PhotoImage(pil_img)

            # 在画布中央显示
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2,
                                image=photo, anchor="center")
            canvas.image = photo  # 保持引用

        except Exception as e:
            print(f"显示图像失败: {e}")

    def run(self):
        """运行程序"""
        self.create_gui()
        self.root.mainloop()


def main():
    """主函数"""
    try:
        matcher = ImageMatcher()
        matcher.run()
    except Exception as e:
        print(f"程序运行失败: {e}")


if __name__ == "__main__":
    main()