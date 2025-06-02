"""
图像配准和区域截取器（优化版）
功能：读取图片文件，按照标准图进行配准，截取相同区域
优化：提高配准精度，减少位置偏移
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "regions_rules.json")
image_path = os.path.join(base_dir, "live_capture.jpg")
reference_path = os.path.join(base_dir, "standard.jpg")

class ImageMatcher:
    def __init__(self, regions_file_path, target_image_path):
        """
        初始化图像配准器

        Args:
            regions_file_path: 区域特征文件路径
            target_image_path: 目标图像文件路径
        """
        self.regions_file_path = regions_file_path
        self.target_image_path = target_image_path

        self.reference_image = None
        self.target_image = None
        self.regions_data = None
        self.reference_image_path = ""

        # 配准相关
        self.transform_matrix = None
        self.matching_result = None
        self.registration_quality = 0.0

        print("=== 图像配准和区域截取器（优化版）===")
        print(f"区域文件: {regions_file_path}")
        print(f"目标图像: {target_image_path}")

    def load_data(self):
        """加载所有必要的数据文件"""
        print("\n--- 加载数据文件 ---")

        # 加载区域特征文件
        if not self.load_regions_file():
            return False

        # 加载目标图像
        if not self.load_target_image():
            return False

        print("✓ 所有数据文件加载完成")
        return True

    def load_regions_file(self):
        """加载区域特征文件"""
        try:
            print(f"正在加载区域文件: {self.regions_file_path}")

            if not os.path.exists(self.regions_file_path):
                raise FileNotFoundError(f"区域文件不存在: {self.regions_file_path}")

            with open(self.regions_file_path, 'r', encoding='utf-8') as f:
                self.regions_data = json.load(f)

            self.reference_image_path = reference_path

            # 加载参考图像
            if not os.path.exists(self.reference_image_path):
                raise FileNotFoundError(f"参考图像不存在: {self.reference_image_path}")

            self.reference_image = cv2.imread(self.reference_image_path)
            if self.reference_image is None:
                raise ValueError(f"无法读取参考图像: {self.reference_image_path}")

            print(f"✓ 参考图像: {self.reference_image_path}")
            print(f"✓ 参考图像尺寸: {self.reference_image.shape}")
            print(f"✓ 区域数量: {self.regions_data['regions_count']}")

            return True

        except Exception as e:
            print(f"✗ 加载区域文件失败: {e}")
            return False

    def load_target_image(self):
        """加载目标图像"""
        try:
            print(f"正在加载目标图像: {self.target_image_path}")

            if not os.path.exists(self.target_image_path):
                raise FileNotFoundError(f"目标图像不存在: {self.target_image_path}")

            self.target_image = cv2.imread(self.target_image_path)
            if self.target_image is None:
                raise ValueError(f"无法读取目标图像: {self.target_image_path}")

            print(f"✓ 目标图像尺寸: {self.target_image.shape}")

            return True

        except Exception as e:
            print(f"✗ 加载目标图像失败: {e}")
            return False

    def preprocess_images(self):
        """预处理图像以提高配准效果"""
        try:
            # 转换为灰度图
            ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

            # 直方图均衡化提高对比度
            ref_gray = cv2.equalizeHist(ref_gray)
            target_gray = cv2.equalizeHist(target_gray)

            # 高斯滤波减少噪声
            ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
            target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)

            return ref_gray, target_gray

        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None, None

    def detect_and_match_features(self, ref_gray, target_gray):
        """检测和匹配特征点"""
        try:
            print("正在检测SIFT特征点...")

            # 创建SIFT检测器，增加特征点数量
            sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=20)

            # 检测关键点和描述符
            kp1, des1 = sift.detectAndCompute(ref_gray, None)
            kp2, des2 = sift.detectAndCompute(target_gray, None)

            print(f"参考图像特征点: {len(kp1)}")
            print(f"目标图像特征点: {len(kp2)}")

            if len(kp1) < 50 or len(kp2) < 50:
                print("⚠ 特征点较少，可能影响配准精度")

            if des1 is None or des2 is None:
                raise ValueError("无法提取特征描述符")

            # 使用FLANN匹配器进行快速匹配
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # 应用Lowe比率测试，使用更严格的阈值
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.6 * n.distance:  # 更严格的阈值
                        good_matches.append(m)

            print(f"有效匹配点数量: {len(good_matches)}")

            if len(good_matches) < 20:
                raise ValueError("匹配点太少，无法进行可靠配准")

            return kp1, kp2, good_matches

        except Exception as e:
            print(f"特征检测和匹配失败: {e}")
            return None, None, None

    def calculate_robust_homography(self, kp1, kp2, good_matches):
        """计算鲁棒的单应性矩阵"""
        try:
            print("正在计算变换矩阵...")

            # 提取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用更严格的RANSAC参数计算单应性矩阵
            transform_matrix, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=3.0,    # 更严格的重投影阈值
                maxIters=5000,                # 增加最大迭代次数
                confidence=0.995              # 提高置信度
            )

            if transform_matrix is None:
                raise ValueError("无法计算变换矩阵")

            # 计算配准质量
            inliers_count = np.sum(mask) if mask is not None else 0
            self.registration_quality = inliers_count / len(good_matches)

            print(f"内点数量: {inliers_count}/{len(good_matches)}")
            print(f"内点比例: {self.registration_quality*100:.1f}%")

            # 检查配准质量
            if self.registration_quality < 0.3:
                print("⚠ 配准质量较低，结果可能不准确")
            elif self.registration_quality > 0.7:
                print("✓ 配准质量良好")

            # 检查变换矩阵的合理性
            if not self.validate_homography(transform_matrix):
                print("⚠ 变换矩阵可能不合理")

            return transform_matrix, mask

        except Exception as e:
            print(f"计算变换矩阵失败: {e}")
            return None, None

    def validate_homography(self, H):
        """验证单应性矩阵的合理性"""
        try:
            # 检查矩阵条件数
            det = np.linalg.det(H[:2, :2])
            if abs(det) < 0.1 or abs(det) > 10:
                return False

            # 检查尺度变化是否合理
            scale_x = np.sqrt(H[0,0]**2 + H[1,0]**2)
            scale_y = np.sqrt(H[0,1]**2 + H[1,1]**2)

            if scale_x < 0.1 or scale_x > 10 or scale_y < 0.1 or scale_y > 10:
                return False

            return True

        except:
            return False

    def register_images(self):
        """图像配准主函数"""
        try:
            print("\n--- 开始图像配准 ---")

            # 图像预处理
            ref_gray, target_gray = self.preprocess_images()
            if ref_gray is None or target_gray is None:
                return False

            # 特征检测和匹配
            kp1, kp2, good_matches = self.detect_and_match_features(ref_gray, target_gray)
            if kp1 is None or kp2 is None or good_matches is None:
                return False

            # 计算单应性矩阵
            self.transform_matrix, mask = self.calculate_robust_homography(kp1, kp2, good_matches)
            if self.transform_matrix is None:
                return False

            # 保存匹配结果用于可视化
            self.matching_result = {
                'keypoints_ref': kp1,
                'keypoints_target': kp2,
                'good_matches': good_matches,
                'inliers': mask.ravel().tolist() if mask is not None else []
            }

            print("✓ 图像配准完成")
            print(f"变换矩阵:\n{self.transform_matrix}")

            return True

        except Exception as e:
            print(f"✗ 图像配准失败: {e}")
            return False

    def extract_regions_from_target(self):
        """从目标图像中截取区域"""
        if self.transform_matrix is None:
            print("✗ 请先进行图像配准")
            return None

        try:
            print("\n--- 开始截取区域 ---")
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

                # 获取扩展的边界框以确保完整截取
                x, y, w, h = cv2.boundingRect(dst_polygon)

                # 添加边距以防止截取不完整
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(w + 2*margin, self.target_image.shape[1] - x)
                h = min(h + 2*margin, self.target_image.shape[0] - y)

                # 确保边界框在图像范围内
                if w <= 0 or h <= 0:
                    print(f"  ⚠ 区域 {region_id} 超出图像边界，跳过")
                    continue

                # 创建精确的掩码
                mask = np.zeros(self.target_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [dst_polygon], 255)

                # 截取区域（使用精确掩码）
                region_roi = self.target_image[y:y+h, x:x+w].copy()
                mask_roi = mask[y:y+h, x:x+w]

                # 保存完整的ROI（不应用掩码遮挡）
                # 这样可以保留完整的区域信息

                # 计算区域在ROI中的相对位置
                roi_polygon = dst_polygon.copy()
                roi_polygon[:, 0] -= x
                roi_polygon[:, 1] -= y

                # 保存区域信息
                region_info = {
                    'id': region_id,
                    'name': region_name,
                    'original_polygon': polygon,
                    'transformed_polygon': dst_polygon.tolist(),
                    'roi_polygon': roi_polygon.tolist(),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'roi_image': region_roi,
                    'mask': mask_roi
                }

                extracted_regions.append(region_info)
                print(f"  ✓ 成功截取: 尺寸 {w}x{h}, 位置 ({x},{y})")

                # 输出配准精度信息
                center_orig = np.mean(src_polygon.reshape(-1, 2), axis=0)
                center_transformed = np.mean(dst_polygon, axis=0)
                print(f"    区域中心: 原始({center_orig[0]:.1f},{center_orig[1]:.1f}) -> 变换({center_transformed[0]:.1f},{center_transformed[1]:.1f})")

            print(f"✓ 共成功截取 {len(extracted_regions)} 个区域")
            return extracted_regions

        except Exception as e:
            print(f"✗ 区域截取失败: {e}")
            return None

    def save_extracted_regions(self, extracted_regions, save_dir=None):
        """保存截取的区域"""
        try:
            print("\n--- 保存截取结果 ---")

            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_name = os.path.splitext(os.path.basename(self.target_image_path))[0]
                save_dir = "split"

            os.makedirs(save_dir, exist_ok=True)
            print(f"保存目录: {save_dir}")

            # 保存配准质量信息
            quality_info = {
                'registration_quality': self.registration_quality,
                'quality_level': 'Good' if self.registration_quality > 0.7 else 'Fair' if self.registration_quality > 0.3 else 'Poor'
            }

            with open(os.path.join(save_dir, 'quality_report.txt'), 'w', encoding='utf-8') as f:
                f.write(f"配准质量报告\n")
                f.write(f"================\n")
                f.write(f"内点比例: {self.registration_quality*100:.1f}%\n")
                f.write(f"质量等级: {quality_info['quality_level']}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 保存每个区域图像
            print("保存区域图像:")
            for region_info in extracted_regions:
                region_id = region_info['id']
                region_name = region_info['name']
                roi_image = region_info['roi_image']

                # 保存区域图像
                roi_filename = f"region_{region_name}.jpg"
                roi_path = os.path.join(save_dir, roi_filename)
                cv2.imwrite(roi_path, roi_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                print(f"  ✓ {roi_filename}")

            # 创建可视化结果
            self.create_visualization(extracted_regions, save_dir)

            print(f"\n✓ 截取结果已保存到: {save_dir}")
            print(f"✓ 共保存 {len(extracted_regions)} 个区域")
            print(f"✓ 配准质量: {quality_info['quality_level']} ({self.registration_quality*100:.1f}%)")

            return save_dir

        except Exception as e:
            print(f"✗ 保存失败: {e}")
            return None

    def create_visualization(self, extracted_regions, save_dir):
        """创建可视化结果"""
        try:
            print("创建可视化结果:")

            # 创建配准结果可视化（显示匹配点）
            if self.matching_result:
                print("  正在生成配准匹配图...")

                # 只显示内点
                inlier_matches = []
                for i, match in enumerate(self.matching_result['good_matches']):
                    if i < len(self.matching_result['inliers']) and self.matching_result['inliers'][i]:
                        inlier_matches.append(match)

                # 绘制匹配结果
                vis_matches = cv2.drawMatches(
                    self.reference_image, self.matching_result['keypoints_ref'],
                    self.target_image, self.matching_result['keypoints_target'],
                    inlier_matches[:50], None,  # 只显示前50个最好的匹配
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                match_file = os.path.join(save_dir, 'registration_matches.jpg')
                cv2.imwrite(match_file, vis_matches, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"  ✓ registration_matches.jpg")

            # 创建区域标注图
            print("  正在生成标注结果图...")
            target_annotated = self.target_image.copy()

            # 绘制所有变换后的区域
            for region_info in extracted_regions:
                polygon = np.array(region_info['transformed_polygon'], dtype=np.int32)

                # 绘制区域边界
                cv2.polylines(target_annotated, [polygon], True, (0, 255, 0), 2)

                # 填充半透明区域
                overlay = target_annotated.copy()
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                target_annotated = cv2.addWeighted(target_annotated, 0.8, overlay, 0.2, 0)

                # 添加区域标签
                center = np.mean(polygon, axis=0).astype(int)
                label_text = f"{region_info['id']}:{region_info['name']}"

                # 添加文字背景
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(target_annotated,
                             (center[0] - text_width//2 - 5, center[1] - text_height - 5),
                             (center[0] + text_width//2 + 5, center[1] + 5),
                             (255, 255, 255), -1)

                cv2.putText(target_annotated, label_text,
                           (center[0] - text_width//2, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            annotated_file = os.path.join(save_dir, 'annotated_result.jpg')
            cv2.imwrite(annotated_file, target_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  ✓ annotated_result.jpg")

            print("✓ 可视化结果已保存")

        except Exception as e:
            print(f"✗ 创建可视化失败: {e}")

    def process(self):
        """执行完整的处理流程"""
        print("开始处理流程...")

        # 步骤1: 加载数据
        if not self.load_data():
            print("✗ 数据加载失败，处理终止")
            return False

        # 步骤2: 图像配准
        if not self.register_images():
            print("✗ 图像配准失败，处理终止")
            return False

        # 步骤3: 截取区域
        extracted_regions = self.extract_regions_from_target()
        if extracted_regions is None:
            print("✗ 区域截取失败，处理终止")
            return False

        # 步骤4: 保存结果
        save_dir = self.save_extracted_regions(extracted_regions)
        if save_dir is None:
            print("✗ 结果保存失败，处理终止")
            return False

        print(f"\n🎉 处理完成！结果保存在: {save_dir}")
        return True

def main():
    """主函数 - 在这里配置文件路径"""

    # ==================== 配置区域 ====================
    # 修改这些路径为你的实际文件路径

    REGIONS_FILE_PATH = json_path  # 区域特征文件路径
    TARGET_IMAGE_PATH = image_path # 目标图像路径

    # ================================================

    try:
        # 创建图像配准器实例
        matcher = ImageMatcher(REGIONS_FILE_PATH, TARGET_IMAGE_PATH)

        # 执行处理流程
        success = matcher.process()

        if success:
            print("\n✅ 程序执行成功！")
            #送入模型读取split中的分割图
        else:
            print("\n❌ 程序执行失败！")
            #在ui界面中设置弹窗

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

if __name__ == "__main__":
    main()