import sys
import cv2
import numpy as np
import logging
import time
import base64
import io
import json
import requests
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QProgressBar, QStatusBar, QTabWidget, QFileDialog, QMessageBox, QLineEdit, QFormLayout, QDialog,
    QDialogButtonBox, QAction
)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
import uuid
# OpenAI GPT支持
try:
    import openai
except ImportError:
    openai = None

OPENAI_API_KEY = "在这里填写你的OpenAI API Key"  # 例如：sk-xxxxxx

logging.basicConfig(filename='app.log', level=logging.INFO)

class CameraThread(QThread):
    new_display_frame = pyqtSignal(bytes)
    capture_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.last_frame = None

    def run(self):
        cap = None
        try:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    break
            else:
                raise RuntimeError("无法打开摄像头")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                self.last_frame = frame.copy()
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if success:
                    self.new_display_frame.emit(buffer.tobytes())
                self.msleep(30)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()

    def stop(self):
        self.running = False
        self.wait(1000)

    def capture_current_frame(self):
        if self.last_frame is not None:
            self.capture_frame_signal.emit(self.last_frame.copy())

class AnalysisWorker(QThread):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, frame_data, model_server, model_name, prompt=None):
        super().__init__()
        self.frame_data = frame_data
        self.model_server = model_server
        self.model_name = model_name
        self.prompt = prompt

    def enhance_image(self, img):
        # 自动白平衡
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # 对比度增强
        alpha = 1.2  # 对比度控制
        beta = 10    # 亮度控制
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # 锐化
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    def run(self):
        try:
            buffer = np.frombuffer(self.frame_data, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (224, 224))
            frame = self.enhance_image(frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with io.BytesIO() as buf:
                img.save(buf, format='JPEG', quality=70)
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            prompt = self.prompt if self.prompt else "请分析当前电路接线操作"
            api_url = f"{self.model_server}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [img_b64],
                "options": {"temperature": 0.2, "num_predict": 512}
            }
            resp = requests.post(api_url, json=data, timeout=60)
            if resp.status_code != 200:
                raise Exception(f"模型服务错误: {resp.text}")
            result = resp.json().get('response', '')
            try:
                json_start = result.find('{')
                json_end = result.rfind('}')
                if json_start != -1 and json_end != -1:
                    result_json = json.loads(result[json_start:json_end+1])
                else:
                    result_json = {"操作描述": result, "得分": 0, "是否正确": False, "错误提示": "未能解析得分", "正确接法": ""}
            except Exception:
                result_json = {"操作描述": result, "得分": 0, "是否正确": False, "错误提示": "未能解析得分", "正确接法": ""}
            self.analysis_complete.emit(result_json)
        except Exception as e:
            self.analysis_complete.emit({"操作描述": f"分析失败: {str(e)}", "得分": 0, "是否正确": False, "错误提示": "系统异常", "正确接法": ""})

class MarkableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marks = []  # [{"name": "C1", "pos": [x, y]}]
        self.display_pixmap = None

    def setPixmap(self, pixmap):
        self.display_pixmap = pixmap
        super().setPixmap(pixmap)

    def set_marks(self, marks):
        self.marks = marks
        if self.display_pixmap:
            self._draw_marks()

    def _draw_marks(self):
        if not self.display_pixmap:
            return
        pixmap = self.display_pixmap.copy()
        painter = QPainter(pixmap)
        painter.setPen(QColor(255,0,0))
        font = QFont("Microsoft YaHei", 16, QFont.Bold)
        painter.setFont(font)
        for mark in self.marks:
            x, y = mark["pos"]
            painter.drawText(x, y, mark["name"])
        painter.end()
        super().setPixmap(pixmap)

class OpenAIGPTWorker(QThread):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, prompt, model_name="gpt-3.5-turbo", api_key=None):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.api_key = api_key

    def run(self):
        if openai is None:
            self.analysis_complete.emit({"操作描述": "未安装openai库", "得分": 0, "是否正确": False, "错误提示": "未安装openai库", "正确接法": ""})
            return
        try:
            openai.api_key = self.api_key or OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            result = response.choices[0].message["content"]
            try:
                json_start = result.find('{')
                json_end = result.rfind('}')
                if json_start != -1 and json_end != -1:
                    result_json = json.loads(result[json_start:json_end+1])
                else:
                    result_json = {"操作描述": result, "得分": 0, "是否正确": False, "错误提示": "未能解析得分", "正确接法": ""}
            except Exception:
                result_json = {"操作描述": result, "得分": 0, "是否正确": False, "错误提示": "未能解析得分", "正确接法": ""}
            self.analysis_complete.emit(result_json)
        except Exception as e:
            self.analysis_complete.emit({"操作描述": f"OpenAI分析失败: {str(e)}", "得分": 0, "是否正确": False, "错误提示": "系统异常", "正确接法": ""})

class PDFParseWorker(QThread):
    parse_complete = pyqtSignal(str, str)  # (text, error)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            import PyPDF2
            with open(self.path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
            self.parse_complete.emit(text, "")
        except Exception as e:
            self.parse_complete.emit("", str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电路配盘接线操作分析系统")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QMainWindow { background: #f4f6fb; }
            QGroupBox { border-radius: 12px; border: 1.5px solid #e0e3e7; background: #fff; margin-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; font-size: 20px; color: #4361ee; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4361ee, stop:1 #38b6ff); color: white; border: none; border-radius: 8px; padding: 10px 24px; font-size: 18px; font-weight: bold; }
            QPushButton:disabled { background: #b0b8c1; color: #fff; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3a56d4, stop:1 #38b6ff); }
            QLabel#scoreLabel { font-size: 36px; color: #e67e22; font-weight: bold; margin-top: 10px; }
            QLabel#commentLabel { font-size: 20px; color: #333; margin-bottom: 10px; }
            QTextEdit { border-radius: 8px; border: 1.5px solid #e0e3e7; background: #fafdff; font-size: 18px; padding: 10px; }
            QTableWidget { border-radius: 8px; border: 1.5px solid #e0e3e7; background: #fff; font-size: 16px; }
            QHeaderView::section { background: #f4f6fb; font-weight: bold; font-size: 16px; border: none; height: 32px; }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:selected { background: #cce8ff; }
        """)
        self._init_ui()
        self._init_variables()
        self._update_responsive_ui()  # 初始化时自适应

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_ui()

    def _update_responsive_ui(self):
        w = self.width()
        h = self.height()
        scale = min(w / 1200, h / 700)
        # 主字体
        base_font_size = 12
        font = QFont("Microsoft YaHei", max(int(base_font_size * scale), 10))
        self.setFont(font)
        # 标题字体
        title_label = self.findChild(QLabel, "titleLabel")
        if title_label:
            title_label.setFont(QFont("Microsoft YaHei", max(int(32 * scale), 16), QFont.Bold))
        # 评分和评语字体
        self.score_label.setFont(QFont("Microsoft YaHei", max(int(36 * scale), 16), QFont.Bold))
        self.comment_label.setFont(QFont("Microsoft YaHei", max(int(20 * scale), 12)))
        # 视频区最小尺寸
        self.video_label.setMinimumSize(int(480 * scale), int(360 * scale))
        # 表格字体和行高
        table_font = QFont("Microsoft YaHei", max(int(16 * scale), 10))
        self.history_table.setFont(table_font)
        for row in range(self.history_table.rowCount()):
            self.history_table.setRowHeight(row, int(32 * scale))
        self.history_table.horizontalHeader().setDefaultSectionSize(int(120 * scale))

    def _init_ui(self):
        # 顶部彩色标题栏
        title_bar = QWidget()
        title_bar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4361ee, stop:1 #38b6ff); border-radius: 0 0 18px 18px;")
        title_layout = QHBoxLayout(title_bar)
        title_label = QLabel("电路配盘接线操作分析系统")
        title_label.setObjectName("titleLabel")
        title_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold; letter-spacing: 2px;")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        title_layout.setContentsMargins(0, 12, 0, 12)

        # 新增：管理接线规则按钮
        self.btn_manage_rules = QPushButton("管理接线规则")
        self.btn_manage_rules.setStyleSheet("background: #38b6ff; color: white; border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: bold;")
        self.btn_manage_rules.clicked.connect(self.show_rule_manager)
        title_layout.addWidget(self.btn_manage_rules)
        title_layout.setSpacing(16)

        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(title_bar)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(30, 20, 30, 20)
        content_layout.setSpacing(24)
        main_layout.addLayout(content_layout)
        self.setCentralWidget(main_widget)

        # 左侧摄像头区
        left_panel = QVBoxLayout()
        left_panel.setSpacing(18)
        video_card = QGroupBox("摄像头实时画面")
        video_layout = QVBoxLayout(video_card)
        self.video_label = MarkableLabel("等待摄像头启动...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("background:#222;color:white;font-size:20px;border-radius:10px;")
        video_layout.addWidget(self.video_label)
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("启动摄像头")
        self.btn_start.clicked.connect(self.toggle_camera)
        self.btn_analyze = QPushButton("分析当前操作")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze_current)
        self.btn_describe = QPushButton("描述当前接线")
        self.btn_describe.setEnabled(False)
        self.btn_describe.clicked.connect(self.describe_current)
        self.btn_import_std = QPushButton("导入标准电路图")
        self.btn_import_std.clicked.connect(self.import_standard_image)
        self.btn_add_rule = QPushButton("新增接线规则")
        self.btn_add_rule.clicked.connect(self.add_rule)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_describe)
        btn_layout.addWidget(self.btn_import_std)
        btn_layout.addWidget(self.btn_add_rule)
        video_layout.addLayout(btn_layout)
        self.std_img_label = QLabel("未导入标准电路图")
        self.std_img_label.setAlignment(Qt.AlignCenter)
        self.std_img_label.setMinimumHeight(100)
        self.std_img_label.setStyleSheet("background:#eee;color:#888;font-size:16px;border-radius:8px;")
        video_layout.addWidget(self.std_img_label)
        left_panel.addWidget(video_card)
        left_panel.addStretch(1)

        # 右侧分析区
        right_panel = QVBoxLayout()
        right_panel.setSpacing(18)
        result_card = QGroupBox("分析结果与评分")
        result_layout = QVBoxLayout(result_card)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("分析结论将在此显示")
        self.result_text.setMinimumHeight(80)
        self.score_label = QLabel("得分：--")
        self.score_label.setObjectName("scoreLabel")
        self.comment_label = QLabel("评语：--")
        self.comment_label.setObjectName("commentLabel")
        result_layout.addWidget(self.result_text)
        result_layout.addWidget(self.score_label)
        result_layout.addWidget(self.comment_label)
        right_panel.addWidget(result_card)

        # 历史记录卡片
        history_card = QGroupBox("历史记录")
        history_layout = QVBoxLayout(history_card)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["时间", "操作描述", "得分", "是否正确"])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("QTableWidget {alternate-background-color: #f4f6fb;}")
        history_layout.addWidget(self.history_table)
        export_btn = QPushButton("导出历史记录")
        export_btn.clicked.connect(self.export_history)
        history_layout.addWidget(export_btn)
        right_panel.addWidget(history_card)
        right_panel.addStretch(1)

        content_layout.addLayout(left_panel, 2)
        content_layout.addLayout(right_panel, 3)

        # 状态栏
        self.statusBar().showMessage("系统就绪，等待启动摄像头...")

        # 菜单栏-模型设置
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("设置")
        model_action = QAction("模型服务器设置", self)
        model_action.triggered.connect(self.show_model_settings)
        settings_menu.addAction(model_action)

    def _init_variables(self):
        self.camera_thread = CameraThread()
        self.camera_thread.new_display_frame.connect(self._update_video_frame)
        self.camera_thread.capture_frame_signal.connect(self._on_capture_frame)
        self.model_server = "http://localhost:11434"
        self.model_name = "modelscope.cn/lmstudio-community/MiniCPM-o-2_6-gguf:latest"
        self.last_frame = None
        self.standard_image = None
        self.standard_image_data = None
        self.standard_struct = None
        self.auto_marks = []
        self.model_type = "local"
        self.openai_model_name = "gpt-3.5-turbo"
        self.openai_api_key = OPENAI_API_KEY
        # 新增：规则管理
        self.rules = []  # [{"name": str, "content": str or dict}]
        self.selected_rule_index = None

    def toggle_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.btn_start.setText("启动摄像头")
            self.btn_analyze.setEnabled(False)
            self.btn_describe.setEnabled(False)
            self.statusBar().showMessage("摄像头已停止")
        else:
            self.camera_thread.start()
            self.btn_start.setText("停止摄像头")
            self.btn_analyze.setEnabled(True)
            self.btn_describe.setEnabled(True)
            self.statusBar().showMessage("摄像头运行中...")

    def _update_video_frame(self, frame_data):
        self.last_frame = frame_data
        buffer = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            marks = []
            self.auto_marks = marks
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
            self.video_label.set_marks(marks)

    def analyze_current(self):
        if not self.camera_thread.isRunning() or self.last_frame is None:
            QMessageBox.warning(self, "提示", "请先启动摄像头")
            return
        rule = self.get_current_rule()
        if not rule:
            QMessageBox.warning(self, "提示", "请先新增并选择一个接线规则")
            return
        self.statusBar().showMessage("正在分析当前操作...")
        self.btn_analyze.setEnabled(False)
        self.camera_thread.capture_current_frame()

    def describe_current(self):
        if not self.camera_thread.isRunning() or self.last_frame is None:
            QMessageBox.warning(self, "提示", "请先启动摄像头")
            return
        self.statusBar().showMessage("正在描述当前接线...")
        self.btn_describe.setEnabled(False)
        self._describe_frame(self.last_frame)

    def _describe_frame(self, frame_data):
        self.describe_worker = DescribeWorker(
            frame_data, self.model_server, self.model_name
        )
        self.describe_worker.describe_complete.connect(self._on_describe_result)
        self.describe_worker.start()

    def _on_describe_result(self, result):
        self.btn_describe.setEnabled(True)
        QMessageBox.information(self, "接线描述", result.get("描述", "无描述"))
        self.statusBar().showMessage("描述完成")

    def import_standard_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入标准电路图", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            img = Image.open(path)
            self.standard_image = img
            with open(path, 'rb') as f:
                self.standard_image_data = f.read()
            qimg = QImage(path)
            pixmap = QPixmap.fromImage(qimg)
            self.std_img_label.setPixmap(pixmap.scaled(self.std_img_label.width(), self.std_img_label.height(), Qt.KeepAspectRatio))
            self.std_img_label.setText("")
            self.statusBar().showMessage(f"已导入标准电路图：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导入错误", str(e))

    def add_rule(self):
        path, _ = QFileDialog.getOpenFileName(self, "新增接线规则", "", "所有支持文件 (*.json *.txt *.pdf *.doc *.docx);;JSON文件 (*.json);;文本文件 (*.txt);;PDF文件 (*.pdf);;Word文档 (*.doc *.docx)")
        if not path:
            return
        try:
            ext = path.split('.')[-1].lower()
            rule_name = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
            content = None
            if ext == 'json':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    content = json.loads(content)
                except Exception:
                    pass
            elif ext in ['txt']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext in ['pdf']:
                self.statusBar().showMessage("正在解析PDF，请稍候...")
                self.pdf_worker = PDFParseWorker(path)
                self.pdf_worker.parse_complete.connect(lambda text, error, name=rule_name: self._on_add_rule_pdf_parsed(text, error, name))
                self.pdf_worker.start()
                return
            elif ext in ['doc', 'docx']:
                try:
                    import docx
                    doc = docx.Document(path)
                    content = '\n'.join([p.text for p in doc.paragraphs])
                except Exception as e:
                    QMessageBox.critical(self, "导入错误", f"Word读取失败: {e}")
                    return
            else:
                with open(path, 'rb') as f:
                    content = f.read()
            self.rules.append({"name": rule_name, "content": content})
            if self.selected_rule_index is None:
                self.selected_rule_index = 0
            self.statusBar().showMessage(f"已新增接线规则：{rule_name}")
        except Exception as e:
            QMessageBox.critical(self, "导入错误", str(e))

    def _on_add_rule_pdf_parsed(self, text, error, rule_name):
        if error:
            QMessageBox.critical(self, "导入错误", f"PDF读取失败: {error}")
        else:
            self.rules.append({"name": rule_name, "content": text})
            if self.selected_rule_index is None:
                self.selected_rule_index = 0
            self.statusBar().showMessage(f"PDF解析完成，已新增接线规则：{rule_name}")

    def _on_capture_frame(self, frame):
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            rule = self.get_current_rule()
            rules_text = rule["content"] if rule else ""
            if not isinstance(rules_text, str):
                rules_text = json.dumps(rules_text, ensure_ascii=False)
            prompt = (
                "你是一名电气工程教学专家。请分析当前画面中学生的电路接线操作，描述学生连接了哪些区域的哪些接线端。"
                "请严格按照以下接线规则和评分标准进行评价和打分：\n" + rules_text +
                "\n如果学生操作不符合任意一条规则，请指出错误，并给出正确接法建议。"
                "请以如下JSON格式返回：{\"操作描述\": \"\", \"得分\": 90, \"是否正确\": true, \"错误提示\": \"\", \"正确接法\": \"\"}"
            )
            if self.model_type == "openai":
                self.analysis_worker = OpenAIGPTWorker(prompt, self.openai_model_name, self.openai_api_key)
            else:
                self.analysis_worker = AnalysisWorker(
                    buffer.tobytes(), self.model_server, self.model_name,
                    prompt=prompt
                )
            self.analysis_worker.analysis_complete.connect(self._on_analysis_result)
            self.analysis_worker.start()

    def _on_analysis_result(self, result):
        self.btn_analyze.setEnabled(True)
        # 记录操作历史
        if not hasattr(self, 'operation_history'):
            self.operation_history = []
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.operation_history.append({
            "时间": now,
            "操作描述": result.get("操作描述", ""),
            "得分": result.get("得分", ""),
            "是否正确": result.get("是否正确", False),
            "错误提示": result.get("错误提示", ""),
            "正确接法": result.get("正确接法", "")
        })
        # 展示结构化内容
        parts = []
        parts.append("<b>操作描述：</b><br>" + result.get("操作描述", "无描述"))
        parts.append("<b>得分：</b>" + str(result.get("得分", "--")))
        parts.append("<b>是否正确：</b>" + ("正确" if result.get("是否正确", False) else "错误"))
        self.result_text.setHtml('<br><br>'.join(parts))
        self.score_label.setText(f"得分：{result.get('得分', '--')}")
        self.comment_label.setText(f"评语：{result.get('错误提示', '--')}")
        # 错误弹窗和正确接法
        if not result.get("是否正确", False):
            msg = QMessageBox(self)
            msg.setWindowTitle("接线错误提示")
            msg.setText(result.get("错误提示", "操作不符合规则"))
            if result.get("正确接法", ""):
                btn = msg.addButton("查看正确接法", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)
            def show_correct():
                QMessageBox.information(self, "正确接法", result.get("正确接法", "无"))
            if result.get("正确接法", ""):
                btn.clicked.connect(show_correct)
            msg.exec_()
        self.statusBar().showMessage("分析完成")
        # 历史表格同步
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(now))
        self.history_table.setItem(row, 1, QTableWidgetItem(result.get("操作描述", "")))
        self.history_table.setItem(row, 2, QTableWidgetItem(str(result.get("得分", ""))))
        self.history_table.setItem(row, 3, QTableWidgetItem("正确" if result.get("是否正确", False) else "错误"))

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出历史记录", "history.csv", "CSV文件 (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("时间,操作描述,得分,是否正确\n")
                for record in self.operation_history:
                    f.write(f"{record['时间']},{record['操作描述']},{record['得分']},{'正确' if record['是否正确'] else '错误'}\n")
            self.statusBar().showMessage(f"历史记录已导出到 {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", str(e))

    def show_model_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("模型服务器设置")
        tabs = QTabWidget(dialog)
        # 本地/远程模型tab
        local_tab = QWidget()
        local_layout = QFormLayout(local_tab)
        server_input = QLineEdit(local_tab)
        server_input.setText(self.model_server)
        model_input = QLineEdit(local_tab)
        model_input.setText(self.model_name)
        local_layout.addRow("服务器地址：", server_input)
        local_layout.addRow("模型名称：", model_input)
        tabs.addTab(local_tab, "本地/远程模型")
        # OpenAI模型tab
        openai_tab = QWidget()
        openai_layout = QFormLayout(openai_tab)
        openai_model_input = QLineEdit(openai_tab)
        openai_model_input.setText(self.openai_model_name)
        openai_key_input = QLineEdit(openai_tab)
        openai_key_input.setText(self.openai_api_key)
        openai_key_input.setEchoMode(QLineEdit.Password)
        openai_layout.addRow("OpenAI模型名称：", openai_model_input)
        openai_layout.addRow("OpenAI API Key：", openai_key_input)
        tabs.addTab(openai_tab, "OpenAI模型")
        # 主布局
        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(tabs)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        main_layout.addWidget(btn_box)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        # 默认选中当前模型类型
        if self.model_type == "openai":
            tabs.setCurrentIndex(1)
        else:
            tabs.setCurrentIndex(0)
        if dialog.exec_() == QDialog.Accepted:
            if tabs.currentIndex() == 0:
                self.model_type = "local"
                self.model_server = server_input.text()
                self.model_name = model_input.text()
            else:
                self.model_type = "openai"
                self.openai_model_name = openai_model_input.text().strip()
                self.openai_api_key = openai_key_input.text().strip()
            self.statusBar().showMessage("模型服务器设置已更新")

    def closeEvent(self, event):
        self.camera_thread.stop()
        cv2.destroyAllWindows()
        event.accept()

    def detect_components(self, frame):
        return []

    def export_auto_marks(self):
        pass

    def show_rule_manager(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("接线规则管理")
        layout = QVBoxLayout(dialog)
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QHBoxLayout
        rule_list = QListWidget()
        rule_list.setSelectionMode(QAbstractItemView.SingleSelection)
        for idx, rule in enumerate(self.rules):
            item = QListWidgetItem(rule["name"])
            rule_list.addItem(item)
            if idx == self.selected_rule_index:
                item.setSelected(True)
        layout.addWidget(rule_list)
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("新增规则")
        btn_add.clicked.connect(self.add_rule)
        btn_del = QPushButton("删除选中规则")
        def delete_rule():
            row = rule_list.currentRow()
            if row >= 0:
                del self.rules[row]
                if self.selected_rule_index == row:
                    self.selected_rule_index = 0 if self.rules else None
                elif self.selected_rule_index and self.selected_rule_index > row:
                    self.selected_rule_index -= 1
                rule_list.takeItem(row)
        btn_del.clicked.connect(delete_rule)
        btn_select = QPushButton("设为当前规则")
        def select_rule():
            row = rule_list.currentRow()
            if row >= 0:
                self.selected_rule_index = row
                dialog.accept()
        btn_select.clicked.connect(select_rule)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_del)
        btn_layout.addWidget(btn_select)
        layout.addLayout(btn_layout)
        dialog.exec_()

    def get_current_rule(self):
        if self.selected_rule_index is not None and 0 <= self.selected_rule_index < len(self.rules):
            return self.rules[self.selected_rule_index]
        return None

class DescribeWorker(QThread):
    describe_complete = pyqtSignal(dict)

    def __init__(self, frame_data, model_server, model_name):
        super().__init__()
        self.frame_data = frame_data
        self.model_server = model_server
        self.model_name = model_name

    def enhance_image(self, img):
        # 自动白平衡
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # 对比度增强
        alpha = 1.2
        beta = 10
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # 锐化
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    def run(self):
        try:
            buffer = np.frombuffer(self.frame_data, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (224, 224))
            frame = self.enhance_image(frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with io.BytesIO() as buf:
                img.save(buf, format='JPEG', quality=70)
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            prompt = (
                "你是一名电气工程教学专家。请详细描述当前画面中学生的电路接线情况，指出各个元件的连接关系。请用简明中文描述。"
            )
            api_url = f"{self.model_server}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [img_b64],
                "options": {"temperature": 0.2, "num_predict": 512}
            }
            resp = requests.post(api_url, json=data, timeout=60)
            if resp.status_code != 200:
                raise Exception(f"模型服务错误: {resp.text}")
            result = resp.json().get('response', '')
            self.describe_complete.emit({"描述": result})
        except Exception as e:
            self.describe_complete.emit({"描述": f"描述失败: {str(e)}"})

if __name__ == '__main__':
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
    app.setPalette(palette)
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())