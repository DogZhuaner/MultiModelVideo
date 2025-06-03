import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QStatusBar, QTabWidget, QAction, QLineEdit
)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QCoreApplication


class MarkableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        title_bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4361ee, stop:1 #38b6ff); border-radius: 0 0 18px 18px;")
        title_layout = QHBoxLayout(title_bar)
        title_label = QLabel("电路配盘接线操作分析系统")
        title_label.setObjectName("titleLabel")
        title_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold; letter-spacing: 2px;")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        title_layout.setContentsMargins(0, 12, 0, 12)

        # 管理接线规则按钮
        self.btn_manage_rules = QPushButton("管理接线规则")
        self.btn_manage_rules.setStyleSheet(
            "background: #38b6ff; color: white; border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: bold;")
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
        self.btn_analyze = QPushButton("分析当前操作")
        self.btn_analyze.setEnabled(False)
        self.btn_describe = QPushButton("描述当前接线")
        self.btn_describe.setEnabled(False)
        self.btn_import_std = QPushButton("导入标准电路图")
        self.btn_add_rule = QPushButton("新增接线规则")
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
        history_layout.addWidget(export_btn)
        right_panel.addWidget(history_card)
        right_panel.addStretch(1)

        content_layout.addLayout(left_panel, 2)
        content_layout.addLayout(right_panel, 3)

        # 状态栏
        self.statusBar().showMessage("系统就绪")

        # 菜单栏-模型设置
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("设置")
        model_action = QAction("模型服务器设置", self)
        settings_menu.addAction(model_action)

    def _init_variables(self):
        """初始化变量 - 只保留UI相关变量"""
        # 只保留基本UI状态变量
        self.statusBar().showMessage("系统就绪")


def launch_main_window():
    """启动主窗口的接口函数"""
    app = QApplication(sys.argv)
    # 设置高DPI支持
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    # 设置应用程序样式
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
    app.setPalette(palette)
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == '__main__':
    launch_main_window()