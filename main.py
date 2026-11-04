import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_main import Ui_MainWindow
from canvas import Canvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 创建画板并放置在 centralwidget 上（覆盖 UI 中预留的区域）
        self.canvas = Canvas(self)
        self.canvas.setGeometry(20, 10, 280, 280)
        self.canvas.setParent(self.ui.centralwidget)

        # 连接清空按钮
        self.ui.clearBtn.clicked.connect(self.clear_canvas)

        # 初始化显示
        self.ui.lcdNumber.display(0)

    def clear_canvas(self):
        self.canvas.clear()
        self.ui.lcdNumber.display(0)

    def update_result(self, digit):
        """画板调用此方法来更新 LCD 显示"""
        self.ui.lcdNumber.display(digit)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
