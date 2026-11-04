import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from PyQt5.QtWidgets import QWidget, QLCDNumber
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QRect
from preprocess import preprocess_image
from model import Net
from paths import MODEL_PATH


class Canvas(QWidget):
    def __init__(self, parent=None):
        super(Canvas, self).__init__(parent)
        self.setMouseTracking(False)
        self.setMinimumSize(280, 280)
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        self.pos_xy = []
        self.model = None
        print("画板初始化完成，尚未加载模型")

    def get_model(self):
        if self.model is None:
            print("首次识别，加载模型中...")
            self.model = Net()
            state_dict = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.model.eval()
            print("模型加载成功")
        return self.model

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        if len(self.pos_xy) > 1:
            start_point = self.pos_xy[0]
            for p in self.pos_xy:
                end_point = p
                if end_point == (-1, -1):
                    start_point = (-1, -1)
                    continue
                if start_point == (-1, -1):
                    start_point = end_point
                    continue
                painter.drawLine(start_point[0], start_point[1], end_point[0], end_point[1])
                start_point = end_point
        painter.end()

    def mouseMoveEvent(self, event):
        pos = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos)
        self.update()

    def mouseReleaseEvent(self, event):
        self.pos_xy.append((-1, -1))
        self.update()

        # 截图画板
        rect = QRect(0, 0, self.width(), self.height())
        pixmap = self.grab(rect)
        pixmap.save("tmp.png")

        # 预处理 + 推理
        img_array = preprocess_image("tmp.png")
        tensor = torch.from_numpy(img_array)
        model = self.get_model()
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()
        print("识别结果:", pred)

        # 直接查找并更新 LCD 控件（不依赖父窗口的方法）
        main_window = self.window()
        if main_window:
            lcd = main_window.findChild(QLCDNumber, "lcdNumber")
            if lcd is not None:
                lcd.display(pred)
                print(f"LCD 已更新为 {pred}")
            else:
                print("错误：未找到 lcdNumber 控件")
        else:
            print("错误：无法获取顶层窗口")

    def clear(self):
        self.pos_xy = []
        self.update()