# 手写数字识别

基于 PyTorch 训练的 CNN 手写数字识别模型，配了个 PyQt5 手写画板界面，可以在窗口里写数字，实时识别。

## 技术栈

- PyTorch（CNN 模型）
- MNIST 数据集
- PyQt5（图形界面）

## 实现路径

```
# 装依赖
pip install torch torchvision pyqt5

# 训练模型（会自动下载 MNIST 数据集，跑完模型存在 model/params.pth）
python train.py

# 启动手写界面
python main.py
```

界面打开后在画板上写数字，右边 LCD 会显示识别结果。

## 模型结构

两层卷积 + 全连接：
- Conv2d(1, 32) → MaxPool
- Conv2d(32, 64) → MaxPool
- Linear(64*7*7, 128) → Linear(128, 10)

训练 5 个 epoch，测试集准确率大概 98% 左右。
