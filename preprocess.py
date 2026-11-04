import numpy as np
from PIL import Image


def preprocess_image(image_path):
    # 读取灰度图
    img = Image.open(image_path).convert('L')
    # 反转颜色：白底黑字 -> 黑底白字
    img_array = 255 - np.array(img)

    # 找到数字的边界（非零像素）
    coords = np.argwhere(img_array > 50)  # 阈值50，忽略轻微噪点
    if coords.size == 0:
        # 没有数字，返回全黑图像
        img_small = np.zeros((28, 28), dtype=np.uint8)
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # 裁剪出数字区域
        cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
        # 缩放到 20x20
        cropped_img = Image.fromarray(cropped)
        cropped_img = cropped_img.resize((20, 20), Image.Resampling.LANCZOS)
        cropped_array = np.array(cropped_img)
        # 放到 28x28 中央
        result = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - 20) // 2
        x_offset = (28 - 20) // 2
        result[y_offset:y_offset + 20, x_offset:x_offset + 20] = cropped_array
        img_small = result

    # 归一化到 [0,1]
    img_small = img_small / 255.0
    # 增加 batch 和 channel 维度
    img_small = img_small.reshape(1, 1, 28, 28).astype(np.float32)
    return img_small