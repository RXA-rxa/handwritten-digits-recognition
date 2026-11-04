import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time

# -------------------- 数据预处理 --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 抽样：训练集 7000 张，测试集只取 1000 张（加快测试速度）
train_indices = torch.randperm(len(trainset))[:7000]
test_indices = torch.randperm(len(testset))[:1000]   # 从 3000 减到 1000
sampled_train = Subset(trainset, train_indices)
sampled_test = Subset(testset, test_indices)

train_loader = DataLoader(sampled_train, batch_size=128, shuffle=True)
test_loader = DataLoader(sampled_test, batch_size=128, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------- 模型定义 --------------------
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------- 训练函数 --------------------
def main(epochs=10, test_interval=2):   # test_interval: 每几个 epoch 测试一次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device == "cpu":
        print("提示：当前使用 CPU，训练会较慢。建议使用 GPU 环境（如 Colab）。")

    model = CIFAR10CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []
    epochs_list = []

    start_time = time.time()

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 每 test_interval 个 epoch 测试一次
        if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(labels.numpy())
            acc = accuracy_score(all_labels, all_preds)
            test_accuracies.append(acc)
            epochs_list.append(epoch + 1)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n总训练时间: {elapsed:.2f} 秒")

    # ---------- 最终详细评估（使用全部测试集，但这里已经是 1000 张）----------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\n最终测试准确率: {final_acc:.4f}")

    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # 绘制损失曲线（测试准确率曲线可能点较少，但也可画）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    if epochs_list:
        plt.subplot(1, 2, 2)
        plt.plot(epochs_list, test_accuracies, marker='o', color='orange')
        plt.title('Test Accuracy (every {} epochs)'.format(test_interval))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

    # 显示 5 张示例图像（不会增加训练时间）
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:5], labels[:5]
    def denormalize(img_tensor):
        return (img_tensor * 0.5 + 0.5).clamp(0, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        plt.imshow(img)
        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

    return final_acc

if __name__ == "__main__":
    # 参数：epochs=10, 每2个epoch测试一次
    acc = main(epochs=10, test_interval=2)