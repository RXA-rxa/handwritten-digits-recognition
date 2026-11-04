import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net
from paths import DATA_DIR, MODEL_PATH   # 导入绝对路径

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 直接用绝对路径
    train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        print(f"Test Acc: {correct/len(test_dataset):.4f}")

    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"模型保存至 {MODEL_PATH}")

if __name__ == '__main__':
    train()