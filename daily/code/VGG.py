import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16


# 定义VGG网络结构，这里使用VGG16作为示例
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        original_vgg = vgg16(weights='IMAGENET1K_V1')
        # 替换最后的全连接层以适应MNIST的10个类别
        self.features = nn.Sequential(*list(original_vgg.features.children()))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # 调整尺寸以匹配全连接层的期望输入
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 实例化模型
model = VGG(num_classes=10)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# MNIST数据集的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练和测试数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10):
    for epoch in range(epochs):
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

        print(f'Epoch {epoch + 1} - Training loss: {running_loss / len(train_loader)}')

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1} - Test accuracy: {100 * correct / total}%')


# 调用训练函数
train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10)