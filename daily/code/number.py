import torch
from torch import optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np

# 准备数据
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])  # 操作
mnist_train = MNIST(root='./data', train=True, download=False, transform=mnist_transform)  # 转成CHW的tensor
mnist_test = MNIST(root='./data', train=False, download=False, transform=mnist_transform)
train_dataloader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=mnist_test, batch_size=64, shuffle=True, drop_last=True)

acc_list = []
loss_list = []


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 根据形状变形
        x = self.fc1(x)  # [batch_size,28]
        x = F.relu(x)
        out = self.fc2(x)  # [batch_size,10]
        return F.log_softmax(out, dim=-1)


mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)  # .parameters是获取模型中参数的函数
if os.path.exists("./model/model.pth"):
    mnist_net.load_state_dict(torch.load('./model/model.pth'))  # 加载保存的模型
    optimizer.load_state_dict(torch.load('./model/optimizer.pth'))  # 加载保存的优化器，主要是学习率
    print("load model successfully")


def train(epoch):
    data_loader = train_dataloader
    for idx, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = mnist_net(inputs)  # 调用模型，得到预测值
        loss = F.cross_entropy(outputs, target=labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        if idx % 50 == 0:
            print('Epoch:', idx, 'Loss:', loss.item())

        if idx % 100 == 0:
            torch.save(mnist_net.state_dict(), "./model/model.pth")
            torch.save(optimizer.state_dict(), "./model/optimizer.pth")


def test():
    data_loader = test_dataloader
    for idx, (inputs, labels) in enumerate(data_loader):
        with torch.no_grad():
            outputs = mnist_net(inputs)
            loss = F.cross_entropy(outputs, target=labels)
            loss_list.append(loss)
            # 计算准确率
            # output[batch_size,10] target[batch_size]
            # 输出每一行都有十个数，代表对应0-9的概率，找出最大的概率代表着模型识别出来的值
            pred = torch.max(outputs, dim=-1)[1]
            cur_acc = pred.eq(labels).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率:", np.mean(acc_list), "平均损失:", np.mean(loss_list))


if __name__ == '__main__':
    for i in range(1):
        train(epoch=i)
        test()
