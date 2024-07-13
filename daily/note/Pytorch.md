# Pytorch

## dataset

## Tensorboard

一个可视化工具

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
```

先导入summarywriter

导入Image

```python
writer = SummaryWriter("logs")
image_path = "dataset/train/ants/5650366_e22b7e1065.jpg"  # 设置图片路径
img_PIL = Image.open(image_path)  #
img_array = np.array(img_PIL)  # 转为RGB存储每一个图像像素点成为numpy数组
print(img_array.shape)  # 发现是HWC，高宽通

writer.add_image("text", img_array, 1, dataformats='HWC')  # 转为HWC
writer.close()
```

在本地终端输入

```shell
tensorboard --logdir=logs
```

在端口查看图表

## transform

### Image.open()

PIL

### ToTensor()

tensor

### cv.imread()

np.arrays 格式HWC高宽通

```python
from torchvision import transforms
from PIL import Image
```

导入

```python
img_path = "dataset/train/bees/16838648_415acd9e3f.jpg" # 设置路径
img = Image.open(img_path)
print(img)
```

创建PIL对象，也就是图片对象

```python
tensor_trans = transforms.ToTensor()  # 创建对象，一个转化的工具
tensor_img = tensor_trans(img) # 把PIL类型变成张量img
# print(tensor_img)  # 转为张量tensor数组
print(tensor_img.shape)  # WHC,宽高通
img.show()
```

## 深度学习

## 创建网络结构

```python
class asd(nn.Module):
    def __init__(self):  # 创建结构，构造网络
        super(asd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)  # 卷积层，彩色图像通道数3

    def forward(self, x):  # 网络运行逻辑
        x = self.conv1(x)  # 进入第一层
        return x
```

测试

```python
dataset = torchvision.datasets.CIFAR10(root='./torch_dataset', train=False, download=True,transform=torchvision.transforms.ToTensor())#测试集

```
调用网络结构，生成一个网络
```
test = asd()
print(test)
```

### 反向传播

在深度学习中，链接权重相当于函数的变化率，通过求梯度找到变化率也就是参数，使得损失函数最小，从后往前依照链式法则求偏导，找到参数的最小值

将结果与输入的中间所有的导数相乘，得到最终结果y关于输入值x的偏导，让偏导

```python
import torch
x=torch.ones(2,2,requires_grad=True)#先设置requires_grad，让每一步操作都被记录，来追踪其计算历史
y=x+2
z=y*y*3
out=z.mean()
out.backward()#反向传播
x.grad#查看关于x的梯度值
```

对于x，操作时如果使用的是x.data，则不会记录操作，因为x.data是记录tensor里的数据值

#### 反向传播的线性回归

```python
import torch
x = torch.rand([500, 1])
y = 3 * x + 0.8#相当于训练集的y
alpha=0.05#学习率
```

设置参数w和偏置量b

```python
w = torch.rand([1, 1], requires_grad=True)
b = torch.rand(1, requires_grad=True)
```

循环计算loss，并且更新梯度，计算新一轮损失loss

```python
for i in range(1500):
    #计算y_predit
    y_predit = torch.matmul(x, w) + b#matmul是内积
    #计算loss
    loss=(y-y_predit ).pow(2).mean()
    if w.grad is not None:
        w.data.zero_()#下划线表示就地修改，把w的梯度改成0，避免重复累加
    if b.grad is not None:
        b.data.zero_()#下划线表示就地修改，把b的梯度改成0，避免重复累加
    loss.backward()
    w.data=w.data-alpha*w.grad.data#新权重=旧权重-学习率*权重梯度
    b.data=b.data-alpha*b.grad.data
    print("w,b,loss",w.item(),b.item(),loss.data)
```

#### 用pytorch的api实现

#### 用gpu进行操作

```python
device1=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda is available?",torch.cuda.is_available())
#把训练集预测集改成GPU类的tensor
x = torch.rand([500, 1]).to(device1)
#把模型改成GPU类的tensor，这个过程会自动更改模型里的参数
model = Lr().to(device1)
```

## 卷积

卷积核煎饼移动，然后算出来填进去

进行卷积，绘图

```python
writer = SummaryWriter('test')
step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)#利用网络产生输出
    writer.add_images('imput', imgs, global_step=step)#imput
    output = torch.reshape(output, (-1,3,30,30))#匹配结构
    writer.add_images('output', output, global_step=step)#output
    step = step + 1
writer.close()
```

### 卷积核

卷积核有3x3 5x5，是个矩阵，在深度学习中**卷积核内的值**是一个**需要计算**的参数，一开始是随机分配的后来在**反向传播**中固定下来，特别的1x1卷积核有重要意义

卷积核的选取决定了每一个卷积核的**感受野的大小**，**有多少个卷积核**就会产生**多少个输出**，每一个输出都是对应的图像关于这个**卷积核操作后的结果**

## 池化操作

**降低**特征图**的空间维度**，从而减少参数数量和**计算复杂度**，同时**保留**重要的**特征信息**

### 最大池化

最大池化通过在输入特征图上滑动一个固定大小的池化窗口（pooling kernel），在窗口覆盖的区域内选择最大的元素值作为输出

# BP神经网络

1. **初始化**：随机初始化网络的权重和偏置
2. **前向传播**：将输入数据通过网络进行前向传播，得到预测输出
3. **计算损失**：使用损失函数计算预测输出与真实标签之间的差异
4. **反向传播**：根据损失函数计算梯度，使用链式法则从输出层反向传播到输入层
5. **参数更新**：根据计算得到的梯度更新网络的权重和偏置
6. **迭代优化**：重复上述步骤，直到网络在验证集上的性能不再提升或达到预定的迭代次数

导入库，准备数据

```python
from torch import nn
from torch import optim
import torch
#准备数据
x = torch.rand([500, 1])
y_true = x * 3 + 0.8
```

搭建神经网络

```python
class Lr(nn.Module):
    def __init__(self):#网络结构
        super(Lr, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  #创建一个全连接层，输入是1输出也是1
#前向传播：将输入数据通过网络进行前向传播，得到预测输出
    def forward(self, x):#运行逻辑
        out = self.linear(x)
        return out
```

实例化模型

```python
#实例化模型
model = Lr()
#实例化优化器
optimizer = optim.SGD(model.parameters(),lr=0.01)#可以调整学习率
#实例化损失函数
loss_fn = nn.MSELoss()
```

**迭代优化**：


```python
for i in range(5000):
    #得到预测值
    y_predict=model(x)
    #根据损失函数工具loss_fn算出损失函数loss
    #计算损失：使用损失函数计算预测输出与真实标签之间的差异
    loss=loss_fn(y_predict,y_true)
    #梯度置为零
    optimizer.zero_grad()
    #反向传播：根据损失函数计算梯度，使用链式法则从输出层反向传播到输入层
    loss.backward()
    #参数更新：根据计算得到的梯度更新网络的权重和偏置
    optimizer.step()
    if i % 200 == 0:
        params=list(model.parameters())
        print(loss.item(),params[0].item(),params[1].item())
```

# FashionMnist目标检测实战

## 导入库

主要是torch，有一部分os和numpy

```python
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import torch
import os
```

## 收集处理数据

```python
#mnist数据集是灰度图，分别是均值和标准差，操作后具有均值为0，标准差为1的特性
mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,), (0.353))])
transform_to_tensor = transforms.ToTensor()
#加载数据集
data_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transforms)
data_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=mnist_transforms)
#加载dataloader
dataloader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True, drop_last=True)  #28x28
dataloader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True, drop_last=True)
#FashionMnist由图片和类别索引构成
```

### Compose

```python
mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,), (0.353))])
transform_to_tensor = transforms.ToTensor()
```

结合了ToTensor和Normalize，先转化图片为tensor然后再进行归一化操作，操作后具有均值为0，标准差为1的特性，0.286是通道的均值，0.353是标准差

### DataLoader

```python
from torch.utils.data import DataLoader
```

batch_size表示一批有多少处理对象

shuffle表示抽取时是否打乱顺序

drop_last表示是否忽略余数

### GPU的调用

检查是否可以使用GPU，不行的话device选择cpu

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda is available?",torch.cuda.is_available())
```

## 构建网络结构

#### 定义网络结构

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)  #灰度图像，通道数为1，有32个输出通道，也就是有32个卷积核产生图像，输出26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  #第二层卷积，通道数为32，输出64，输出完变成
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(13 * 13 * 64, 10)#返回10个数值
```

第一层为卷积层，第二层卷积层，之后进行一次Relu激活，第三层池化层，第四层为全连接层，返回10个数值，对应10种衣服

#### 定义前向传播

```python
def forward(self, x):
    x = self.conv1(x)#调用池化层1
    x = self.conv2(x)#调用池化层2
    x = self.relu(x)#Relu激活
    x = self.max_pool(x)#最大池化
    x = x.view(-1, 13 * 13 * 64)#重排列数据
    x = self.fc1(x)#全链接
    #x = F.softmax(x, dim=1)# 原本需要softmax的，但是在后面的loss计算中使用了F.cross_entropy，自动计算了softmax，所以直接return x
    return x
```

`x = x.view(-1, 13 * 13 * 64)`:

- 将池化层的输出`x`转换成一个一维数组，然后将其重新排列（reshape）成一个二维数组，其中第二维的大小是`13 * 13 * 64`。这里的`13 * 13`是池化后的特征图尺寸，`64`是通道数。`-1`表示第一维的大小由第二维的大小推导得出。

## 训练函数

```python
def train(train_best_loss):
    dataloader = dataloader_train
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, target=labels)
        loss.backward()
        optimizer.step()
        if loss < train_best_loss:
            train_best_loss = loss
            print("idx:", idx, "current best loss:", train_best_loss.item())
            torch.save(model.state_dict(), "./FashionModel/model.pth")
            torch.save(optimizer.state_dict(), "./FashionModel/optimizer.pth")
    return train_best_loss
```

## 测试函数

```python
def test():
    model.eval()  #评估模式
    data_loader = dataloader_test
    acc_list = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在 GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # 计算准确率
            # output[batch_size,10] target[batch_size]
            # 输出每一行都有十个数，代表对应0-9的概率，找出最大的概率代表着模型识别出来的值
            pred = torch.max(outputs, dim=1)[1]
            pred = predicted.cpu()
            cur_acc = pred.eq(labels.cpu()).float().mean().item()
            acc_list.append(cur_acc)
    print("平均准确率:", np.mean(acc_list))
```

## 调用

```python
best_loss = float('inf')
for i in range(1):
    best_loss=train(train_best_loss=best_loss)
test()
```

# RNN

循环神经网络

1. 可以处理序列数据
2. 每个神经元在时间步上循环连接，输出会在下一轮作为输入传入神经元

## GRU

是**LSTM**的一个变体，把LSTM中的**遗忘门和输入门**合成一个**更新门**，合并**细胞状态**和**隐藏状态**，**简化了结构**

具有**更新门**，**输出门**

## LSTM

细胞状态就是上面的线，类似长期记忆

隐藏状态是下面的线，更容易受到短期影响，长期的部分会梯度爆炸或者消失

## LSTM和GRU

### 2.1 LSTM的基础介绍

假如现在有这样一个需求，根据现有文本预测下一个词语，比如`天上的云朵漂浮在__`，通过间隔不远的位置就可以预测出来词语是`天上`，但是对于其他一些句子，可能需要被预测的词语在前100个词语之前，那么此时由于间隔非常大，随着间隔的增加可能会导致真实的预测值对结果的影响变的非常小，而无法非常好的进行预测（RNN中的长期依赖问题（long-Term Dependencies））

那么为了解决这个问题需要**LSTM**（**Long Short-Term Memory网络**）

LSTM是一种RNN特殊的类型，可以学习长期依赖信息。在很多问题上，LSTM都取得相当巨大的成功，并得到了广泛的应用。

一个LSMT的单元就是下图中的一个绿色方框中的内容：

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053747.jpg)



其中$\sigma$表示sigmod函数，其他符号的含义：

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053886.jpg)



### 2.2 LSTM的核心

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053056.png)

LSTM的核心在于单元（细胞）中的状态，也就是上图中最上面的那根线。

但是如果只有上面那一条线，那么没有办法实现信息的增加或者删除，所以在LSTM是通过一个叫做`门`的结构实现，门可以选择让信息通过或者不通过。

这个门主要是通过sigmoid和点乘（`pointwise multiplication`）实现的

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053241.png)



我们都知道，$sigmoid$的取值范围是在(0,1)之间，如果接近0表示不让任何信息通过，如果接近1表示所有的信息都会通过



### 2.3 逐步理解LSTM

#### 2.3.1 遗忘门

遗忘门通过sigmoid函数来决定哪些信息会被遗忘

在下图就是$h_{t-1}和x_t$进行合并（concat）之后乘上权重和偏置，通过sigmoid函数，输出0-1之间的一个值，这个值会和前一次的细胞状态($C_{t-1}$)进行点乘，从而决定遗忘或者保留

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053436.png)



#### 2.3.2 输入门

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053581.png)

下一步就是决定哪些新的信息会被保留，这个过程有两步：

1. 一个被称为`输入门`的sigmoid 层决定哪些信息会被更新
2. `tanh`会创造一个新的候选向量$\widetilde{C}_{t}$，后续可能会被添加到细胞状态中

例如：

`我昨天吃了苹果，今天我想吃菠萝`，在这个句子中，通过遗忘门可以遗忘`苹果`,同时更新新的主语为`菠萝`



现在就可以更新旧的细胞状态$C_{t-1}$为新的$C_{ t }$ 了。

更新的构成很简单就是：

1. 旧的细胞状态和遗忘门的结果相乘
2. 然后加上 输入门和tanh相乘的结果

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053165.png)



#### 2.3.3 输出门

最后，我们需要决定什么信息会被输出，也是一样这个输出经过变换之后会通过sigmoid函数的结果来决定那些细胞状态会被输出。

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053472.png)

步骤如下：

1. 前一次的输出和当前时间步的输入的组合结果通过sigmoid函数进行处理得到$O_t$
2. 更新后的细胞状态$C_t$会经过tanh层的处理，把数据转化到(-1,1)的区间
3. tanh处理后的结果和$O_t$进行相乘，把结果输出同时传到下一个LSTM的单元



### 2.4 GRU，LSTM的变形

GRU(Gated Recurrent Unit),是一种LSTM的变形版本， 它将遗忘和输入门组合成一个“更新门”。它还合并了单元状态和隐藏状态，并进行了一些其他更改，由于他的模型比标准LSTM模型简单，所以越来越受欢迎。

![](https://linsixi-1327856365.cos.ap-guangzhou.myqcloud.com/imgs/blog202407122053433.png)

