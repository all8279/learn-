import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集，CIFAR10转换为tensor数据类型
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 把网络放到序列中
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),  # 展平
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

tudui = Tudui()
# 创建损失函数 交叉熵
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # SGD 随机梯度下降

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10  # 训练轮数

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i + 1))  # i从0-9
    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()  # 要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()  # 对参数进行优化
        total_train_step += 1
        if total_train_step % 100 == 0:  # 逢百才打印记录
            print("训练次数：{},loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
    print("整体测试集上的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

writer.close()