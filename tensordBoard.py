import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集，CIFAR10 数据集是PIL Image，要转换为tensor数据类型
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 看一下训练数据集和测试数据集都有多少张（如何获得数据集的长度）
train_data_size = len(train_data)  # length 长度
test_data_size = len(test_data)
# 如果train_data_size=10，那么打印出的字符串为：训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))  # 字符串格式化，把format中的变量替换{}
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()  # 分类问题可以用交叉熵

# 定义优化器
learning_rate = 0.01  # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # SGD 随机梯度下降

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练轮数

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i + 1))  # i从0-9
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()  # 对参数进行优化
        total_train_step += 1
        if total_train_step % 100 == 0:  # 逢百才打印记录
            print("训练次数：{},loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():  # 无梯度，不进行调优
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