import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from mycnn import MyCNN
from load_data import test_dataset, train_loader

# 如果有GPU就用GPU否则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 64
# 定义反向传播学习率
LR = 0.001
# 实例化卷积类
net = MyCNN().to(device)
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数使用 Adam 自适应优化算法
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
)
# 批次数
epoch = 1
if __name__ == '__main__':
    for epoch in range(epoch):
        # 总体损失为0
        sum_loss = 0.0
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 使用GPU时候使用
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # 将梯度归零
            optimizer.zero_grad()
            # 将数据传入网络进行前向运算
            outputs = net(inputs)
            # 得到损失函数
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 通过梯度做一步参数更新
            optimizer.step()
            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
