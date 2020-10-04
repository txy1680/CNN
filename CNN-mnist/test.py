from train import net
from load_data import test_dataset, test_loader
import torch
from torch.autograd import Variable

net.eval()  # 将模型变换为测试模式
correct = 0
total = 0
for data_test in test_loader:
    images, labels = data_test
    images, labels = Variable(images).cuda(), Variable(labels).cuda()
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)  # predicted为返回的最大的概率值的索引
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("correct1: ", correct)
print("Test acc: {0}".format(correct.item() / len(test_dataset)))
