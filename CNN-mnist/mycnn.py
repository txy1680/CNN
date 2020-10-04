import torch.nn as nn


class MyCNN(nn.Module):  # 继承module
    def __init__(self):  # 定义构造函数
        super(MyCNN, self).__init__()  # 计算公式pading=(k_size-1)/2
        self.con1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),  # 1*28*28--->=6*28*28
            nn.MaxPool2d(2),  # 6*28*28-->=6*14*14
            nn.ReLU()
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(6, 60, 5, 1, 2),  # 6*14*14-->60*14*14
            nn.MaxPool2d(2),  # 60*14*14-->60*7*7
            nn.ReLU()
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(60, 96, 5, 1, 2),  # 60*7*7-->96*7*7
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 7 * 7, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, inputs):
        out = self.con1(inputs)
        out = self.con2(out)
        out = self.con3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
