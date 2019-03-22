import torch.nn as nn
import math


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(9, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.policy1 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.policy2 = nn.Sequential(
            nn.Linear(450, 15 * 15),
            nn.LeakyReLU(inplace=True)
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.policy1)
        self.weight_init(self.policy2)

    def head(self, x):
        x = self.policy1(x)
        x = x.view(x.size(0), -1)
        x = self.policy2(x)

        return x

    @staticmethod
    def weight_init(elem):
        for m in elem.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.convolutional(x)

        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out

        return self.head(x)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(11, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.value1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.value2 = nn.Sequential(
            nn.Linear(15 * 15, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Tanh()
        )

        self.weight_init(self.convolutional)
        self.weight_init(self.residual1)
        self.weight_init(self.residual2)
        self.weight_init(self.value1)
        self.weight_init(self.value2)

    def head(self, x):
        x = self.value1(x)
        x = x.view(x.size(0), -1)
        x = self.value2(x)

        return x

    @staticmethod
    def weight_init(elem):
        for m in elem.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.convolutional(x)
        out = self.residual1(x)
        x = x + out
        out = self.residual2(x)
        x = x + out

        return self.head(x)
