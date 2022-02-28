import sys

import torch
from torch import nn

'''----------------------------------------------'''


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.layer1(x)


class LeNetReLU(nn.Module):
    def __init__(self):
        super(LeNetReLU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


class AlexNetSmall(nn.Module):
    def __init__(self):
        super(AlexNetSmall, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(),  # AlexNet: 3->96, k=11, s=4, p=0
            nn.MaxPool2d(kernel_size=2, stride=2),  # AlexNet: k=3, s=2
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(),  # AlexNet： 96->256, k=5, p=2
            nn.MaxPool2d(kernel_size=2, stride=2),  # AlexNet： k=3, s=2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),  # AlexNet： 256->384, k=3, p=1
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),  # AlexNet： 384->384, k=3, p=1
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),  # AlexNet： 384->256, k=3, p=1
            nn.MaxPool2d(kernel_size=2, stride=2),  # AlexNet： k=3, s=2

            # 参数量远多于LeNet，故增加Dropout防止过拟合
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 512), nn.ReLU(),  # AlexNet: 6000->4096
            nn.Dropout(p=0.2),
            nn.Linear(512, 512), nn.ReLU(),  # AlexNet: 4096->4096
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)  # AlexNet: 4096->1000
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG9Small(nn.Module):
    def __init__(self, conv_settings=None):
        super(VGG9Small, self).__init__()
        if not conv_settings:
            conv_settings = ((2, 32), (2, 64), (2, 128))  # (层数，输出通道数)

        in_channels = 1
        vgg_layers = []
        for (num_convs, out_channels) in conv_settings:
            vgg_layers.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.layer1 = nn.Sequential(
            *vgg_layers,

            # 降低通道数
            nn.Conv2d(out_channels, 64, kernel_size=1), nn.ReLU(),

            # 参数量远多于LeNet，故增加Dropout防止过拟合
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 512), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


def nin_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1)
    )


class NiN(nn.Module):
    """启发自AlexNet，用nin block代替卷积和全连接层"""

    def __init__(self, conv_settings=None):
        super(NiN, self).__init__()
        self.layer1 = nn.Sequential(
            nin_block(1, 32, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nin_block(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nin_block(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 增加Dropout防止过拟合
            nn.Dropout(p=0.2),
            nin_block(128, 10, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 给定输出尺寸，通道不变，自适应设置kernel_size和stride
            nn.Flatten()
        )

    def forward(self, x):
        return self.layer1(x)


'''----------------------------------------------'''


class Inception(nn.Module):
    # c1-c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1: 1层1*1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2: 1*1+3*3
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3: 1*1+5*5
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4:3*3maxpooling，s1，p1；1*1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 这样pooling后图宽高不变
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 四条通路进行输出
        p1 = self.relu(self.p1_1(x))
        p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3 = self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4 = self.relu(self.p4_2(self.p4_1(x)))

        # 在通道维度上拼接这四个输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # 1->32，尺寸28-4=24
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
        )
        # 32->96，尺寸减半24->12
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        # 96->240，尺寸减半12->6
        self.layer3 = nn.Sequential(
            Inception(96, 32, (48, 64), (8, 16), 16),
            Inception(128, 64, (64, 96), (16, 48), 32),
            nn.MaxPool2d(2),
        )
        # 240->416，尺寸减半6->3
        self.layer4 = nn.Sequential(
            Inception(240, 96, (48, 104), (8, 24), 32),
            Inception(256, 80, (56, 112), (12, 32), 32),
            Inception(256, 64, (64, 128), (12, 32), 32),
            Inception(256, 56, (122, 144), (16, 32), 32),
            Inception(264, 128, (80, 160), (16, 64), 64),
            nn.MaxPool2d(kernel_size=2)
        )
        # 416->512，尺寸3->1
        self.layer5 = nn.Sequential(
            Inception(416, 128, (80, 160), (16, 64), 64),
            Inception(416, 192, (96, 192), (24, 64), 64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # 完整结构，最后加一个线性层512->10
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)


'''----------------------------------------------'''


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(x1))
        if self.conv3:
            x = self.conv3(x)
        x1 += x
        return self.relu(x1)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:  # 除第一个block外，尺寸减半，通道加倍
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 1->32，尺寸28-4=24
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 32->32，尺寸24->24
        self.layer2 = nn.Sequential(*resnet_block(32, 32, 2, first_block=True))
        # 32->64，尺寸24->12
        self.layer3 = nn.Sequential(*resnet_block(32, 64, 2))
        # 64->128，尺寸12->6
        self.layer4 = nn.Sequential(*resnet_block(64, 128, 2))
        # 128->256，尺寸6->3
        self.layer5 = nn.Sequential(*resnet_block(128, 256, 2))
        # 3->1，全局平均池化，完整结构
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)


# ResnetV2，BN->Relu->weight的顺序
class ResidualV2(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super(ResidualV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(self.relu(self.bn1(x)))
        x1 = self.conv2(self.relu(self.bn2(x1)))
        if self.conv3:
            x = self.conv3(x)
        x1 += x
        return x1


def resnet_blockV2(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:  # 除第一个block外，尺寸减半，通道加倍
            blk.append(ResidualV2(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResidualV2(out_channels, out_channels))
    return blk


class ResNetV2(nn.Module):
    def __init__(self):
        super(ResNetV2, self).__init__()
        # 1->32，尺寸28-4=24
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
        )
        # 32->32，尺寸24->24
        self.layer2 = nn.Sequential(*resnet_blockV2(32, 32, 2, first_block=True))
        # 32->64，尺寸24->12
        self.layer3 = nn.Sequential(*resnet_blockV2(32, 64, 2))
        # 64->128，尺寸12->6
        self.layer4 = nn.Sequential(*resnet_blockV2(64, 128, 2))
        # 128->256，尺寸6->3
        self.layer5 = nn.Sequential(*resnet_blockV2(128, 256, 2))
        # 3->1，全局平均池化，完整结构
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)


'''----------------------------------------------'''


def bn_relu_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(bn_relu_conv(in_channels + i * growth_rate, growth_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = torch.cat((X, Y), dim=1)
        return X


# 过渡层，用于减半尺寸，以及降低通道数
def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        # 1->32，尺寸28-4=24
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1),
        )

        num_channels = 64  # 当前通道数
        growth_rate = 32
        num_convs_in_dense_blocks = [4, 4, 4]
        blks = []

        # 32->240, 24*24->3*3
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate  # 稠密块输出通道数

            # 两个稠密块之间加一个过渡层减半宽高和通道数
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2  # 通道数减半

        # 3->1，全局平均池化，完整结构
        self.layers = nn.Sequential(
            self.layer1,
            *blks,
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, 10)
        )

    def forward(self, x):
        return self.layers(x)
