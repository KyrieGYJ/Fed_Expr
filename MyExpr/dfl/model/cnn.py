import torch
import torch.nn as nn
import torch.nn.functional as F


# network for mnist
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        # print(x.shape)
        x = self.act(self.conv1(x))
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.act(self.conv2(x))
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x


# network for cifar10
class TFConvNet(nn.Module):
    """
    Network architecture in the Tensorflow image classification tutorial
    """
    def __init__(self, num_classes=10):
        super(TFConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def embed(self, x, layer=3):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        if layer == 1:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        if layer == 2:
            return self.fc2(x)


class BaseConvNet(nn.Module):
    """
    Network architecture in the PyTorch image classification tutorial
    """
    def __init__(self, num_classes):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def embed(self, x, layer=2):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        if layer == 1:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        if layer == 2:
            return self.fc2(x)
        x = F.relu(self.fc2(x))
        if layer == 3:
            return self.fc3(x)