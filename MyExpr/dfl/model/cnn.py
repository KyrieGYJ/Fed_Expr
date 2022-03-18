
import torch.nn as nn
import torch.nn.functional as F

# A CNN with two 5x5 convolution layers
# the first with 32 channels, the second with 64, each followed with 2x2 max pooling
# a fully connected layer with 512 units and ReLu activation,
# a final softmax output layer (1,663,370 total parameters).
# class CNN_McMahan(nn.Module):
#
#     def __init__(self):
#         super(CNN_McMahan, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
#             nn.MaxPool2d(2, 2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
#             nn.MaxPool2d(2, 2)
#         )
#         self.fc = nn.Linear(64 * 4 * 4, 10)
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.fc(x)
#         x = self.softmax(x)
#
#         return x

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