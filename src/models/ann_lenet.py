from torch import nn
import torch.nn.functional as F


class LeNetANN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * feature_size * feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
