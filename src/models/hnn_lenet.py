import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from src.models.lif import ConvLIFBlock, LinearLIFBlock, LIFNeuron


class LeNetHNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 4,
        num_classes: int = 10,
        time_steps: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=kernel_size)
        self.first_spike = LIFNeuron(beta=beta, threshold=threshold)
        self.conv2 = ConvLIFBlock(6, 16, kernel_size=kernel_size, beta=beta, threshold=threshold)
        self.fc1 = LinearLIFBlock(16 * feature_size * feature_size, 120, beta=beta, threshold=threshold)
        self.fc2 = LinearLIFBlock(120, 84, beta=beta, threshold=threshold)
        self.readout = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep image input analog until after the first convolutional layer.
        first_current = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        memories: dict[str, Optional[torch.Tensor]] = {
            "first": None,
            "conv2": None,
            "fc1": None,
            "fc2": None,
        }
        logits = []

        for _ in range(self.time_steps):
            out, memories["first"] = self.first_spike(first_current, memories["first"])
            out, memories["conv2"] = self.conv2(out, memories["conv2"])
            out = F.avg_pool2d(out, kernel_size=2)
            out = out.flatten(1)
            out, memories["fc1"] = self.fc1(out, memories["fc1"])
            out, memories["fc2"] = self.fc2(out, memories["fc2"])
            logits.append(self.readout(out))

        return torch.stack(logits, dim=0).mean(dim=0)
