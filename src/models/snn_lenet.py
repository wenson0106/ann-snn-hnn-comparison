import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from src.models.lif import ConvLIFBlock, LinearLIFBlock


class LeNetSNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        feature_size: int = 5,
        num_classes: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.conv1 = ConvLIFBlock(input_channels, 6, kernel_size=kernel_size, beta=beta, threshold=threshold)
        self.conv2 = ConvLIFBlock(6, 16, kernel_size=kernel_size, beta=beta, threshold=threshold)
        self.fc1 = LinearLIFBlock(16 * feature_size * feature_size, 120, beta=beta, threshold=threshold)
        self.fc2 = LinearLIFBlock(120, 84, beta=beta, threshold=threshold)
        self.readout = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        memories: dict[str, Optional[torch.Tensor]] = {
            "conv1": None,
            "conv2": None,
            "fc1": None,
            "fc2": None,
        }
        logits = []

        for step in range(x.shape[1]):
            out, memories["conv1"] = self.conv1(x[:, step], memories["conv1"])
            out = F.avg_pool2d(out, kernel_size=2)
            out, memories["conv2"] = self.conv2(out, memories["conv2"])
            out = F.avg_pool2d(out, kernel_size=2)
            out = out.flatten(1)
            out, memories["fc1"] = self.fc1(out, memories["fc1"])
            out, memories["fc2"] = self.fc2(out, memories["fc2"])
            logits.append(self.readout(out))

        return torch.stack(logits, dim=0).mean(dim=0)
