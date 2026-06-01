import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from src.models.lif import ConvLIFBlock, LinearLIFBlock
from src.models.stdp import STDPConv2d


class HybridSTDPLeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feature_size: int = 5,
        num_classes: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        stdp_lr: float = 0.01,
        tau_pre: float = 10.0,
        tau_post: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.stdp1 = STDPConv2d(
            in_channels, 6, kernel_size=5, beta=beta, threshold=threshold,
            stdp_lr=stdp_lr, tau_pre=tau_pre, tau_post=tau_post,
            A_plus=A_plus, A_minus=A_minus,
        )
        self.conv2 = ConvLIFBlock(6, 16, kernel_size=5, beta=beta, threshold=threshold)
        self.fc1 = LinearLIFBlock(16 * feature_size * feature_size, 120, beta=beta, threshold=threshold)
        self.fc2 = LinearLIFBlock(120, 84, beta=beta, threshold=threshold)
        self.readout = nn.Linear(84, num_classes)

    def stdp_forward(self, x: torch.Tensor) -> None:
        B, T = x.shape[:2]
        memb: Optional[torch.Tensor] = None
        for t in range(T):
            _, memb = self.stdp1(x[:, t], memb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        memories: dict[str, Optional[torch.Tensor]] = {
            "conv2": None,
            "fc1": None,
            "fc2": None,
        }
        memb1: Optional[torch.Tensor] = None
        logits = []

        for t in range(T):
            out, memb1 = self.stdp1(x[:, t], memb1)
            out = F.avg_pool2d(out, kernel_size=2)
            out, memories["conv2"] = self.conv2(out, memories["conv2"])
            out = F.avg_pool2d(out, kernel_size=2)
            out = out.flatten(1)
            out, memories["fc1"] = self.fc1(out, memories["fc1"])
            out, memories["fc2"] = self.fc2(out, memories["fc2"])
            logits.append(self.readout(out))

        return torch.stack(logits, dim=0).mean(dim=0)

    def freeze_stdp(self) -> None:
        for p in self.stdp1.parameters():
            p.requires_grad = False
