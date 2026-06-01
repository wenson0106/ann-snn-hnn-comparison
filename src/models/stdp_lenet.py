import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from src.models.stdp import STDPConv2d


class STDPLeNet(nn.Module):
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
        self.stdp2 = STDPConv2d(
            6, 16, kernel_size=5, beta=beta, threshold=threshold,
            stdp_lr=stdp_lr, tau_pre=tau_pre, tau_post=tau_post,
            A_plus=A_plus, A_minus=A_minus,
        )
        self.readout = nn.Linear(16 * feature_size * feature_size, num_classes)

    def stdp_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        memb1: Optional[torch.Tensor] = None
        memb2: Optional[torch.Tensor] = None
        spike_count: Optional[torch.Tensor] = None

        for t in range(T):
            s1, memb1 = self.stdp1(x[:, t], memb1)
            s1 = F.avg_pool2d(s1, kernel_size=2)
            s2, memb2 = self.stdp2(s1, memb2)
            s2 = F.avg_pool2d(s2, kernel_size=2)
            if spike_count is None:
                spike_count = s2
            else:
                spike_count = spike_count + s2

        features = spike_count / T
        return features  # [B, 16, 5, 5]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.readout(features.flatten(1))

    def extract_features(self, loader, device: torch.device):
        self.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                self.stdp1.reset_state()
                self.stdp2.reset_state()
                feats = self.stdp_forward(inputs)
                all_features.append(feats.flatten(1).cpu())
                all_labels.append(labels)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
