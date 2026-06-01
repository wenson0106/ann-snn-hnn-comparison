import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class STDPConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        beta: float = 0.95,
        threshold: float = 1.0,
        stdp_lr: float = 0.01,
        tau_pre: float = 10.0,
        tau_post: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.beta = beta
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.05
        )

        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.dw_accum: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.pre_trace = None
        self.post_trace = None
        self.dw_accum = None

    def forward(self, x: torch.Tensor, membrane: Optional[torch.Tensor] = None):
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(x, device=device, dtype=dtype)
            H_out = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
            self.post_trace = torch.zeros(
                B, self.out_channels, H_out, W_out, device=device, dtype=dtype
            )
            self.dw_accum = torch.zeros_like(self.weight, device=device, dtype=dtype)

        current = F.conv2d(x, self.weight, None, self.stride, self.padding)

        if membrane is None:
            membrane = torch.zeros_like(current, device=device, dtype=dtype)
        membrane = self.beta * membrane + current
        spike = (membrane >= self.threshold).to(dtype)
        membrane = membrane * (1.0 - spike.detach())

        decay_pre = math.exp(-1.0 / self.tau_pre) if self.tau_pre > 0 else 0.0
        decay_post = math.exp(-1.0 / self.tau_post) if self.tau_post > 0 else 0.0
        self.pre_trace = self.pre_trace * decay_pre + x
        self.post_trace = self.post_trace * decay_post + spike

        if x.any() or spike.any():
            pre_unfold = F.unfold(self.pre_trace, self.kernel_size, padding=self.padding)
            post_spike_flat = spike.flatten(2)
            dw_pot = torch.bmm(post_spike_flat, pre_unfold.transpose(1, 2))

            post_trace_flat = self.post_trace.flatten(2)
            pre_unfold_spike = F.unfold(x, self.kernel_size, padding=self.padding)
            dw_dep = torch.bmm(post_trace_flat, pre_unfold_spike.transpose(1, 2))

            dw = self.A_plus * dw_pot - self.A_minus * dw_dep
            dw = dw.mean(dim=0)
            dw = dw.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            self.dw_accum = self.dw_accum + dw

        return spike, membrane

    def apply_update(self) -> None:
        with torch.no_grad():
            self.weight.data.add_(self.stdp_lr * self.dw_accum)
        self.reset_state()
