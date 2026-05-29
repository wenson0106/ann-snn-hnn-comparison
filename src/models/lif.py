import torch
from torch import nn
from typing import Optional


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        distance = (membrane - threshold).abs()
        grad = (1.0 / (1.0 + distance).pow(2)).clamp(max=1.0)
        return grad_output * grad, None


class LIFNeuron(nn.Module):
    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input_current: torch.Tensor, membrane: Optional[torch.Tensor]):
        if membrane is None:
            membrane = torch.zeros_like(input_current)

        membrane = self.beta * membrane + input_current
        spike = SpikeFunction.apply(membrane, self.threshold)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class ConvLIFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        beta: float = 0.95,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.lif = LIFNeuron(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor, membrane: Optional[torch.Tensor]):
        current = self.conv(x)
        return self.lif(current, membrane)


class LinearLIFBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.95,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = LIFNeuron(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor, membrane: Optional[torch.Tensor]):
        current = self.linear(x)
        return self.lif(current, membrane)
