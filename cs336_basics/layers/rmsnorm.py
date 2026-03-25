import torch
import torch.nn as nn
import einops

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        ms = einops.reduce(x**2, '... d_model -> ... 1', 'mean')
        rms = torch.rsqrt(ms + self.eps)
        rmsnorm = einops.einsum(x, rms, self.weight, '... d, ... 1, d -> ... d')
        return rmsnorm.to(in_type)