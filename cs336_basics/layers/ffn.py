import torch
import torch.nn as nn
from . import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
    
    def GLU(self, x: torch.Tensor):
        return self.SiLU(self.w1(x)) * self.w3(x)
    
    def forward(self, x: torch.Tensor):
        return self.w2(self.GLU(x))