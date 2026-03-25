import torch
import torch.nn as nn

from . import RMSNorm, MHA, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float = None, max_seq_len: int = None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MHA(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor):
        x = x + (self.attn(self.ln1(x)))
        y = x + (self.ffn(self.ln2(x)))
        return y