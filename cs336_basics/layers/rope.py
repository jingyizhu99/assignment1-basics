import torch
import torch.nn as nn
import einops

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # will calculate all possible angles for every i & k
        # store in tensor of size (max_seq_len, d_k / 2)
        k = torch.arange(0, d_k // 2) # [0, 1, ... d//2 - 1], odd d_k leave the last dimension unrotated
        i = torch.arange(max_seq_len).float() # [0, 1, ..., max_seq_len-1]
        inv = 1.0 / (theta ** (2 * k / d_k))
        angles = einops.einsum(i, inv, 'i, j -> i j')

        # calculate cos, sin pair and store as buffer
        self.register_buffer("emb_cos", angles.cos(), persistent=False)
        self.register_buffer("emb_sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x of size (..., seq_len, d_k)
        # token_positions of size (..., seq_len)
        cos = self.emb_cos[token_positions] # (..., seq_len, d/2)
        sin = self.emb_sin[token_positions]
        
        # split into x_{2k-1} and x_2k, where each tensor has size (... seq_len, d_k / 1)
        x_paired = einops.rearrange(x, '... (d_half pair) -> ... d_half pair', pair=2)
        x_2k_min_1 = x_paired[..., 0] # (... d_half)
        x_2k = x_paired[..., 1]

        x_rot_1 = x_2k_min_1 * cos - x_2k * sin
        x_rot_2 = x_2k_min_1 * sin + x_2k * cos

        out = torch.stack([x_rot_1, x_rot_2], dim=-1) # (... d_half) stack with (... d_half) -> (... d_half, 2)
        return einops.rearrange(out, '... d_half pair -> ... (d_half pair)')




