import torch
import torch.nn as nn
import einops

from . import Linear, RotaryPositionalEmbedding

def softmax(v: torch.Tensor, dim: int) -> torch.Tensor:
    m = v.max(dim=dim, keepdim=True).values # same size as v
    exp = torch.exp(v - m)
    exp_sum = exp.sum(dim=dim, keepdim=True)
    return exp / exp_sum

# key, query: (batch_size, ..., seq_len, d_k)
# value: (batch_size, ..., seq_len, d_v)
# mask: (seq_len, seq_len)
# return: (batch_size, ..., d_v)
def scaled_dot_product_attention(key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    QK = einops.einsum(query, key, '... n d_k, ... m d_k -> ... n m') # n = m = seq_len, row = query, column = key
    d_k = key.size(-1)
    pre_softmax = QK * (d_k ** -0.5) # (... seq_len seq_len)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    softmax_res = softmax(pre_softmax, -1)
    return einops.einsum(softmax_res, value, '... n m, ... m d_v -> ... n d_v')


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 theta: float = None, max_seq_len: int = None, device=None, dtype=None):
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * d_k, device, dtype)
        self.k_proj = Linear(d_model, num_heads * d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * d_v, device, dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device, dtype)
        self.num_heads = num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len) if theta is not None and max_seq_len is not None else None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        # causal masking
        ones = torch.ones(seq_len, seq_len)
        mask = torch.tril(ones).bool()

        # calculate QKV
        Q = self.q_proj(x) # (... seq_len, num_heads * d_k)
        K = self.k_proj(x)
        V = self.v_proj(x) # (... seq_len, num_heads * d_v)
        Q = einops.rearrange(Q, '... seq_len (h d_k) -> ... h seq_len d_k', h = self.num_heads)
        K = einops.rearrange(K, '... seq_len (h d_k) -> ... h seq_len d_k', h = self.num_heads)
        V = einops.rearrange(V, '... seq_len (h d_v) -> ... h seq_len d_v', h = self.num_heads)

        # rope
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            token_positions = einops.rearrange(token_positions, '... seq_len -> ... 1 seq_len')
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn = scaled_dot_product_attention(K, Q, V, mask=mask) # (... h seq_len d_v)

        # mearge heads back
        attn = einops.rearrange(attn, '... h seq_len d_v -> ... seq_len (h d_v)')
        return self.output_proj(attn)


