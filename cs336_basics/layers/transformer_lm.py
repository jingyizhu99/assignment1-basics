import torch
import torch.nn as nn
from . import Embedding, TransformerBlock, RMSNorm, Linear

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int, 
                 theta: float = None, context_length: int = None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device, dtype) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
        self.num_layers = num_layers

    def forward(self, token_ids: torch.Tensor):
        x = self.token_embeddings(token_ids)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = self.ln_final(x)
        return self.lm_head(x)