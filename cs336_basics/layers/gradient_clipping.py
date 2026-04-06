import torch
import torch.nn as nn
from typing import List

def gradient_clipping(params: List[nn.Parameter], M, ep = 1e-6):
    total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in params if p.grad is not None))
    if total_norm >= M:
        scale = M / (total_norm + ep)
        for p in params:
            if p.grad is not None:
                p.grad *= scale