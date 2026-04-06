import torch
from math import sqrt
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 1
                grad = p.grad.data
                t = state["t"]
                m = beta1 * state["m"] + (1-beta1) * grad
                v = beta2 * state["v"] + (1-beta2) * grad ** 2
                lrt = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data = p.data - lrt * m / (v.sqrt() + eps)
                p.data = p.data - lr * lam * p.data
                state["t"] += 1
                state["m"] = m
                state["v"] = v
        return loss
    