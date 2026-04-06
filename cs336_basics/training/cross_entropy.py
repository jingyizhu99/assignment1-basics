import torch
from einops import rearrange, reduce

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (b, V) logits, targets: (b,) integer class indices
    max = inputs.max(dim=-1, keepdim=True).values # (b,)
    stable = inputs - max

    # M + \log \sum_a \exp(o_i[a] - M)
    sum_exp = reduce(torch.exp(stable), '... v -> ...', 'sum')
    lse = rearrange(max, '... 1 -> ...') + torch.log(sum_exp)

    # o_i[x_{i+1}]
    targets_unsqueezed = rearrange(targets, '... -> ... 1')
    target_scores = torch.gather(inputs, dim=-1, index=targets_unsqueezed)
    target_scores = rearrange(target_scores, '... 1 -> ...')

    loss_per_token = lse - target_scores

    return reduce(loss_per_token, '... -> ', 'mean')