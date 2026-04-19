import torch
from einops import rearrange, reduce


@torch.no_grad()
def decoding(model, tokenizer, prompt, max_tokens, temperature=1.0, top_p=1.0, device="cpu"):
    context_length = model.token_embeddings.weight.shape[0]

    ids = tokenizer.encode(prompt)
    ids_tensor = rearrange(torch.tensor(ids, dtype=torch.long, device=device), 't -> 1 t')  # (1, T)

    # find EOS token id if available
    eos_bytes = "<|endoftext|>".encode("utf-8")
    eos_id = tokenizer.token_to_id.get(eos_bytes, None)

    for _ in range(max_tokens):
        # truncate to context window
        ctx = ids_tensor[:, -context_length:]

        logits = model(ctx)                                             # (1, T, V)
        logits = rearrange(logits[:, -1, :], '1 v -> v')               # last position → (V,)

        # temperature scaling
        logits = logits / max(temperature, 1e-8)

        probs = torch.softmax(logits, dim=-1)                          # (V,)

        # top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            # renormalize using einops reduce
            sorted_probs = sorted_probs / reduce(sorted_probs, 'v -> ', 'sum')
            next_token = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)]  # scalar
        else:
            next_token = torch.multinomial(probs, num_samples=1)       # scalar

        ids_tensor = torch.cat([ids_tensor, rearrange(next_token, '-> 1 1')], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    generated_ids = rearrange(ids_tensor, '1 t -> t').tolist()
    return tokenizer.decode(generated_ids)

