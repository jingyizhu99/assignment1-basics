import argparse
import os

import numpy as np
import torch
from einops import rearrange

from cs336_basics.layers import TransformerLM
from cs336_basics.training import cross_entropy, AdamW, learning_rate_schedule, gradient_clipping, data_loading
from cs336_basics.training.checkpointing import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TransformerLM")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized training data (.npy / memmap)")
    parser.add_argument("--val_data",   type=str, required=True, help="Path to tokenized validation data (.npy / memmap)")
    parser.add_argument("--data_dtype", type=str, default="uint16", help="numpy dtype of the memmap files")

    # Model
    parser.add_argument("--vocab_size",      type=int,   default=50257)
    parser.add_argument("--context_length",  type=int,   default=256)
    parser.add_argument("--d_model",         type=int,   default=512)
    parser.add_argument("--num_heads",       type=int,   default=8)
    parser.add_argument("--d_ff",            type=int,   default=2048)
    parser.add_argument("--num_layers",      type=int,   default=6)
    parser.add_argument("--theta",           type=float, default=10000.0)

    # Optimizer
    parser.add_argument("--lr_max",       type=float, default=3e-4)
    parser.add_argument("--lr_min",       type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int,   default=100,  help="Linear warmup steps (Tw)")
    parser.add_argument("--cosine_iters", type=int,   default=10000, help="End of cosine decay (Tc)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1",        type=float, default=0.9)
    parser.add_argument("--beta2",        type=float, default=0.95)
    parser.add_argument("--eps",          type=float, default=1e-8)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    # Training
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_iters",   type=int, default=10000)
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging / checkpointing
    parser.add_argument("--log_every",        type=int, default=100)
    parser.add_argument("--val_batches",      type=int, default=10,   help="Number of batches for validation estimate")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir",   type=str, default="checkpoints")
    parser.add_argument("--resume",           type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, val_batches, vocab_size):
    model.eval()
    losses = []
    for _ in range(val_batches):
        x, y = data_loading(val_data, batch_size, context_length, device)
        logits = model(x)                               # (B, T, V)
        loss = cross_entropy(rearrange(logits, 'b t v -> (b t) v'), rearrange(y, 'b t -> (b t)'))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()

    # Data  (memory-mapped — never loaded fully into RAM)
    dtype = np.dtype(args.data_dtype)
    train_data = np.memmap(args.train_data, dtype=dtype, mode="r")
    val_data   = np.memmap(args.val_data,   dtype=dtype, mode="r")

    # Model
    device = args.device
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        theta=args.theta,
        context_length=args.context_length,
        device=device,
    )
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # (Optional) resume from checkpoint
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(src=args.resume, model=model, optimizer=optimizer)
        print(f"Resumed from checkpoint '{args.resume}' at iteration {start_iter}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    for step in range(start_iter + 1, args.max_iters + 1):

        # 1. Learning-rate schedule
        lr = learning_rate_schedule(
            t=step,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            Tw=args.warmup_iters,
            Tc=args.cosine_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 2. Sample a batch
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        # 3. Forward pass + loss
        logits = model(x) # (B, T, V)
        loss = cross_entropy(rearrange(logits, 'b t v -> (b t) v'), rearrange(y, 'b t -> (b t)'))

        # 4. Backward + gradient clip + optimizer step
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(list(model.parameters()), args.grad_clip)
        optimizer.step()

        # 5. Logging
        if step % args.log_every == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length,
                device, args.val_batches, args.vocab_size,
            )
            print(
                f"step {step:6d} | lr {lr:.2e} | "
                f"train_loss {loss.item():.4f} | val_loss {val_loss:.4f}"
            )

        # 6. Checkpointing
        if step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, out=ckpt_path)
            print(f"  -> saved checkpoint to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()