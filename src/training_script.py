import argparse

from einops import rearrange
import numpy as np
import torch
import time

import wandb

from src.transformer import TransformerLM, RotaryPositionalEmbedding
from src.training_helpers import (
    cross_entropy,
    AdamW,
    lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    run = wandb.init(
        entity="alexandrasuriya-ml",
        project="cs336-a1",
        # Track hyperparameters and run metadata.
        config={
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "rope_theta": args.rope_theta,
            "lr": args.lr,
            "lr_min": args.lr_min,
            "warmup_steps": args.warmup_steps,
            "betas": args.betas,
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm,
            "device": args.device,
            "architecture": "TransformerLM",
        },
    )

    # load data efficiently
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    rope = RotaryPositionalEmbedding(args.rope_theta, args.d_model // args.num_heads, max_seq_len=args.context_length)
    lm = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        rope=rope,
    )
    optimizer = AdamW(lm.parameters(), args.lr, args.betas, args.eps, args.weight_decay)

    token_positions = torch.arange(args.context_length, device=args.device)
    lm.to(device=args.device)
    # training loop
    start_time = time.time()
    for t in range(args.num_steps):
        new_lr = lr_cosine_schedule(t, args.lr, args.lr_min, args.warmup_steps, args.num_steps)
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device=args.device)
        logits = lm.forward(inputs, token_positions)
        loss = cross_entropy(rearrange(logits, "b c v -> (b c) v"), rearrange(targets, "b c -> (b c)"))
        loss.backward()
        gradient_clipping(lm.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - start_time
        # logging
        if t % 100 == 0:
            # test on validation dataset
            with torch.no_grad():
                val_inputs, val_targets = get_batch(val_data, args.batch_size, args.context_length, device=args.device)
                val_logits = lm(val_inputs, token_positions)
                val_loss = cross_entropy(
                    rearrange(val_logits, "b t v -> (b t) v"), rearrange(val_targets, "b t -> (b t)")
                )
                val_perplexity = torch.exp(val_loss)

            print(f"Step {t} | Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f} | Time: {elapsed}")
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "val_perplexity": val_perplexity.item(),
                    "step": t,
                    "wall_clock_seconds": elapsed,
                }
            )

        # checkpointing
        if t % 1000 == 0:
            save_checkpoint(lm, optimizer, t, args.checkpoint_path)

    save_checkpoint(lm, optimizer, t, args.checkpoint_path)


if __name__ == "__main__":
    main()
