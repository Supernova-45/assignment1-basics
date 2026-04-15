import os

import modal
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=7200,
    max_containers=5,
)
def train(lr: float):
    import numpy as np
    import torch
    import time
    import wandb
    from einops import rearrange
    from src.transformer import TransformerLM, RotaryPositionalEmbedding
    from src.training_helpers import (
        cross_entropy,
        AdamW,
        lr_cosine_schedule,
        gradient_clipping,
        get_batch,
        save_checkpoint,
    )

    os.makedirs(str(DATA_PATH / "checkpoints"), exist_ok=True)

    # config
    wandb.init(
        entity="alexandrasuriya-ml",
        project="cs336-a1",
        config={
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "vocab_size": 10000,
            "context_length": 256,
            "rope_theta": 10000.0,
            "lr": lr,
            "lr_min": 1e-4,
            "warmup_steps": 500,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "num_steps": 10000,
            "batch_size": 128,
            "max_grad_norm": 1.0,
            "device": "cuda",
            "architecture": "TransformerLM",
            "checkpoint_path": str(DATA_PATH / "checkpoints" / f"nope_lr_{lr}.pt"),
        },
    )
    cfg = wandb.config

    # Use DATA_PATH for data
    train_data = np.memmap(str(DATA_PATH / "tokenized" / "tiny_train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(DATA_PATH / "tokenized" / "tiny_valid.bin"), dtype=np.uint16, mode="r")

    # num_heads, max_seq_len=cfg.context_length)
    lm = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
    )
    lm.to(device=cfg.device)
    lm = torch.compile(lm)
    optimizer = AdamW(lm.parameters(), cfg.lr, cfg.betas, cfg.eps, cfg.weight_decay)

    token_positions = torch.arange(cfg.context_length, device=cfg.device)
    # training loop
    start_time = time.time()
    for t in range(cfg.num_steps):
        new_lr = lr_cosine_schedule(t, cfg.lr, cfg.lr_min, cfg.warmup_steps, cfg.num_steps)
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, device=cfg.device)
        logits = lm.forward(inputs, token_positions)
        loss = cross_entropy(rearrange(logits, "b c v -> (b c) v"), rearrange(targets, "b c -> (b c)"))
        loss.backward()
        gradient_clipping(lm.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - start_time
        # logging
        if t % 100 == 0:
            # test on validation dataset
            with torch.no_grad():
                val_inputs, val_targets = get_batch(val_data, cfg.batch_size, cfg.context_length, device=cfg.device)
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
            save_checkpoint(lm, optimizer, t, cfg.checkpoint_path)

    save_checkpoint(lm._orig_mod, optimizer, t, cfg.checkpoint_path)

    user_volume = modal.Volume.from_name("basics-alexskim")
    user_volume.commit()
    wandb.finish()


@app.local_entrypoint()
def main():
    lr = [0.0005, 0.001, 0.005, 0.01]
    for result in train.map(lr):
        print(result)


if __name__ == "__main__":
    main()
