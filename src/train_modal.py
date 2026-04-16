import os

import modal
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=21600,
    max_containers=4,
)
def train(config):
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
        config=config,
    )
    cfg = wandb.config

    # Use DATA_PATH for data
    train_data = np.memmap(str(DATA_PATH / "tokenized" / "owt_train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(DATA_PATH / "tokenized" / "owt_valid.bin"), dtype=np.uint16, mode="r")

    rope = RotaryPositionalEmbedding(cfg.rope_theta, cfg.d_model // cfg.num_heads, max_seq_len=cfg.context_length)
    lm = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        rope=rope,
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = lm.forward(inputs, token_positions)
            loss = cross_entropy(rearrange(logits, "b c v -> (b c) v"), rearrange(targets, "b c -> (b c)"))
        loss.backward()
        gradient_clipping(lm.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - start_time
        # logging
        if t % 200 == 0:
            # test on validation dataset
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    total_val_loss = 0
                    for i in range(10):
                        val_inputs, val_targets = get_batch(val_data, cfg.batch_size, cfg.context_length, device=cfg.device)
                        val_logits = lm(val_inputs, token_positions)
                        total_val_loss += cross_entropy(
                            rearrange(val_logits, "b t v -> (b t) v"), rearrange(val_targets, "b t -> (b t)")
                        )
                    val_loss = total_val_loss / 10
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
    # final val loss check
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0

        max_start = len(val_data) - cfg.context_length - 1
        starts = list(range(0, max_start + 1, cfg.context_length))

        for j in range(0, len(starts), cfg.batch_size):
            batch_starts = starts[j : j + cfg.batch_size]

            val_inputs = torch.stack([
                torch.tensor(val_data[s : s + cfg.context_length], dtype=torch.long, device=cfg.device)
                for s in batch_starts
            ])
            val_targets = torch.stack([
                torch.tensor(val_data[s + 1 : s + cfg.context_length + 1], dtype=torch.long, device=cfg.device)
                for s in batch_starts
            ])

            token_positions = torch.arange(val_inputs.shape[1], device=cfg.device)
            val_logits = lm(val_inputs, token_positions)

            batch_loss = cross_entropy(
                rearrange(val_logits, "b t v -> (b t) v"),
                rearrange(val_targets, "b t -> (b t)")
            )

            n_tokens = val_targets.numel()
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens

        val_final_loss = total_loss / total_tokens
        wandb.log({
            "val_final_loss": val_final_loss
        })

    user_volume = modal.Volume.from_name("basics-alexskim")
    user_volume.commit()
    wandb.finish()


@app.local_entrypoint()
def main():
    """
    lrs = [0.001, 0.005, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02]
    lr_configs = [
        {
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "vocab_size": 32000,
            "context_length": 512,
            "rope_theta": 10000.0,
            "lr": lr,
            "lr_min": 1e-4,
            "warmup_steps": 500,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "num_steps": 10000,
            "batch_size": 256,
            "max_grad_norm": 1.0,
            "device": "cuda",
            "architecture": "TransformerLM",
            "checkpoint_path": str(DATA_PATH / "checkpoints" / f"owt_{lr}.pt"),
        }
        for lr in lrs
    ]
    for result in train.map(lr_configs):
        print(result)

    batch_sizes = [64, 128, 256, 512]
    batch_size_configs = [
        {
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "vocab_size": 32000,
            "context_length": 512,
            "rope_theta": 10000.0,
            "lr": 0.008,
            "lr_min": 1e-4,
            "warmup_steps": 500,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "num_steps": 10000,
            "batch_size": batch_size,
            "max_grad_norm": 1.0,
            "device": "cuda",
            "architecture": "TransformerLM",
            "checkpoint_path": str(DATA_PATH / "checkpoints" / f"owt_{batch_size}.pt"),
        }
        for batch_size in batch_sizes
    ]
    for result in train.map(batch_size_configs):
        print(result)
    """

    warmup_steps = [100, 200, 500, 1000]
    warmup_step_configs = [
        {
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "vocab_size": 32000,
            "context_length": 512,
            "rope_theta": 10000.0,
            "lr": 0.007,
            "lr_min": 1e-4,
            "warmup_steps": warmup_step,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "num_steps": 10000,
            "batch_size": 256,
            "max_grad_norm": 1.0,
            "device": "cuda",
            "architecture": "TransformerLM",
            "checkpoint_path": str(DATA_PATH / "checkpoints" / f"owt_{warmup_step}.pt"),
        }
        for warmup_step in warmup_steps
    ]
    for result in train.map(warmup_step_configs):
        print(result)

    weight_decays = [0.0, 0.001, 0.01, 0.05]
    weight_decay_configs = [
        {
            "d_model": 512,
            "num_heads": 16,
            "num_layers": 4,
            "d_ff": 1344,
            "vocab_size": 32000,
            "context_length": 512,
            "rope_theta": 10000.0,
            "lr": 0.007,
            "lr_min": 1e-4,
            "warmup_steps": 500,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": weight_decay,
            "num_steps": 10000,
            "batch_size": 256,
            "max_grad_norm": 1.0,
            "device": "cuda",
            "architecture": "TransformerLM",
            "checkpoint_path": str(DATA_PATH / "checkpoints" / f"owt_{weight_decay}.pt"),
        }
        for weight_decay in weight_decays
    ]
    for result in train.map(weight_decay_configs):
        print(result)


if __name__ == "__main__":
    main()
