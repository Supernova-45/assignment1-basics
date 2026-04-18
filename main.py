import os

import modal
import typing

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from einops import rearrange
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import numpy.typing as npt
import math

from pathlib import Path, PurePosixPath

SUNET_ID = "alexskim"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"basics-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"basics-{SUNET_ID}", create_if_missing=True)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim().apt_install("wget", "gzip").uv_sync()
    image = image.add_local_python_source("cs336_basics")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"SOME_ENV_VAR": "some-value"}), modal.Secret.from_name("my-secret")]
    return secrets


class Linear(torch.nn.Module):
    """Linear class that performs a linear transformation."""

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.empty(size=(self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        weights_std = math.sqrt(2 / (self.in_features + self.out_features))
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(tensor=self.W, mean=0, std=weights_std, a=-3 * weights_std, b=3 * weights_std)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.T


class Embedding(torch.nn.Module):
    """Embedding class that performs an embedding lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        tie_embeddings: bool | None = False,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.W = torch.empty(size=(num_embeddings, embedding_dim), device=self.device, dtype=self.dtype)
        if tie_embeddings:
            std = 1 / math.sqrt(embedding_dim)
        else:
            std = 1
        self.W = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor=self.W, mean=0, std=std, a=-3, b=3))

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.W[token_ids]


class RMSNorm(torch.nn.Module):
    """Implements RMSNorm."""

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(1 / self.d_model * torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.gain

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    """Implements SwiGLU feed-forward network, composed of a SiLU activation function and a GLU."""

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_ff = d_ff if d_ff else ((8 / 3 * d_model) // 64 + 1) * 64
        std = math.sqrt(2 / (d_model + self.d_ff))
        self.W1 = torch.nn.Parameter(torch.empty((self.d_ff, d_model)))
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W2 = torch.nn.Parameter(torch.empty((d_model, self.d_ff)))
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W3 = torch.nn.Parameter(torch.empty((self.d_ff, d_model)))
        torch.nn.init.trunc_normal_(self.W3, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.W1.T
        silu = y * torch.sigmoid(y)
        return (silu * (x @ self.W3.T)) @ self.W2.T


class ReLUSquared(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_ff = d_ff if d_ff else (4 * d_model)
        std = math.sqrt(2 / (d_model + self.d_ff))
        self.W1 = torch.nn.Parameter(torch.empty((self.d_ff, d_model)))
        torch.nn.init.trunc_normal_(self.W1, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W2 = torch.nn.Parameter(torch.empty((d_model, self.d_ff)))
        torch.nn.init.trunc_normal_(self.W2, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return (torch.relu(x @ self.W1.T) ** 2) @ self.W2.T


class RotaryPositionalEmbedding(torch.nn.Module):
    """Applies RoPE to input tensor to inject positional information."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        angles = 1 / torch.pow(self.theta, (2 * torch.arange(1, self.d_k / 2 + 1) - 2) / self.d_k)
        mat = torch.outer(torch.arange(self.max_seq_len), angles)
        self.register_buffer(name="cos_vals", tensor=torch.cos(mat), persistent=False)
        self.register_buffer(name="sin_vals", tensor=torch.sin(mat), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos_vals = self.cos_vals[token_positions]
        sin_vals = self.sin_vals[token_positions]
        x = rearrange(x, "... (pairs two) -> ... pairs two", two=2)
        x0 = x[..., 0]
        x1 = x[..., 1]
        new_x0 = x0 * cos_vals - x1 * sin_vals
        new_x1 = x0 * sin_vals + x1 * cos_vals
        return rearrange(torch.stack((new_x0, new_x1), dim=-1), "... pairs two -> ... (pairs two)")


def softmax(x: Tensor, dim: int) -> Tensor:
    input = x - torch.max(x, dim=dim, keepdim=True)[0]
    expd = torch.exp(input)
    result = expd / torch.sum(expd, dim=dim, keepdim=True)
    return result


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    pre_mask = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    if mask is not None:
        pre_mask = pre_mask.masked_fill(~mask, -float("inf"))
    return softmax(pre_mask, -1) @ V


class Attention(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, context_length: int, rope: RotaryPositionalEmbedding | None = None
    ):
        super().__init__()
        self.num_heads = num_heads
        std = math.sqrt(2 / (d_model + d_model))
        self.qkv_weight = torch.nn.Parameter(torch.empty(3 * d_model, d_model))
        torch.nn.init.trunc_normal_(self.qkv_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.o_proj_weight = torch.nn.Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.o_proj_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)).bool(), persistent=False)
        self.rope = rope

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # apply causal masking
        seq_len = x.shape[-2]
        mask = self.mask[:seq_len, :seq_len]
        qkv = x @ self.qkv_weight.T
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        # apply rope
        if self.rope is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        attention = rearrange(scaled_dot_product_attention(q, k, v, mask), "... h seq_len d_k -> ... seq_len (h d_k)")

        return attention @ self.o_proj_weight.T


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        use_relu: bool | None = False,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        if use_relu:
            self.ff = ReLUSquared(d_model, d_ff)
        else:
            self.ff = SwiGLU(d_model, d_ff)
        self.attention = Attention(d_model, num_heads, context_length, rope)

    def forward(self, x: Tensor, token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None):
        # first half
        norm1 = self.norm1.forward(x)
        first = x + self.attention.forward(norm1, token_positions)

        # second half
        norm2 = self.norm2.forward(first)
        second = first + self.ff.forward(norm2)

        return second


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        tie_embeddings: bool | None = False,
        use_relu: bool | None = False,
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, tie_embeddings=tie_embeddings)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, use_relu, rope) for _ in range(num_layers)]
        )
        self.norm3 = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        self.tie_embeddings = tie_embeddings
        if self.tie_embeddings:
            self.linear.W = self.embedding.W

    def forward(self, x: Tensor, token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None):
        # token embeddings
        if self.tie_embeddings:
            x = self.embedding.forward(x) * math.sqrt(self.d_model)
        else:
            x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x, token_positions)
        x = self.norm3.forward(x)
        return self.linear.forward(x)


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs = inputs.float()
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]
    indices = torch.arange(targets.shape[0], device=targets.device)
    loss = -1 * inputs[indices, targets] + torch.log(torch.sum(torch.exp(inputs), dim=-1))
    return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def optimize_sgd():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                new_lr = lr * math.sqrt(1 - group["beta2"] ** t) / (1 - group["beta1"] ** t)
                p.data -= lr * group["weight_decay"] * p.data  # apply weight decay
                m = group["beta1"] * m + (1 - group["beta1"]) * grad  # first moment update
                v = group["beta2"] * v + (1 - group["beta2"]) * grad * grad  # second moment update
                p.data -= (new_lr * m) / (torch.sqrt(v) + group["eps"])  # apply moment adjusted weight update
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1  # Increment iteration number.
        return loss


def lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    parameters = list(parameters)
    eps = 1e-6
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        norm += torch.norm(param.grad) ** 2
    norm = torch.sqrt(norm)

    if norm >= max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad *= max_l2_norm / (norm + eps)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randint(dataset.shape[0] - context_length, (batch_size,))
    inputs_arr = np.stack([dataset[i : i + context_length] for i in positions])
    targets_arr = np.stack([dataset[i + 1 : i + context_length + 1] for i in positions])
    inputs = torch.from_numpy(inputs_arr.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets_arr.astype(np.int64)).to(device)
    return (inputs, targets)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}, out)


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=3000,
    max_containers=1,
)
def train(cfg):
    import numpy as np
    import torch
    import time
    import wandb
    from einops import rearrange
    from cs336_basics.transformer import TransformerLM, RotaryPositionalEmbedding
    from cs336_basics.training_helpers import (
        cross_entropy,
        AdamW,
        lr_cosine_schedule,
        gradient_clipping,
        get_batch,
        save_checkpoint,
    )

    os.makedirs(str(DATA_PATH / "checkpoints"), exist_ok=True)

    # Use DATA_PATH for data
    train_data = np.memmap(str(DATA_PATH / "tokenized" / "owt_train_new.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(DATA_PATH / "tokenized" / "owt_valid_new.bin"), dtype=np.uint16, mode="r")

    torch.set_float32_matmul_precision("high")
    rope = RotaryPositionalEmbedding(
        cfg["rope_theta"],
        cfg["d_model"] // cfg["num_heads"],
        max_seq_len=cfg["context_length"],
    )

    lm = TransformerLM(
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        num_layers=cfg["num_layers"],
        tie_embeddings=cfg["tie_embeddings"],
        use_relu=cfg["use_relu"],
        rope=rope,
    )
    lm.to(device=cfg["device"])
    lm = torch.compile(lm)

    optimizer = AdamW(
        lm.parameters(),
        cfg["lr"],
        cfg["betas"],
        cfg["eps"],
        cfg["weight_decay"],
    )

    token_positions = torch.arange(cfg["context_length"], device=cfg["device"])

    start_time = time.time()
    for t in range(cfg["num_steps"]):
        if time.time() - start_time > cfg["max_time_seconds"]:
            print(f"Time limit reached at step {t}")
            break

        new_lr = lr_cosine_schedule(
            t,
            cfg["lr"],
            cfg["lr_min"],
            cfg["warmup_steps"],
            cfg["num_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = new_lr

        inputs, targets = get_batch(
            train_data,
            cfg["batch_size"],
            cfg["context_length"],
            device=cfg["device"],
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = lm(inputs, token_positions)
            loss = cross_entropy(
                rearrange(logits, "b c v -> (b c) v"),
                rearrange(targets, "b c -> (b c)"),
            )

        loss.backward()
        gradient_clipping(lm.parameters(), cfg["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        elapsed = time.time() - start_time

        if t % 500 == 0:
            # compute val loss
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    total_val_loss = 0
                    for _ in range(3):
                        val_inputs, val_targets = get_batch(
                            val_data,
                            cfg["batch_size"],
                            cfg["context_length"],
                            device=cfg["device"],
                        )
                        val_logits = lm(val_inputs, token_positions)
                        total_val_loss += cross_entropy(
                            rearrange(val_logits, "b t v -> (b t) v"),
                            rearrange(val_targets, "b t -> (b t)"),
                        )
                    val_loss = total_val_loss / 3
            print(f"Step {t} | Train loss: {loss.item():.4f} | Val loss: {val_loss.item():.4f} | Time: {elapsed}")

    save_checkpoint(lm._orig_mod, optimizer, t, cfg["checkpoint_path"])
    # final val loss check
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0

        max_start = len(val_data) - cfg["context_length"] - 1
        starts = list(range(0, max_start + 1, cfg["context_length"]))

        for j in range(0, len(starts), cfg["batch_size"]):
            batch_starts = starts[j : j + cfg["batch_size"]]

            val_inputs = torch.stack(
                [
                    torch.tensor(val_data[s : s + cfg["context_length"]], dtype=torch.long, device=cfg["device"])
                    for s in batch_starts
                ]
            )
            val_targets = torch.stack(
                [
                    torch.tensor(val_data[s + 1 : s + cfg["context_length"] + 1], dtype=torch.long, device=cfg["device"])
                    for s in batch_starts
                ]
            )

            token_positions = torch.arange(val_inputs.shape[1], device=cfg["device"])
            val_logits = lm(val_inputs, token_positions)

            batch_loss = cross_entropy(
                rearrange(val_logits, "b t v -> (b t) v"), rearrange(val_targets, "b t -> (b t)")
            )

            n_tokens = val_targets.numel()
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens

        val_final_loss = total_loss / total_tokens
        print(f"Final val loss: {val_final_loss}")

    user_volume = modal.Volume.from_name("basics-alexskim")
    user_volume.commit()


@app.local_entrypoint()
def main():
    config = {
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "d_ff": 3072,
        "vocab_size": 32000,
        "context_length": 512,
        "rope_theta": 10000.0,
        "lr": 0.003,
        "lr_min": 1e-4,
        "warmup_steps": 500,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "num_steps": 30000,
        "batch_size": 128,
        "max_grad_norm": 1.0,
        "max_time_seconds": 2700,  # 45 minutes,
        "tie_embeddings": True,
        "use_relu": True,
        "device": "cuda",
        "architecture": "TransformerLM",
        "checkpoint_path": str(DATA_PATH / "checkpoints" / "final.pt"),
    }
    result = train.remote(config)
    print(result)


if __name__ == "__main__":
    main()
