import os
import typing

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import numpy.typing as npt
import math

from cs336_basics.transformer import softmax, TransformerLM
from cs336_basics.tokenizer import Tokenizer


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


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    data = torch.load(src)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    return data["iteration"]


def decode_lm(
    model: TransformerLM,
    prompt_tokens,
    max_generated_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer: Tokenizer,
):
    decoded_text = ""
    num_generated = 0
    stop_token = tokenizer.encode("<|endoftext|>")[0]
    with torch.no_grad():
        while num_generated < max_generated_tokens:
            token_positions = torch.arange(prompt_tokens.shape[-1], device=prompt_tokens.device)
            logits = model.forward(prompt_tokens, token_positions)[:, -1, :]

            # apply temperature scaling
            probs = softmax(logits / temperature, -1)
            # apply nucleus / top p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sums = torch.cumsum(sorted_probs, dim=-1)
            mask = sums - sorted_probs >= top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled)
            prompt_tokens = torch.cat([prompt_tokens, next_token], dim=-1)
            decoded_text += tokenizer.decode([next_token.item()])
            num_generated += 1

            if next_token == stop_token:
                return decoded_text

    return decoded_text
