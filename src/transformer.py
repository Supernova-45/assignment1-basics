import math
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor


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
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.W = torch.empty(size=(self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype)
        self.W = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor=self.W, mean=0, std=1, a=-3, b=3))

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
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else ((8 / 3 * d_model) // 64 + 1) * 64
        std = math.sqrt(2 / (d_model + d_ff))
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
    return expd / torch.sum(expd, dim=dim, keepdim=True)


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
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        std = math.sqrt(2 / (d_model + d_model))
        self.q_proj_weight = torch.nn.Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.q_proj_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.k_proj_weight = torch.nn.Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.k_proj_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.v_proj_weight = torch.nn.Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.v_proj_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.o_proj_weight = torch.nn.Parameter(torch.empty((d_model, d_model)))
        torch.nn.init.trunc_normal_(self.o_proj_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.rope = rope

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # apply causal masking
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()
        q = rearrange(x @ self.q_proj_weight.T, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = rearrange(x @ self.k_proj_weight.T, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = rearrange(x @ self.v_proj_weight.T, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        # apply rope
        if self.rope is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        attention = rearrange(scaled_dot_product_attention(q, k, v, mask), "... h seq_len d_k -> ... seq_len (h d_k)")

        return attention @ self.o_proj_weight.T


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.attention = Attention(d_model, num_heads, rope)

    def forward(self, x: Tensor, token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None):
        # first half
        norm1 = self.norm1.forward(x)
        first = x + self.attention.forward(norm1, token_positions)

        # second half
        norm2 = self.norm2.forward(first)
        second = first + norm2 * torch.sigmoid(norm2)

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
        rope: RotaryPositionalEmbedding | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])
        self.norm3 = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: Tensor, token_positions: Float[Tensor, "batch sequence_length d_model"] | None = None):
        # token embeddings
        x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x, token_positions)
        x = self.norm3.forward(x)
        return self.linear.forward(x)
