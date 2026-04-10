import math
import torch
from einops import rearrange


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.W1 = torch.nn.Parameter(torch.empty((self.d_ff, d_model)))
        self.W2 = torch.nn.Parameter(torch.empty((d_model, self.d_ff)))
        self.W3 = torch.nn.Parameter(torch.empty((self.d_ff, d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_vals = self.cos_vals[token_positions]
        sin_vals = self.sin_vals[token_positions]  
        x = rearrange(x, '... (pairs two) -> ... pairs two', two=2)
        x0 = x[..., 0]
        x1 = x[..., 1]
        new_x0 = x0 * cos_vals - x1 * sin_vals
        new_x1 = x0 * sin_vals + x1 * cos_vals
        return rearrange(torch.stack((new_x0, new_x1),dim=-1), '... pairs two -> ... (pairs two)')
        
    
