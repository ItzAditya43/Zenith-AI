from dataclasses import dataclass


@dataclass
class ZenithConfig:
    vocab_size: int = 8192
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2  # grouped-query attention
    hidden_dim: int = 1024  # SwiGLU intermediate size
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
