import torch
import torch.nn as nn

from .layers import RMSNorm, TransformerBlock, precompute_rope
from .kv_cache import ZenithKVCache


class ZenithTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.tok_embeddings.weight.numel()
        return n

    def forward(self, tokens, kv_cache: ZenithKVCache = None, start_pos: int = 0):
        bsz, seqlen = tokens.shape
        x = self.tok_embeddings(tokens)

        cos = self.rope_cos[start_pos:start_pos + seqlen].to(x.device)
        sin = self.rope_sin[start_pos:start_pos + seqlen].to(x.device)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = layer(x, cos, sin, kv_cache=layer_cache)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    def new_kv_cache(self, dtype=None, device=None, batch_size=1):
        dtype = dtype or next(self.parameters()).dtype
        device = device or next(self.parameters()).device
        return ZenithKVCache(self.cfg, dtype, device, batch_size)
