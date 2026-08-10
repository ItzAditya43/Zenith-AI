import torch


class KVCache:
    """Per-layer growing KV cache for autoregressive decoding."""

    def __init__(self, max_seq_len, n_kv_heads, head_dim, dtype, device, batch_size=1):
        self.k = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.seen_tokens = 0

    def update(self, k, v):
        bsz, n_kv_heads, seqlen, head_dim = k.shape
        start = self.seen_tokens
        end = start + seqlen
        self.k[:, :, start:end, :] = k
        self.v[:, :, start:end, :] = v
        self.seen_tokens = end
        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class ZenithKVCache:
    """Holds a KVCache per transformer layer."""

    def __init__(self, cfg, dtype, device, batch_size=1):
        self.layers = [
            KVCache(cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim, dtype, device, batch_size)
            for _ in range(cfg.n_layers)
        ]

    def __getitem__(self, idx):
        return self.layers[idx]

    @property
    def seen_tokens(self):
        return self.layers[0].seen_tokens
