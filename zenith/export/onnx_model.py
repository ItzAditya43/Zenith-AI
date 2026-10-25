"""Export-friendly forward pass with the KV cache as explicit inputs/outputs.

The phone keeps a fixed 64-slot cache per layer and passes it back in on every
keystroke. Writes into the cache and the causal mask are expressed as one-hot
matmuls / comparisons instead of in-place slice assignment, so the graph has no
data-dependent shapes and exports cleanly to ONNX opset 17.

  inputs   tokens   int64 (1, T)              new tokens (T >= 1)
           pos      int64 (1,)                position of tokens[0] (= tokens already cached)
           past_k   float (L, 1, KV, W, D)    W = max_seq_len
           past_v   float (L, 1, KV, W, D)
  outputs  logits   float (1, T, vocab)
           new_k, new_v  same shape as past_k/past_v

When pos + T would exceed W, the caller resets (pos=0, zero cache) and re-feeds
the last ~48 tokens with the language tag — RoPE positions are absolute, so a
sliding window can't just drop the oldest slot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from zenith.model.layers import apply_rope


class ZenithONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.cfg = model.cfg

    def cache_shape(self, batch=1):
        c = self.cfg
        return (c.n_layers, batch, c.n_kv_heads, c.max_seq_len, c.head_dim)

    def forward(self, tokens, pos, past_k, past_v):
        cfg = self.cfg
        m = self.m
        bsz, T = tokens.shape
        W = cfg.max_seq_len
        positions = pos + torch.arange(T, device=tokens.device)          # (T,)
        cos = m.rope_cos[positions]                                       # (T, D/2)
        sin = m.rope_sin[positions]
        slots = torch.arange(W, device=tokens.device)
        write = (slots[None, :] == positions[:, None]).to(past_k.dtype)    # (T, W)
        keep = 1.0 - write.sum(0)                                         # (W,)
        allowed = slots[None, :] <= positions[:, None]                    # (T, W)
        neg = torch.zeros(T, W, dtype=past_k.dtype, device=tokens.device).masked_fill(~allowed, -1e9)

        x = m.tok_embeddings(tokens)
        new_ks, new_vs = [], []
        for i, layer in enumerate(m.layers):
            a = layer.attn
            h = layer.attn_norm(x)
            q = a.wq(h).view(bsz, T, a.n_heads, a.head_dim).transpose(1, 2)
            k = a.wk(h).view(bsz, T, a.n_kv_heads, a.head_dim).transpose(1, 2)
            v = a.wv(h).view(bsz, T, a.n_kv_heads, a.head_dim).transpose(1, 2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            # scatter new k/v into their slots: cache * keep + onehot^T @ new
            ck = past_k[i] * keep[None, None, :, None] + torch.matmul(write.t(), k)
            cv = past_v[i] * keep[None, None, :, None] + torch.matmul(write.t(), v)
            new_ks.append(ck)
            new_vs.append(cv)
            kk = ck.repeat_interleave(a.n_rep, dim=1) if a.n_rep > 1 else ck
            vv = cv.repeat_interleave(a.n_rep, dim=1) if a.n_rep > 1 else cv
            att = torch.matmul(q, kk.transpose(-1, -2)) * (a.head_dim ** -0.5) + neg
            out = torch.matmul(F.softmax(att, dim=-1), vv)
            out = out.transpose(1, 2).reshape(bsz, T, a.n_heads * a.head_dim)
            x = x + a.wo(out)
            x = x + layer.mlp(layer.mlp_norm(x))
        logits = m.output(m.norm(x))
        return logits, torch.stack(new_ks), torch.stack(new_vs)
