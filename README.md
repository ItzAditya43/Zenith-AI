# Zenith AI

A small language model, and the full stack around it — tokenizer, transformer, training loop, inference engine, and CLI — built and trained entirely from scratch. No pretrained weights, no external model APIs. This repo is both the implementation and a report on what it took to get a from-scratch LLM talking on a single consumer laptop GPU.

## Why this exists

Most "build an LLM" exercises stop at wiring together a HuggingFace `Trainer` and someone else's weights. The point here was different: implement every layer of the stack by hand — attention, positional encoding, normalization, the optimizer schedule, the KV cache, the sampler — well enough that a real checkpoint comes out the other end and produces text you can hold a conversation with. The goal wasn't to compete with GPT-class models; it was to understand, and prove, that every piece works by actually training something and reading its output.

## What's in the box

```
zenith/
├── tokenizer/     BPE tokenizer training (byte-level, trained from scratch)
├── data/          Corpus download + tokenize-and-pack pipeline
├── model/         RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, transformer
├── training/      AdamW, cosine LR schedule, AMP, gradient accumulation, checkpointing, DDP hook
├── inference/      Autoregressive generation with KV cache, temperature/top-k/top-p sampling
├── evaluation/     Held-out loss / perplexity
└── cli.py          `zenith train|chat|evaluate|tokenize`
```

## Architecture

A decoder-only, LLaMA-style transformer:

- **Tokenizer**: byte-level BPE, vocab size 8192, trained from scratch on the target corpus (no pretrained vocab).
- **Positional encoding**: RoPE (rotary position embeddings) applied per-head at attention time — no learned/absolute position table.
- **Normalization**: RMSNorm, pre-norm residual blocks.
- **Attention**: multi-head causal self-attention with **grouped-query attention** (6 query heads, 2 KV heads) via `scaled_dot_product_attention`.
- **MLP**: SwiGLU feed-forward.
- **Weight tying**: input embedding and output projection share weights.
- **KV cache**: per-layer growing cache for O(1)-per-token autoregressive decoding, verified numerically identical to a full non-cached forward pass (see Findings).
- **Training**: AdamW (β=0.9/0.95), cosine LR schedule with linear warmup, gradient clipping, gradient accumulation, bf16 mixed precision via autocast, checkpoint/resume, and a DDP code path (untested here — single-GPU machine — but wired in via `torch.distributed`/`DistributedSampler`).
- **Inference**: greedy / temperature / top-k / top-p sampling, streaming token-by-token generation.

## What it was trained on

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — a synthetic corpus of short, simple children's stories, chosen specifically because a tiny model (single-digit millions of parameters) can learn coherent grammar and narrative structure from it in a short training run, unlike a general web-text corpus which needs orders of magnitude more scale to produce readable output.

## Findings

**Hardware**: single RTX 3050 Laptop GPU, 4GB VRAM.

**Model size**: 12.59M parameters (8 layers, dim 384, 6 attention heads / 2 KV heads, hidden dim 1024, 256-token context) — deliberately small to fit comfortably in 4GB with room for batching, and to make a full training run finish in about 90 minutes rather than days.

**Run**: 6000 steps, effective batch size 128 sequences × 255 tokens ≈ 96M training tokens total, throughput ~37k tokens/sec, wall-clock ~90 minutes.

**Result**: validation perplexity dropped from ~1600 (near-random, at step 10) to **5.05** at step 6000. Sample output at convergence:

> Once upon a time, there was a little girl named Lily. She had a favorite toy, a teddy bear named Teddy. One day, Lily was playing with Teddy when she accidentally broke his favorite toy. She felt sad and didn't know what to do. Her mom saw her sad face and asked her what happened...

Grammatically coherent, holds a consistent character and plot thread across multiple sentences, without ever having seen a pretrained embedding.

**Correctness check that mattered most**: the KV cache is the single easiest place to introduce a silent bug in a from-scratch transformer (wrong batch dimension, wrong position offset, cache written before or after the causal-mask decision). Before trusting any generated output, the incremental (cached, one-token-at-a-time) forward pass was checked against the full non-cached forward pass on identical input — outputs matched to float32 precision (`max diff ≈ 1.2e-7`). Only after that passed was inference trusted.

**Bug caught by that check**: the initial KV cache implementation hardcoded batch size 1, which is invisible in single-sequence chat but breaks silently (wrong-shape tensor write) the moment you try batched generation or the equivalence test above. Fixed by threading `batch_size` through `KVCache` → `ZenithKVCache` → `model.new_kv_cache()`.

**Infrastructure issue, not a model issue**: mid-setup, `pip install torch` (~500MB wheel) kept failing at the exact same byte offset on every retry. Turned out `/tmp` (a size-limited tmpfs) was 100% full — not from this project, but ~6.5GB of orphaned temp directories left behind by unrelated crashed processes (stale PyInstaller `_MEI*` and AppImage extraction dirs). Verified with `fuser` that nothing currently running held them open, cleared them, and the install succeeded immediately after. Worth knowing about if you're setting this up somewhere else and installs mysteriously stall at a fixed byte count.

## What "at scale" would actually take

This repo intentionally trains a small model on a laptop GPU in about 90 minutes as a proof that the full stack works end-to-end. A genuinely capable ~125M-parameter chat model trained the same way — same code, bigger config — needs hundreds of billions of tokens and multi-GPU, multi-day compute; nothing about the architecture changes, only the config (`configs/zenith_small.yaml`) and how long you let it run.

## Training on Kaggle

The local run above (12.59M params, 4GB laptop GPU) is a proof that the stack works, not the ceiling. `configs/zenith_kaggle.yaml` scales the same architecture up to **75.5M params** (12 layers, dim 768, 512-token context) sized for a Kaggle T4/P100 (16GB). `kaggle/zenith_kaggle_train.ipynb` clones this repo, installs deps, runs the full data → tokenizer → pack → train → generate pipeline, and checkpoints to `/kaggle/working/checkpoints` every 500 steps so a run can be resumed across Kaggle's 12-hour session cap. Upload the notebook to Kaggle (GPU accelerator + internet on) and run it top to bottom.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get data and train a tokenizer
python -m zenith.data.prepare dump --output-dir data
python -m zenith.tokenizer.train_tokenizer --input data/train.txt --output data/tokenizer.json --vocab-size 8192

# 2. Tokenize + pack into training shards
python -m zenith.data.prepare pack --input data/train.txt --tokenizer data/tokenizer.json --output data/train_packed.npy --seq-len 256
python -m zenith.data.prepare pack --input data/val.txt   --tokenizer data/tokenizer.json --output data/val_packed.npy   --seq-len 256

# 3. Train
python -m zenith.training.train configs/zenith_small.yaml

# 4. Chat with your own weights
python -m zenith.cli chat checkpoints/latest.pt --tokenizer data/tokenizer.json

# 5. Evaluate
python -m zenith.cli evaluate checkpoints/latest.pt --val-path data/val_packed.npy
```
