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

The local run above (12.59M params, 4GB laptop GPU) is a proof that the stack works, not the ceiling. `configs/zenith_kaggle.yaml` scales the same architecture up for a Kaggle T4/P100 (16GB), currently at **~100.7M params** (16 layers, dim 768, 512-token context). `kaggle/zenith_kaggle_train.ipynb` clones this repo, installs deps, runs the full data → tokenizer → pack → train → generate pipeline, and checkpoints to `/kaggle/working/checkpoints` every 250 steps so a run can be resumed across Kaggle's session cap. Push it with `kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4` (or upload manually and run top to bottom with Accelerator = GPU, Internet = On).

### What running this on free-tier Kaggle actually looked like

This section exists because the failures were as instructive as the successes, and because "just run it on Kaggle" undersells how many free-tier-specific things had to be found and fixed before a run actually produced a usable model.

**Accelerator selection is easy to get wrong silently.** `kaggle kernels push --accelerator GPU_T4X2` looks like a reasonable flag but isn't a real value — the Kaggle API only accepts `NvidiaTeslaT4`, `NvidiaTeslaP100`, or `Tpu1VmV3` (found by reading the `kagglesdk` source, not the docs). An invalid value is silently ignored rather than rejected, so two pushes in a row landed on a P100 anyway.

**Free-tier P100s can be incompatible with the preinstalled PyTorch build.** Kaggle still offers Tesla P100 (compute capability sm_60, Pascal-era) as an accelerator, but newer PyTorch builds have dropped sm_60 support entirely. `torch.cuda.is_available()` still returns `True` on a mismatched build — it's only real ops that fail. This crashed the kernel process within about 30 seconds of starting, silently resetting all Python/notebook state. The notebook now checks `nvidia-smi --query-gpu=compute_cap` and reinstalls a CUDA-11.8-era torch build if the assigned GPU is older than compute capability 7.0, plus does an actual `torch.randn(...) @ torch.randn(...)` matmul smoke test rather than trusting `is_available()` alone.

**A crashed kernel process resets more than you'd expect.** The above P100 crash didn't just kill training — it reset `%cd`, so every subsequent cell's `!python -m zenith...` ran from `/kaggle/working` instead of the cloned repo, producing `ModuleNotFoundError` on everything. Fix: every shell command in the notebook now does `!cd {REPO} && python -m ...` inline rather than relying on notebook-wide working-directory state surviving the whole run.

**The real bug, though, was in this repo's `.gitignore`, not Kaggle.** `data/` (no leading slash) was meant to exclude the local dataset folder, but git treats an unanchored pattern as matching a directory of that name *anywhere* in the tree — so it was also silently excluding `zenith/data/`, the actual source module with the dataset-download and packing code. This worked completely invisibly on the machine that wrote the code (the files existed on disk, just untracked), and only broke the moment Kaggle did a fresh `git clone` and got a repo missing an entire module. Fixed by anchoring the pattern (`/data/`) and adding explicit `__init__.py` to every subpackage instead of relying on implicit namespace packages, which had been masking the gap. Lesson: a from-scratch project passing every local test tells you nothing about whether `git clone` on a clean machine reproduces the same tree — worth checking directly (`git ls-files | grep ...`) rather than assuming.

**Free-tier throughput can be an order of magnitude worse than a laptop GPU, with no error to point at.** The local RTX 3050 run hit ~37k tok/s on a 12.59M model. The first real Kaggle T4 run measured a steady ~3,780 tok/s on a 75.5M model — roughly a 10x-per-token gap after accounting for the model being ~6x bigger. Nothing in the code changed between the two; the likely causes are a shared/throttled free-tier GPU allocation and Kaggle's papermill/`debugpy`-instrumented execution environment adding Python-level overhead the local run didn't have. At that rate, a 3000-step/195M-token run projected to ~14.4 hours — well past any single Kaggle session (~9-12h cap) — so it got forcibly cut off around step ~1500-2000 rather than erroring, which looks identical to a hang unless you know to expect it. **Takeaway applied to later runs**: size `total_steps` to the *measured* throughput of the specific session, not to what the model or an assumed A100-class GPU could theoretically do — the config now targets ~6.4h for the 100.7M run based on the throughput actually measured on the previous one, rather than guessing.

**Kaggle's 2-concurrent-batch-session cap bites when a stale run won't die.** Pushing a fixed version doesn't cancel an already-running version of the same kernel — both can run at once, each holding one of your 2 available batch-GPU slots, and `kaggle kernels status` only reports the *latest* version, so a stale earlier version can keep running invisibly and block new pushes with "Maximum batch GPU session count of 2 reached" until it's cancelled from the Kaggle UI (no reliable CLI/API path was found to cancel a specific running session by kernel ref alone).

**Kaggle's short-lived `KGAT_...` API tokens are not a stable credential for a run that outlives ~30-60 minutes.** They're bearer-token-style and expire; monitoring silently falls back to whatever `kaggle.json` happens to exist on disk if one is present, which — if it's for the wrong Kaggle account — fails with a misleading `Permission 'kernels.get' was denied` / "wrong kernel slug" error that has nothing to do with the actual slug. The durable fix is a permanent legacy API key (`kaggle.json`, from Kaggle Settings → API → **Create New Token**, the classic flow, not the new token dialog) generated **while logged into the account that owns the kernel** — easy to get wrong if you have more than one Kaggle account, which produced a `Permission denied` red herring that looked like an auth bug but was actually the right key for the wrong account.

**None of this affects the trained model's quality** — it's entirely about getting a training job to survive a free, shared, session-capped, timeout-prone environment. The architecture, tokenizer, and training loop are the same code validated locally; what changed is everything around getting it to run unattended for hours on infrastructure designed to be interrupted.

**One more, self-inflicted: pushing a Kaggle kernel doesn't push your local file edits.** The notebook does `git clone` from GitHub at the start of every session — it has no idea about uncommitted local changes. Editing `configs/zenith_kaggle.yaml` locally to go from 12 to 16 layers, then running `kaggle kernels push` *before* `git push`-ing that change to GitHub, silently retrained the old 75.5M config a second time instead of the intended 100.7M one. It wasn't wasted, though: this time the 75.5M run actually completed all 1000 steps without getting cut off (see results below) — but the lesson stands: for a notebook-clones-from-git workflow, the GitHub push has to land *before* the Kaggle push, not after or in parallel.

### Results: 75.5M params, full 1000-step run (completed)

Val loss / perplexity, cleanly decreasing the whole way:

| Step | val_loss | perplexity |
|---|---|---|
| 250 | 2.9444 | 19.00 |
| 500 | 2.1682 | 8.74 |
| 750 | 1.8933 | 6.64 |
| 1000 | 1.7714 | **5.88** |

Sample generations at the final checkpoint:

> **Once upon a time**, there was a little girl named Lily. She loved to play with her toys and eat candy. One day, Lily's mom asked her to help clean up her toys. Lily was excited to help, but she didn't want to stop playing. "I don't want to clean up," said Lily. "I want to play more," said her mom. Lily didn't want to clean up, so she got upset. "Why can't I play with your toys?" she...

> **Tom and Lily went to the park**. They saw a big slide. Tom wanted to go on the slide. He asked Lily to let him go. Lily said no. They were scared. Tom said, "Don't be a baby. I will go first. I will help you." Lily said, "I will go first." She put on her shoes. She followed Tom to the slide. They went down the slide together. They were happy.

Coherent dialogue, consistent character names across the whole generation, plausible (if simple) plot logic — a genuinely finished, working checkpoint. This is the first run in this whole process that completed a full training schedule on Kaggle without being cut off by the session timeout, crashing on a GPU mismatch, or failing to find its own source code.

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
