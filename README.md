# Zenith AI

A small language model, and the full stack around it — tokenizer, transformer, training loop, inference engine, and CLI — built and trained entirely from scratch. No pretrained weights, no external model APIs. This repo is both the implementation and a report on what it took to get a from-scratch LLM talking on a single consumer laptop GPU.

It now also contains **Zenith Keyboard**: a 17.8M-parameter English/Spanish/Japanese model trained from scratch to power next-word prediction, autocorrect and kanji choice in an Android keyboard, exported to an 18.3 MB int8 ONNX file (see [Zenith Keyboard](#zenith-keyboard-an-on-device-model-for-a-phone-keyboard)).

## Why this exists

Most "build an LLM" exercises stop at wiring together a HuggingFace `Trainer` and someone else's weights. The point here was different: implement every layer of the stack by hand — attention, positional encoding, normalization, the optimizer schedule, the KV cache, the sampler — well enough that a real checkpoint comes out the other end and produces text you can hold a conversation with. The goal wasn't to compete with GPT-class models; it was to understand, and prove, that every piece works by actually training something and reading its output.

## What's in the box

```
zenith/
├── tokenizer/     Tokenizer training from scratch: byte-level BPE, or Unigram + byte fallback (multilingual)
├── data/          Corpus download + tokenize-and-pack (TinyStories; keyboard_corpus.py for EN/ES/JA)
├── model/         RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, transformer
├── training/      AdamW, cosine LR schedule, AMP, gradient accumulation, checkpointing, DDP hook
├── inference/      Autoregressive generation with KV cache, temperature/top-k/top-p sampling
├── evaluation/     Held-out loss / perplexity; keyboard_eval.py (next-word, prefix, keystroke savings, kanji choice)
├── export/         ONNX export with explicit KV cache, mixed int8 quantization, parity test, phone tokenizer
└── cli.py          `zenith train|chat|evaluate|tokenize`
export/             Shipped keyboard model, tokenizer, eval results, MyKey integration guide (PDF)
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

A second Kaggle run (version 6, actually pulling the fixed 100.7M/16-layer config this time — see the mistake documented above) was launched and left training unattended. Its outcome isn't recorded here: the session tracking it went stale for an extended period, and by the time work resumed the short-lived Kaggle API token used to check on it had long expired. Rather than fight Kaggle's auth flow again, the decision was to move future runs to Google Colab instead — its outcome can be checked directly at [kaggle.com/code/adityapandey444/zenith-ai-kaggle-train](https://www.kaggle.com/code/adityapandey444/zenith-ai-kaggle-train) if you have access to that account, but this repo doesn't depend on it.

## Training on Google Colab

`configs/zenith_colab.yaml` and `colab/zenith_colab_train.ipynb` train the same ~100.7M-param architecture on a free Colab GPU (T4 by default). It's structurally the same pipeline as the Kaggle notebook (GPU-compatibility check, data pipeline, train, generate), with two differences that matter:

**No launch/monitor API.** Kaggle's `kaggle kernels push` let this process launch a run and poll its status without a human in the loop. Colab's free tier has no equivalent — this notebook has to be opened in a browser and run manually (`Runtime` → `Run all`), and checking on it means checking the tab, not asking an agent to query an API. If you're delegating this to an assistant, expect to be the one reporting back "is it done" rather than the other way around.

**Google Drive instead of the checkpoint-dataset dance.** Kaggle's `/kaggle/working` is wiped every session, so resuming a cut-off run meant manually saving checkpoints as a Kaggle Dataset and reattaching it as input on the next run — see the "Optional: resume" cell in the Kaggle notebook. Colab's local disk is equally ephemeral, but mounting Google Drive (`colab/zenith_colab_train.ipynb`'s second cell) and pointing `checkpoint_dir` and the data paths at `/content/drive/MyDrive/zenith_ai/` in `configs/zenith_colab.yaml` sidesteps that entirely — rerunning the notebook in a brand new session just finds `latest.pt` and the already-packed dataset sitting there and resumes/skips automatically, no manual reattach step.

Everything else learned from the Kaggle runs — size `total_steps` to the throughput actually measured on whatever GPU you land on rather than an assumed best case, don't trust `torch.cuda.is_available()` alone, push config changes to GitHub *before* launching a run that clones from GitHub — applies here too, and the notebook already accounts for all of it.

## Training locally: hardware safety and a real correctness bug

`configs/zenith_local_100m.yaml` trains the same ~100.7M architecture directly on a local GPU — in this case a 4GB laptop card, which needs real engineering to use safely for an unattended multi-hour run, not just a smaller batch size.

**Fitting 100.7M params in 4GB.** Static optimizer state alone (params + gradients + AdamW's two momentum buffers, all fp32) is ~1.6GB before any activations. `grad_checkpoint: true` on `ZenithConfig` (implemented via `torch.utils.checkpoint` around each `TransformerBlock`, verified to produce bit-identical gradients to the non-checkpointed path before trusting it) trades ~30% slower steps for far lower activation memory, measured via a live smoke test at ~2.3GB peak VRAM — comfortable margin, not a guess.

**A closed laptop lid caused a full unclean machine crash, not just a training failure.** Partway through the first long run, the machine hard-rebooted with no shutdown sequence logged at all — the log just stops. Investigation (`journalctl`, `last -x`) found the lid had been closed while training was running at full GPU load; this machine has `HandleLidSwitch=ignore` configured, so the OS kept computing with the lid shut, blocking chassis airflow, until hardware-level thermal protection cut power directly. The GPU die itself never got dangerously hot (`nvidia-smi` showed a safe 55-74°C the whole time) — the failure was in the CPU/chassis/NVMe, which nothing was monitoring. `scripts/thermal_watchdog.py` (not part of the portable project — hardcoded to this machine's `sensors` output) fixes this: it polls CPU package temp, chassis temp, NVMe temp, fan RPM, and the GPU die, and treats a closed lid as an *immediate* stop signal rather than waiting for a temperature to catch up. `zenith/training/train.py` gained a `SIGTERM`/`SIGINT` handler that checkpoints before exiting, so a safety stop never loses more than the current step.

**Single-reading thermal checks trip on harmless restart spikes.** A cold process start jumping straight to 100% GPU+CPU load produces a brief temperature spike that outpaces the fan curve, before settling to the same safe range the machine held for 2+ hours in the original run. The watchdog now requires a threshold breach to persist across `CONSECUTIVE_BREACHES_TO_TRIP` consecutive polls before stopping (lid-closed and fan-stall stay as instant triggers — those are unambiguous, nothing to debounce). `scripts/autoresume_train.py` closes the remaining loop: wait for the chassis to cool below a resume threshold, relaunch, repeat, until the run actually completes or a real error (not a routine safety stop) appears — turning "someone needs to notice and relaunch" into "just works overnight."

**The real bug: every restart replayed the same training batches.** With the safety system now cycling the process every few minutes on this chassis, a bug that would've been invisible in one continuous run became obvious: `train.py` calls `torch.manual_seed(seed)` at the start of every process launch for reproducible model init, but the training `DataLoader`'s shuffle order was drawing from that same global RNG — so every restart reset the shuffle to the exact same permutation and replayed the same early batches instead of progressing through the dataset. Caught by watching train/val loss diverge: training loss collapsed toward 0 (the model trivially memorizing a small repeated slice) while validation perplexity got *worse* over the same steps (19.0 → 24.2) — the textbook overfitting signature, except the "overfitting" was actually overfitting to a bug, not real training progress. Fixed by giving the `DataLoader` its own `torch.Generator` seeded from OS entropy (`os.urandom`), decoupled from the reproducible seed used for model initialization — verified directly (two simulated "process restarts" with the same fixed seed now draw different shuffle orders) before trusting it. The corrupted checkpoint was discarded in favor of the last one saved before the restart cycling began.

## Zenith Keyboard: an on-device model for a phone keyboard

**Goal** (from the plan): a model small enough to run offline inside the MyKey Android keyboard that understands the whole sentence, not just the last word. Budget: at most 20 MB, at most 30 ms per prediction, at most 70 MB of RAM, 64-token context. Languages: English, Spanish and Japanese, with Russian and Mandarin token slots reserved.

**Model.** 17.8M parameters (7 layers, dim 320, 5 query heads / 1 KV head, SwiGLU 864, RoPE, tied 32k embedding), `configs/zenith_keyboard.yaml`.

**Data** (`zenith/data/keyboard_corpus.py`). Sources: OpenSubtitles (casual dialogue), Tatoeba, Wikipedia first paragraphs, and CC-100 for informal Japanese. Cleaning: NFKC normalisation, then dropping URLs, digit-heavy lines and lines over 200 characters. The result is 911M unique tokens, mixed to 1.34B training tokens at 35% English, 30% Spanish and 35% Japanese (the smaller corpora are repeated). Every 64-token row starts with `<en>`/`<es>`/`<ja>`, because that's exactly what the keyboard sends.

**Tokenizer.** Unigram + Metaspace + byte fallback + NFKC, 32k vocabulary. Byte-level BPE splits on spaces, and Japanese has none. Results: 1.36 tokens/word for English, 1.47 for Spanish, and 1.9 to 2.2 characters/token for Japanese, with no `<unk>`.

**Training.** 40,000 steps × 32,768 tokens with bf16 and `torch.compile` (about 43k tok/s on the RTX 3050). Final validation perplexity: English 47.0, Spanish 50.2, Japanese 117.6.

**Keyboard eval** (`zenith/evaluation/keyboard_eval.py`, held-out subtitle sentences, 150 per language). The int8 column scores the shipped ONNX file itself.

| Metric | Zenith fp32 | Zenith int8 | Word bigram | Plan target |
|---|---|---|---|---|
| English next-word top-3 | 36.7% | 36.6% | 20.8% | ≥ 35% ✅ |
| English prefix top-3 (1-3 letters typed) | 74.8% | 74.7% | 62.7% | ≥ 70% ✅ |
| English keystroke savings | 53.0% | 53.2% | 44.9% | |
| Spanish next-word top-3 | 34.2% | | 19.1% | ≥ 30% ✅ |
| Spanish prefix top-3 | 68.2% | | 56.1% | ≥ 70% (just missed) |
| Japanese kanji choice top-1 (Mozc candidates) | 81.0% | | 88.0% | beat Mozc order 59.6% ✅ |
| Japanese next-phrase top-3 | 41.2% | | 43.8% | |

Zenith clearly beats the bigram baseline on English and Spanish. On Japanese it beats Mozc's own candidate order, but not an in-domain word bigram, so on the phone it is meant to be *combined* with Mozc cost and user history rather than to replace them.

**Export** (`zenith/export/`). The result is `export/zenith_kb.onnx` at **18.3 MB**, about 2 ms per keystroke step on a PC CPU, with the KV cache as explicit inputs/outputs. int8 top-1 agrees with fp32 on 94 to 95% of positions, and the fp32 top-1 is in the int8 top-3 99.5% of the time. The stricter identical-top-5 test is 76 to 80%. `export/ZENITH_MYKEY_INTEGRATION.pdf` is the integration guide for the Android side.

Things that went wrong and how they were fixed:

- **Naive int8 broke the trained model.** ONNX Runtime dynamic quantization gave under 5% top-5 parity, although the same pipeline had looked fine on an early checkpoint. Quantizing one layer group at a time showed the SwiGLU down-projections were the cause (input max/median about 450×; quantizing them alone dropped logit correlation to 0.45). Fix: the transformer layers use int8 *weight-only* quantization (`DequantizeLinear` into an fp32 MatMul). The 32k output head stays dynamic int8, after **SmoothQuant** (alpha 0.7) flattens its input outliers. That took Japanese parity from 34% to 76%.
- **Tied embeddings were stored twice.** The exporter wrote the embedding table and a transposed copy for the head, giving 28.5 MB. The head's per-output-channel scales are exactly per-token embedding scales, so after quantization the embedding lookup is rebuilt as a `Gather` on the head's int8 matrix plus a 1/s multiply to undo SmoothQuant. That brought it to 18.3 MB.
- **Token healing.** Japanese next-phrase predictions were poor whenever the context ended inside a token (「今日」, when training always saw 「今日は」 as one piece). Dropping the last context token and constraining the continuation to re-spell it took test cases from 2 of 9 to 6 of 9 hits. The phone side needs the same trick.
- **Laptop thermals.** On the balanced power profile the CPU passed the watchdog's limits within 30 seconds, and each restart re-ran `torch.compile` and heated it up again. The fixes were the power-saver profile (which leaves GPU speed unchanged), blocking CUDA sync (so the CPU stops spin-waiting), and an in-loop `thermal_throttle` that pauses between steps at 74°C and resumes at 68°C. With those, the run never tripped the watchdog. Later, a closed lid exposed a gap in auto-resume: it relaunched into a closed lid every 25 seconds. It now waits for the lid to open.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get data and train a tokenizer
python -m zenith.data.prepare dump --output-dir data
python -m zenith.tokenizer.train_tokenizer --model bpe --input data/train.txt --output data/tokenizer.json --vocab-size 8192

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

### Zenith Keyboard pipeline

```bash
python -m zenith.data.keyboard_corpus download        # ~2.3 GB into data_kb/raw (parallel, resumable)
python -m zenith.data.keyboard_corpus clean
python -m zenith.tokenizer.train_tokenizer --input data_kb/tok_sample/*.txt --output data_kb/tokenizer_kb.json   # Unigram, 32k
python -m zenith.data.keyboard_corpus pack            # -> data_kb/train.npy, val_{en,es,ja}.npy
python -m zenith.training.train configs/zenith_keyboard.yaml
python -m zenith.evaluation.keyboard_eval checkpoints_kb/latest.pt --output export/keyboard_eval.json
python -m zenith.export.export_onnx checkpoints_kb/latest.pt   # fp32 ONNX + SmoothQuant head
python -m zenith.export.quantize                               # -> export/zenith_kb.onnx
python -m zenith.export.export_tokenizer                       # -> export/zenith_kb.tok
python -m zenith.export.parity_test checkpoints_kb/latest.pt
python -m zenith.evaluation.keyboard_eval checkpoints_kb/latest.pt --onnx export/zenith_kb.onnx   # eval the shipped file
```

`data_kb/tok_sample/` (made by hand for this run; there is no command for it yet) is a balanced ~110 MB random sample of the cleaned text (about 30 MB each of English and Spanish, 50 MB of Japanese). Keep Japanese at 40% or more, or its pieces degrade to single characters.
