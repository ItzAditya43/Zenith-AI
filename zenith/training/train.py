import argparse
import math
import os
import signal
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from zenith.model import ZenithConfig, ZenithTransformer
from zenith.data.dataset import PackedTokenDataset
from zenith.training.lr_schedule import cosine_with_warmup
from zenith.training.checkpoint import save_checkpoint, load_checkpoint


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp():
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def use_blocking_cuda_sync():
    """Make the CPU sleep instead of spin while waiting on the GPU.

    CUDA's default sync spin-waits, pinning one CPU core at 100% for the whole
    run. On this laptop the CPU and GPU share a heatpipe, and the spinning core
    alone pushed the CPU past the thermal watchdog's limit within a minute.
    Must run before the CUDA context exists.
    """
    import ctypes
    import glob
    libs = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib", "libcudart.so*"))
    libs += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*"))
    for lib in libs:
        try:
            rc = ctypes.CDLL(lib).cudaSetDeviceFlags(4)  # cudaDeviceScheduleBlockingSync
            return rc == 0
        except OSError:
            continue
    return False


class ThermalThrottle:
    """Pause between optimizer steps while the machine is hot.

    A watchdog that kills the run on a temperature limit costs a restart (and a
    torch.compile warm-up that heats the CPU again) every time it fires. Pausing
    *below* that limit keeps the run going at whatever duty cycle the cooling
    can sustain. Sensors are Linux hwmon chip names (e.g. "acpitz", "k10temp");
    the hottest temp1_input among them is used.
    """

    def __init__(self, sensors, pause_c, resume_c):
        import glob
        self.paths = []
        for hw in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                with open(os.path.join(hw, "name")) as f:
                    if f.read().strip() in sensors and os.path.exists(os.path.join(hw, "temp1_input")):
                        self.paths.append(os.path.join(hw, "temp1_input"))
            except OSError:
                continue
        self.pause_c, self.resume_c = pause_c, resume_c
        self.paused_s = 0.0

    def temp(self):
        temps = []
        for p in self.paths:
            try:
                with open(p) as f:
                    temps.append(int(f.read()) / 1000)
            except (OSError, ValueError):
                pass
        return max(temps) if temps else 0.0

    def maybe_pause(self, device, stop_flag):
        if not self.paths or self.temp() < self.pause_c:
            return
        if device.startswith("cuda"):
            torch.cuda.synchronize()  # let queued kernels finish so the GPU actually idles
        t0 = time.time()
        while self.temp() > self.resume_c and not stop_flag["flag"]:
            time.sleep(1.0)
        self.paused_s += time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    ddp = is_distributed()
    if ddp:
        rank, local_rank, world_size = setup_ddp()
        device = f"cuda:{local_rank}"
        is_master = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_master = True

    torch.manual_seed(cfg_dict.get("seed", 1337) + rank)

    cpu_threads = cfg_dict.get("training", {}).get("cpu_threads")
    if cpu_threads:
        torch.set_num_threads(cpu_threads)

    if cfg_dict.get("training", {}).get("blocking_sync") and device.startswith("cuda"):
        ok = use_blocking_cuda_sync()
        if is_master:
            print(f"CUDA blocking sync: {'on' if ok else 'unavailable'}")

    model_cfg = ZenithConfig(**cfg_dict["model"])
    model = ZenithTransformer(model_cfg).to(device)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.startswith("cuda")

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    elif cfg_dict.get("training", {}).get("compile") and device.startswith("cuda"):
        # fuses RMSNorm/RoPE/SwiGLU elementwise ops — worth ~1.3x on a small GPU
        model = torch.compile(model)

    if is_master:
        raw_model = model.module if ddp else getattr(model, "_orig_mod", model)
        print(f"Model params: {raw_model.num_params() / 1e6:.2f}M")

    train_cfg = cfg_dict["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["max_lr"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    train_ds = PackedTokenDataset(cfg_dict["data"]["train_path"])
    val_ds = PackedTokenDataset(cfg_dict["data"]["val_path"])

    sampler = None
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    # A dedicated, OS-entropy-seeded generator for shuffling — deliberately NOT
    # derived from the reproducible `seed` above. That seed is set fresh via
    # torch.manual_seed() on every process launch (including every resume),
    # which would otherwise reset the DataLoader's shuffle order to the exact
    # same permutation each time and replay the same early batches on every
    # restart instead of progressing through the dataset (caught via a
    # diverging train/val loss: train loss collapsed toward 0 while val
    # perplexity got worse, across ~20 watchdog-triggered restarts).
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int.from_bytes(os.urandom(8), "little"))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["micro_batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        generator=shuffle_generator if sampler is None else None,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["micro_batch_size"], shuffle=False, drop_last=True)
    # optional per-language validation sets ({lang: path}) — a mixed val loss can
    # hide one language lagging behind the others
    lang_val_loaders = {
        lang: DataLoader(PackedTokenDataset(path), batch_size=train_cfg["micro_batch_size"],
                         shuffle=False, drop_last=True)
        for lang, path in cfg_dict["data"].get("val_paths_by_lang", {}).items()
    }

    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    total_steps = train_cfg["total_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 100)
    max_lr = train_cfg["max_lr"]
    min_lr = train_cfg.get("min_lr", max_lr * 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_every = train_cfg.get("checkpoint_every", 500)
    eval_every = train_cfg.get("eval_every", 250)
    log_every = train_cfg.get("log_every", 10)
    step_sleep = train_cfg.get("step_sleep", 0.0)
    throttle = None
    if train_cfg.get("thermal_throttle"):
        tcfg = train_cfg["thermal_throttle"]
        throttle = ThermalThrottle(tcfg["sensors"], tcfg["pause_c"], tcfg["resume_c"])
        if is_master:
            print(f"Thermal throttle: pause at {tcfg['pause_c']}C, resume at {tcfg['resume_c']}C "
                  f"({len(throttle.paths)} sensors, now {throttle.temp():.0f}C)")
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    resume_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(resume_path):
        step = load_checkpoint(resume_path, model, optimizer, map_location=device)
        if is_master:
            print(f"Resumed from step {step}")

    scaler = torch.amp.GradScaler(enabled=use_amp and dtype == torch.float16)

    stop_requested = {"flag": False}

    def _request_stop(signum, frame):
        # Save-and-exit on the next safe point instead of dying mid-step, so a
        # watchdog (thermal/memory limit) or a manual Ctrl-C never loses more
        # than one optimizer step of progress.
        stop_requested["flag"] = True
        if is_master:
            print(f"\nReceived signal {signum} — will checkpoint and stop after the current step.")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < total_steps and not stop_requested["flag"]:
        lr = cosine_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(step)
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            sync_ctx = model.no_sync() if (ddp and micro_step < grad_accum_steps - 1) else _nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                     dtype=dtype, enabled=use_amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            loss_accum += loss.detach()  # no per-micro-step .item(): each one forces a GPU sync

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if step_sleep:
            # duty-cycle the GPU so a laptop's shared CPU/GPU cooling keeps up;
            # synchronize first, otherwise queued kernels keep the GPU busy
            # through the sleep and nothing cools
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            time.sleep(step_sleep)
        if throttle is not None:
            throttle.maybe_pause(device, stop_requested)

        if is_master and step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (train_cfg["micro_batch_size"] * grad_accum_steps * model_cfg.max_seq_len * log_every) / dt
            extra = f" | {throttle.temp():.0f}C, paused {throttle.paused_s / 60:.1f} min total" if throttle else ""
            print(f"step {step}/{total_steps} | loss {loss_accum.item():.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s{extra}")

        if is_master and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device, dtype, use_amp, max_batches=20)
            print(f"  [eval] step {step} | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            if lang_val_loaders:
                per_lang = {lang: evaluate(model, loader, device, dtype, use_amp, max_batches=10)
                            for lang, loader in lang_val_loaders.items()}
                print("  [eval] " + " | ".join(f"{lang} ppl {math.exp(v):.2f}" for lang, v in per_lang.items()))
            model.train()

        if is_master and step % ckpt_every == 0:
            save_checkpoint(resume_path, model, optimizer, step, model_cfg)
            save_checkpoint(os.path.join(ckpt_dir, f"step_{step}.pt"), model, optimizer, step, model_cfg)
            print(f"  [ckpt] saved at step {step}")

    if is_master:
        save_checkpoint(resume_path, model, optimizer, step, model_cfg)
        if stop_requested["flag"]:
            print(f"Stopped safely at step {step}/{total_steps} — checkpoint saved, rerun the same command to resume.")
        else:
            print("Training complete.")

    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.no_grad()
def evaluate(model, loader, device, dtype, use_amp, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                             dtype=dtype, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
