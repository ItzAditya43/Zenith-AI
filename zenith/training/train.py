import argparse
import math
import os
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

    model_cfg = ZenithConfig(**cfg_dict["model"])
    model = ZenithTransformer(model_cfg).to(device)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.startswith("cuda")

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    if is_master:
        raw_model = model.module if ddp else model
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

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["micro_batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["micro_batch_size"], shuffle=False, drop_last=True)

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
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    resume_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(resume_path):
        step = load_checkpoint(resume_path, model, optimizer, map_location=device)
        if is_master:
            print(f"Resumed from step {step}")

    scaler = torch.amp.GradScaler(enabled=use_amp and dtype == torch.float16)

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < total_steps:
        lr = cosine_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
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
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if is_master and step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = (train_cfg["micro_batch_size"] * grad_accum_steps * model_cfg.max_seq_len * log_every) / dt
            print(f"step {step}/{total_steps} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s")

        if is_master and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device, dtype, use_amp, max_batches=20)
            print(f"  [eval] step {step} | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            model.train()

        if is_master and step % ckpt_every == 0:
            save_checkpoint(resume_path, model, optimizer, step, model_cfg)
            save_checkpoint(os.path.join(ckpt_dir, f"step_{step}.pt"), model, optimizer, step, model_cfg)
            print(f"  [ckpt] saved at step {step}")

    if is_master:
        save_checkpoint(resume_path, model, optimizer, step, model_cfg)
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
