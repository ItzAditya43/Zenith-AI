import os
import torch


def save_checkpoint(path, model, optimizer, step, cfg, extra=None):
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }
    if extra:
        payload.update(extra)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("step", 0)
