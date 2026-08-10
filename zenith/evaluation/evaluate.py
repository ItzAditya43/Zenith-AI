import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zenith.model import ZenithConfig, ZenithTransformer
from zenith.data.dataset import PackedTokenDataset
from zenith.training.checkpoint import load_checkpoint


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ZenithConfig(**payload["config"])
    model = ZenithTransformer(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    ds = PackedTokenDataset(args.val_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    loss, ppl = run_eval(model, loader, device)
    print(f"val_loss={loss:.4f}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
