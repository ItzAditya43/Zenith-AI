"""Export a Zenith checkpoint to ONNX (fp32) with an explicit KV cache."""
import argparse

import torch

from zenith.model import ZenithConfig, ZenithTransformer
from zenith.training.checkpoint import load_checkpoint
from zenith.export.onnx_model import ZenithONNX


def load_fp32(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ZenithTransformer(ZenithConfig(**payload["config"]))
    load_checkpoint(checkpoint, model)
    return model.eval()


@torch.no_grad()
def check_against_pytorch(model, wrapper):
    """The wrapper (prefill then one cached step) must match the plain forward."""
    ids = torch.randint(10, model.cfg.vocab_size, (1, 12))
    ref = model(ids)
    k = torch.zeros(wrapper.cache_shape())
    v = torch.zeros(wrapper.cache_shape())
    lg1, k, v = wrapper(ids[:, :11], torch.tensor([0]), k, v)
    lg2, _, _ = wrapper(ids[:, 11:], torch.tensor([11]), k, v)
    got = torch.cat([lg1, lg2], dim=1)
    diff = (got - ref).abs().max().item()
    print(f"wrapper vs PyTorch forward: max |diff| = {diff:.2e}")
    assert diff < 1e-3, "export wrapper diverges from the model"


@torch.no_grad()
def smooth_output_head(model, data_dir, alpha=0.7, rows=64):
    """SmoothQuant for the tied output head.

    The final-norm output has a few channels ~40x larger than the rest, so int8
    activation quantisation of the 32k-way head crushed the small channels
    (Japanese top-5 parity fell to ~34%). Divide those channels by s and
    multiply the head's weight columns by s: identical maths, flatter
    activations. 1/s folds into the final norm weight; the head gets its own
    (untied) weight E*s, and the embedding lookup later divides by s again
    (see quantize.share_tied_embedding). Returns s.
    """
    import numpy as np
    feats = []
    hook = model.norm.register_forward_hook(lambda m, i, o: feats.append(o.abs().amax(dim=(0, 1))))
    for lang in ["en", "es", "ja"]:
        x = torch.from_numpy(np.load(f"{data_dir}/val_{lang}.npy", mmap_mode="r")[:rows, :model.cfg.max_seq_len].astype(np.int64))
        model(x)
    hook.remove()
    x_max = torch.stack(feats).amax(0).clamp(min=1e-5)
    w = model.tok_embeddings.weight
    w_max = w.abs().amax(0).clamp(min=1e-5)
    s = (x_max ** alpha / w_max ** (1 - alpha)).clamp(min=1e-2)
    model.norm.weight.div_(s)
    model.output = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    model.output.weight = torch.nn.Parameter(w * s[None, :])
    print(f"smoothed output head: activation max/median {x_max.max() / x_max.median():.1f}x -> "
          f"{(x_max / s).max() / (x_max / s).median():.1f}x")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--output", default="export/zenith_kb_fp32.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--data-dir", default="data_kb", help="val_<lang>.npy for SmoothQuant calibration")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model = load_fp32(args.checkpoint)
    ref = load_fp32(args.checkpoint)
    smooth = None if args.no_smooth else smooth_output_head(model, args.data_dir)
    wrapper = ZenithONNX(model).eval()
    check_against_pytorch(ref, wrapper)

    tokens = torch.randint(10, model.cfg.vocab_size, (1, 8))
    pos = torch.tensor([0])
    k = torch.zeros(wrapper.cache_shape())
    v = torch.zeros(wrapper.cache_shape())
    torch.onnx.export(
        wrapper, (tokens, pos, k, v), args.output,
        input_names=["tokens", "pos", "past_k", "past_v"],
        output_names=["logits", "new_k", "new_v"],
        dynamic_axes={"tokens": {1: "T"}, "logits": {1: "T"}},
        opset_version=args.opset,
        dynamo=False,
    )
    if smooth is not None:
        # stash 1/s in the graph so quantize.py can undo it on the embedding path
        import onnx
        m = onnx.load(args.output)
        m.graph.initializer.append(onnx.numpy_helper.from_array((1.0 / smooth).numpy().astype("float32"), "zenith_unsmooth"))
        onnx.save(m, args.output)
    print(f"Exported -> {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
