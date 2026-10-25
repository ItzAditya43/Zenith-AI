"""ONNX (int8) vs PyTorch (fp32) parity, per language.

Feeds real held-out rows through both and compares the top-5 next-token sets
at every position. Target: >= 95% of positions identical top-5 (as a set), per
language. Also exercises the cache path: prefill half the row, then feed the
rest one token at a time, like keystrokes.
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort
import torch

from zenith.export.export_onnx import load_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--onnx", default="export/zenith_kb.onnx")
    ap.add_argument("--data-dir", default="data_kb")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model = load_fp32(args.checkpoint)
    cfg = model.cfg
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    shape = (cfg.n_layers, 1, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
    results = {}

    for lang in ["en", "es", "ja"]:
        rows = np.load(f"{args.data_dir}/val_{lang}.npy", mmap_mode="r")[: args.rows, : cfg.max_seq_len]
        match = total = top1 = top1_in3 = 0
        step_times = []
        for row in rows:
            ids = torch.from_numpy(row.astype(np.int64))[None]
            with torch.no_grad():
                ref = model(ids)[0].numpy()
            k = np.zeros(shape, np.float32)
            v = np.zeros(shape, np.float32)
            half = ids.shape[1] // 2
            lg, k, v = sess.run(None, {"tokens": ids[:, :half].numpy(), "pos": np.array([0]), "past_k": k, "past_v": v})
            outs = [lg[0]]
            for p in range(half, ids.shape[1]):
                t0 = time.perf_counter()
                lg, k, v = sess.run(None, {"tokens": ids[:, p:p + 1].numpy(), "pos": np.array([p]),
                                           "past_k": k, "past_v": v})
                step_times.append(time.perf_counter() - t0)
                outs.append(lg[0])
            got = np.concatenate(outs)
            for a, b in zip(ref, got):
                ra, rb = np.argsort(-a)[:5], np.argsort(-b)[:5]
                match += set(ra) == set(rb)
                top1 += ra[0] == rb[0]
                top1_in3 += ra[0] in rb[:3]
                total += 1
        results[lang] = {"top5_match": match / total, "top1_match": top1 / total,
                         "fp32_top1_in_int8_top3": top1_in3 / total, "positions": total,
                         "ms_per_step_pc_cpu": 1000 * float(np.median(step_times))}
        status = "OK" if match / total >= 0.95 else "BELOW TARGET"
        print(f"[{lang}] top-5 set match {match / total:.1%} | top-1 match {top1 / total:.1%} | "
              f"fp32 top-1 in int8 top-3 {top1_in3 / total:.1%} | {total} positions "
              f"({results[lang]['ms_per_step_pc_cpu']:.2f} ms/step on this CPU) {status}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
