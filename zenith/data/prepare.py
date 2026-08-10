"""Download TinyStories and dump it to a plain text file for tokenizer training,
then tokenize + pack it into fixed-length shards for fast memmap-based loading."""
import argparse
import os

import numpy as np
from tqdm import tqdm


def dump_text(output_dir: str, limit: int = None):
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    ds = load_dataset("roneneldan/TinyStories")

    def write_split(split_name, path, limit=None):
        with open(path, "w", encoding="utf-8") as f:
            rows = ds[split_name]
            n = len(rows) if limit is None else min(limit, len(rows))
            for i in tqdm(range(n), desc=f"writing {split_name}"):
                text = rows[i]["text"].strip()
                if text:
                    f.write(text + "\n<eos>\n")

    write_split("train", train_path, limit)
    write_split("validation", val_path, limit=2000)
    print(f"Wrote {train_path} and {val_path}")
    return train_path, val_path


def tokenize_and_pack(text_path: str, tokenizer_path: str, out_path: str, seq_len: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id("<eos>")

    all_ids = []
    with open(text_path, "r", encoding="utf-8") as f:
        buf = []
        for line in tqdm(f, desc=f"tokenizing {os.path.basename(text_path)}"):
            line = line.rstrip("\n")
            if line == "<eos>":
                buf.append(eos_id)
                continue
            if line:
                ids = tok.encode(line).ids
                buf.append(tok.token_to_id("<bos>"))
                buf.extend(ids)
            if len(buf) > 1_000_000:
                all_ids.extend(buf)
                buf = []
        all_ids.extend(buf)

    arr = np.array(all_ids, dtype=np.uint16)
    n_tokens = (len(arr) // seq_len) * seq_len
    arr = arr[:n_tokens].reshape(-1, seq_len)
    np.save(out_path, arr)
    print(f"Saved {arr.shape[0]} sequences of length {seq_len} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--output-dir", default="data")
    d.add_argument("--limit", type=int, default=None)

    t = sub.add_parser("pack")
    t.add_argument("--input", required=True)
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--seq-len", type=int, default=512)

    args = ap.parse_args()
    if args.cmd == "dump":
        dump_text(args.output_dir, args.limit)
    elif args.cmd == "pack":
        tokenize_and_pack(args.input, args.tokenizer, args.output, args.seq_len)


if __name__ == "__main__":
    main()
