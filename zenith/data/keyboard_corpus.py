"""Build the mixed English/Spanish/Japanese keyboard corpus.

Stages:
  download  fetch raw sources into <raw_dir> (resumable curl; big corpora are
            taken as a compressed prefix via HTTP range requests)
  clean     normalise + filter every source into per-language text files,
            one example per line, no language tag (the packer adds it)
  pack      tokenize per language, then mix rows to the target language ratio
            into a single (n_rows, seq_len) uint16 array for PackedTokenDataset

Every packed row starts with the language token (<en>/<es>/<ja>), even when it
starts mid-sentence: that's exactly what MyKey sends at inference time (tag +
the last ~64 tokens of the field), so the model always knows the language.
"""
import argparse
import bz2
import glob
import gzip
import hashlib
import lzma
import os
import random
import re
import subprocess
import unicodedata
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

LANGS = ["en", "es", "ja"]
TARGET_MIX = {"en": 0.35, "es": 0.30, "ja": 0.35}
MAX_CHARS = 200
VAL_FRACTION = 0.002

HF_WIKI = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main"
SOURCES = {
    # name: (url, max_compressed_bytes or None for the whole file)
    "tatoeba_en.tsv.bz2": ("https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2", None),
    "tatoeba_es.tsv.bz2": ("https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences.tsv.bz2", None),
    "tatoeba_ja.tsv.bz2": ("https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2", None),
    "os_ja.txt.gz": ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ja.txt.gz", None),
    "os_en.txt.gz": ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz", 420_000_000),
    "os_es.txt.gz": ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz", 360_000_000),
    "cc100_ja.txt.xz": ("https://data.statmt.org/cc-100/ja.txt.xz", 300_000_000),
    "wiki_en.parquet": (f"{HF_WIKI}/20231101.en/train-00001-of-00041.parquet", None),
    "wiki_es.parquet": (f"{HF_WIKI}/20231101.es/train-00001-of-00013.parquet", None),
    "wiki_ja.parquet": (f"{HF_WIKI}/20231101.ja/train-00001-of-00015.parquet", None),
}

URL_RE = re.compile(r"https?://|www\.|\.com\b|@\w+\.", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")
JA_SENT_RE = re.compile(r"(?<=[。！？!?])")
LATIN_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'¿¡(]?[A-ZÁÉÍÓÚÑ0-9])")
SUB_JUNK_RE = re.compile(r"subt[ií]t|opensubtitles|sync|translated by|traducci[oó]n|字幕|翻訳", re.I)
BRACKETS_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|<[^>]*>|\{[^}]*\}")


# ---------------------------------------------------------------- download

def _content_length(url):
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", out)
    return int(sizes[-1])


def _fetch_range(job):
    url, part, lo, hi = job
    want = hi - lo + 1
    for _ in range(100):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return part
        with open(part, "ab") as f:
            subprocess.run(["curl", "-sSL", "--retry", "3", "--speed-time", "60", "--speed-limit", "1000",
                            "-r", f"{lo + have}-{hi}", url], stdout=f)
    raise RuntimeError(f"failed to fetch {part}")


def download(raw_dir, workers=16, chunk=8 << 20):
    """Parallel range downloads: some mirrors cap each connection at ~40KB/s, so
    every file is split into 8MB ranges fetched by many concurrent curls."""
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(raw_dir, exist_ok=True)
    plan = []
    for name, (url, max_bytes) in SOURCES.items():
        out = os.path.join(raw_dir, name)
        if os.path.exists(out):
            continue
        total = _content_length(url)
        if max_bytes:
            total = min(total, max_bytes)
        parts_dir = out + ".parts"
        os.makedirs(parts_dir, exist_ok=True)
        jobs = [(url, os.path.join(parts_dir, f"{i:05d}"), lo, min(total, lo + chunk) - 1)
                for i, lo in enumerate(range(0, total, chunk))]
        plan.append((name, out, parts_dir, jobs))
        print(f"{name}: {total / 1e6:.0f} MB in {len(jobs)} ranges")

    with ThreadPoolExecutor(workers) as pool, ThreadPoolExecutor(len(plan) or 1) as assembler:
        for name, out, parts_dir, jobs in plan:
            futures = [pool.submit(_fetch_range, j) for j in jobs]
            # assemble as soon as this file's ranges are in (later files keep downloading)
            assembler.submit(_assemble, name, out, parts_dir, futures)


def _assemble(name, out, parts_dir, futures):
    parts = [f.result() for f in futures]
    with open(out + ".tmp", "wb") as dst:
        for part in parts:
            with open(part, "rb") as src:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
    os.replace(out + ".tmp", out)
    for part in parts:
        os.remove(part)
    os.rmdir(parts_dir)
    print(f"done {name}: {os.path.getsize(out) / 1e6:.0f} MB", flush=True)


# ------------------------------------------------------------------- clean

def normalise(text):
    return unicodedata.normalize("NFKC", text).strip()


def is_ja(text):
    return len(CJK_RE.findall(text)) >= 0.3 * len(text)


def keep(text, lang):
    if len(text) < 2 or len(text) > MAX_CHARS:
        return False
    if URL_RE.search(text):
        return False
    digits = sum(c.isdigit() for c in text)
    if digits > 0.4 * len(text):
        return False
    if lang == "ja":
        return is_ja(text)
    if CJK_RE.search(text):
        return False
    # mostly-letters guard against symbol/garbage lines
    letters = sum(c.isalpha() for c in text)
    return letters >= 0.5 * len(text)


def chunk_sentences(sentences, max_chars, joiner):
    """Greedily join consecutive sentences into chunks of <= max_chars."""
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if buf:
                yield buf
            buf = ""
            continue
        cand = s if not buf else buf + joiner + s
        if len(cand) > max_chars:
            yield buf
            buf = s
        else:
            buf = cand
    if buf:
        yield buf


def split_sentences(text, lang):
    if lang == "ja":
        return JA_SENT_RE.split(text)
    return LATIN_SENT_RE.split(text)


def read_lines_tolerant(opener, path):
    """Iterate decoded lines; a truncated compressed prefix just ends the stream."""
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
    except (EOFError, lzma.LZMAError, OSError):
        return


def iter_tatoeba(path, lang):
    for line in read_lines_tolerant(bz2.open, path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            yield normalise(parts[2])


def iter_subtitles(path, lang):
    """OpenSubtitles mono: one subtitle line per row, consecutive rows are dialogue.
    For en/es, join 1-3 consecutive lines (like a short chat message); Japanese
    lines stay single since joining them would need invented punctuation."""
    rng = random.Random(hash(path) & 0xFFFF)
    short_counts = {}
    group, want = [], rng.randint(1, 3)
    prev = None
    for raw in read_lines_tolerant(gzip.open if path.endswith(".gz") else open, path):
        t = normalise(BRACKETS_RE.sub("", raw))
        t = t.lstrip("-–—―ー ").strip()
        if not t or t == prev or SUB_JUNK_RE.search(t) or "♪" in t:
            continue
        prev = t
        # cap very frequent short lines ("Yeah.", "Okay.") so they don't dominate
        if len(t) <= 20:
            c = short_counts.get(t, 0)
            if c >= 300:
                continue
            short_counts[t] = c + 1
            if len(short_counts) > 3_000_000:
                short_counts.clear()
        if lang == "ja":
            yield t
            continue
        group.append(t)
        if len(group) >= want:
            yield " ".join(group)
            group, want = [], rng.randint(1, 3)
    if group and lang != "ja":
        yield " ".join(group)


def iter_cc100(path, lang):
    """CC-100: paragraphs, blank line between documents. Split to sentences
    and re-chunk to keyboard-sized pieces."""
    for raw in read_lines_tolerant(lzma.open, path):
        t = normalise(raw)
        if not t:
            continue
        yield from chunk_sentences(split_sentences(t, lang), 120, "")


def iter_wiki(path, lang):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    joiner = "" if lang == "ja" else " "
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            paras = [p for p in text.split("\n") if len(p) > 40][:2]
            for p in paras:
                yield from chunk_sentences(split_sentences(normalise(p), lang), MAX_CHARS, joiner)


def source_jobs(raw_dir):
    jobs = []
    for lang in LANGS:
        for name, fn in [(f"tatoeba_{lang}.tsv.bz2", "tatoeba"), (f"os_{lang}.txt.gz", "subtitles"),
                         (f"wiki_{lang}.parquet", "wiki"), (f"cc100_{lang}.txt.xz", "cc100")]:
            path = os.path.join(raw_dir, name)
            if os.path.exists(path):
                jobs.append((path, fn, lang))
    return jobs


ITERATORS = {"tatoeba": iter_tatoeba, "subtitles": iter_subtitles, "cc100": iter_cc100, "wiki": iter_wiki}


def clean_one(job):
    path, fn, lang, out_dir = job
    base = os.path.basename(path).split(".")[0]
    train_out = os.path.join(out_dir, f"{lang}.{base}.train.txt")
    val_out = os.path.join(out_dir, f"{lang}.{base}.val.txt")
    n_train = n_val = n_chars = 0
    with open(train_out, "w", encoding="utf-8") as ft, open(val_out, "w", encoding="utf-8") as fv:
        for text in ITERATORS[fn](path, lang):
            text = text.replace("\n", " ").strip()
            if not keep(text, lang):
                continue
            h = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")
            if h / 2**32 < VAL_FRACTION:
                fv.write(text + "\n")
                n_val += 1
            else:
                ft.write(text + "\n")
                n_train += 1
                n_chars += len(text)
    return base, lang, n_train, n_val, n_chars


def clean(raw_dir, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(p, fn, lang, out_dir) for p, fn, lang in source_jobs(raw_dir)]
    with Pool(min(workers, len(jobs))) as pool:
        for base, lang, n_train, n_val, n_chars in pool.imap_unordered(clean_one, jobs):
            print(f"[{lang}] {base}: {n_train:,} train / {n_val:,} val lines, {n_chars / 1e6:.0f}M chars")


# -------------------------------------------------------------------- pack

def _tokenize_file(args):
    path, tokenizer_path, out_path = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    eos = tok.token_to_id("<eos>")
    n = 0
    with open(path, encoding="utf-8") as f, open(out_path, "wb") as out:
        batch = []

        def flush():
            nonlocal n
            ids = []
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                ids.extend(enc.ids)
                ids.append(eos)
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n += len(ids)

        for line in f:
            batch.append(line.rstrip("\n"))
            if len(batch) >= 20000:
                flush()
                batch = []
        if batch:
            flush()
    return path, n


def tokenize_all(text_dir, tokenizer_path, workers):
    jobs = []
    for path in sorted(glob.glob(os.path.join(text_dir, "*.txt"))):
        jobs.append((path, tokenizer_path, path[:-4] + ".tok.bin"))
    counts = {}
    with Pool(min(workers, len(jobs))) as pool:
        for path, n in tqdm(pool.imap_unordered(_tokenize_file, jobs), total=len(jobs), desc="tokenizing"):
            counts[path] = n
            print(f"  {os.path.basename(path)}: {n / 1e6:.1f}M tokens")
    return counts


def rows_from_stream(bin_paths, lang_id, seq_len, out_path):
    """Cut token streams into rows of seq_len, each starting with lang_id, written
    straight to a .npy memmap (one language can be ~1GB; RAM here is tight)."""
    body = seq_len - 1
    streams = [np.memmap(p, dtype=np.uint16, mode="r") for p in bin_paths]
    n_rows = sum(len(s) // body for s in streams)
    rows = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(n_rows, seq_len))
    r = 0
    for s in streams:
        n = len(s) // body
        for i in range(0, n, 1_000_000):
            j = min(n, i + 1_000_000)
            rows[r + i:r + j, 0] = lang_id
            rows[r + i:r + j, 1:] = np.asarray(s[i * body:j * body]).reshape(j - i, body)
        r += n
    rows.flush()
    return np.load(out_path, mmap_mode="r")


def pack(text_dir, tokenizer_path, out_dir, seq_len, total_tokens, workers, seed=1337):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    if not glob.glob(os.path.join(text_dir, "*.tok.bin")):
        tokenize_all(text_dir, tokenizer_path, workers)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    rows = {}
    for split in ["val", "train"]:
        for lang in LANGS:
            bins = sorted(glob.glob(os.path.join(text_dir, f"{lang}.*.{split}.tok.bin")))
            rows[split, lang] = rows_from_stream(bins, tok.token_to_id(f"<{lang}>"), seq_len,
                                                 os.path.join(out_dir, f"{split}_{lang}.npy"))
            print(f"[{split}] {lang}: {rows[split, lang].shape[0]:,} rows "
                  f"({rows[split, lang].size / 1e6:.0f}M tokens)")

    # val: equal rows per language (per-language val files are kept too)
    n_each = min(rows["val", lang].shape[0] for lang in LANGS)
    mixed = np.concatenate([rows["val", lang][np.sort(rng.permutation(rows["val", lang].shape[0])[:n_each])]
                            for lang in LANGS])
    np.save(os.path.join(out_dir, "val.npy"), mixed[rng.permutation(mixed.shape[0])])

    # train: hit the target language mix, repeating a language whose unique data
    # is short of its share — temperature-style balancing so the smaller corpora
    # aren't drowned. Row order doesn't matter: the DataLoader shuffles.
    total_rows = total_tokens // seq_len
    out = np.lib.format.open_memmap(os.path.join(out_dir, "train.npy"), mode="w+",
                                    dtype=np.uint16, shape=(total_rows, seq_len))
    start = 0
    for k, lang in enumerate(LANGS):
        src = rows["train", lang]
        n = src.shape[0]
        want = int(total_rows * TARGET_MIX[lang]) if k < len(LANGS) - 1 else total_rows - start
        full, rem = divmod(want, n)
        pos = start
        for _ in range(full):
            for i in range(0, n, 1_000_000):
                j = min(n, i + 1_000_000)
                out[pos:pos + j - i] = src[i:j]
                pos += j - i
        idx = np.sort(rng.choice(n, size=rem, replace=False))
        for i in range(0, rem, 1_000_000):
            chunk = idx[i:i + 1_000_000]
            out[pos:pos + len(chunk)] = src[chunk]
            pos += len(chunk)
        print(f"[train] {lang}: {want:,} rows = {want / total_rows:.1%} ({want / n:.2f} epochs of unique data)")
        start += want
    out.flush()
    print(f"Saved {total_rows:,} rows x {seq_len} -> {os.path.join(out_dir, 'train.npy')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--raw-dir", default="data_kb/raw")

    c = sub.add_parser("clean")
    c.add_argument("--raw-dir", default="data_kb/raw")
    c.add_argument("--out-dir", default="data_kb/text")
    c.add_argument("--workers", type=int, default=10)

    p = sub.add_parser("pack")
    p.add_argument("--text-dir", default="data_kb/text")
    p.add_argument("--tokenizer", default="data_kb/tokenizer_kb.json")
    p.add_argument("--out-dir", default="data_kb")
    p.add_argument("--seq-len", type=int, default=65, help="max_seq_len + 1 (inputs and shifted targets)")
    p.add_argument("--total-tokens", type=int, default=1_340_000_000)
    p.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "download":
        download(args.raw_dir)
    elif args.cmd == "clean":
        clean(args.raw_dir, args.out_dir, args.workers)
    elif args.cmd == "pack":
        pack(args.text_dir, args.tokenizer, args.out_dir, args.seq_len, args.total_tokens, args.workers)


if __name__ == "__main__":
    main()
