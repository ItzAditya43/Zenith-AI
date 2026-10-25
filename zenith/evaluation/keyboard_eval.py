"""Keyboard-specific evaluation: does Zenith actually help typing?

Per language, on held-out chat-style sentences (subtitle val split):

English / Spanish
  next-word top-3      is the real next word among the top three suggestions?
  prefix top-3         given the first 1-3 letters, is the word in the top three?
  keystroke savings    share of letters saved by tapping a top-3 suggestion

Japanese
  kanji choice top-1   for each kanji word (fugashi + UniDic segmentation), look up
                       every Mozc candidate for its reading and check whether the
                       model ranks the original surface first, given the sentence
                       before it. Compared against Mozc's own cost order.
  next-phrase top-3    is the next word among the model's top three continuations?

Every metric is also run with a word-bigram baseline built from the training text,
so Zenith has to win per language, not just on average.
"""
import argparse
import json
import math
import os
import random
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from zenith.model import ZenithConfig, ZenithTransformer
from zenith.training.checkpoint import load_checkpoint

MOZC_URL = "https://raw.githubusercontent.com/google/mozc/master/src/data/dictionary_oss/dictionary{:02d}.txt"
WORD_RE = re.compile(r"^[\w'\-áéíóúñü]+$", re.I)
KANJI_RE = re.compile(r"[㐀-鿿]")
MAX_CTX = 63  # model window is 64 incl. the language tag


# ------------------------------------------------------------------ model

class LM:
    def __init__(self, checkpoint, tokenizer_path, device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg = ZenithConfig(**payload["config"])
        self.model = ZenithTransformer(self.cfg).to(device)
        load_checkpoint(checkpoint, self.model, map_location=device)
        self.model.eval()
        self.step = payload.get("step")
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.device = device
        vocab = self.tok.get_vocab()
        self.pieces = [None] * len(vocab)
        for p, i in vocab.items():
            self.pieces[i] = p
        self.special = {i for i, p in enumerate(self.pieces) if p.startswith("<") and p.endswith(">")}
        self.eos = self.tok.token_to_id("<eos>")
        # tokens that end the current word: a new word start, punctuation, eos
        self.word_start = torch.zeros(len(vocab), dtype=torch.bool)
        self.terminator = torch.zeros(len(vocab), dtype=torch.bool)
        for i, p in enumerate(self.pieces):
            if p.startswith("▁"):
                self.word_start[i] = True
                self.terminator[i] = True
            elif i in self.special or (p and all(unicodedata.category(c).startswith("P") for c in p)):
                self.terminator[i] = True
        self.terminator[self.eos] = True
        self.terminator = self.terminator.to(device)
        self.word_start = self.word_start.to(device)

    def encode(self, lang, text):
        ids = self.tok.encode(text, add_special_tokens=False).ids if text else []
        return [self.tok.token_to_id(f"<{lang}>")] + ids[-MAX_CTX:]

    def hidden(self, x):
        """Final normed hidden states — the 32k-vocab projection is applied only
        where it's needed (it dominates the cost of these small forwards)."""
        m = self.model
        h = m.tok_embeddings(x)
        cos, sin = m.rope_cos[:x.shape[1]], m.rope_sin[:x.shape[1]]
        for layer in m.layers:
            h = layer(h, cos, sin)
        return m.norm(h)

    @torch.no_grad()
    def next_logprobs(self, seqs):
        """Log-probs of the next token for each sequence (list of id lists)."""
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        last = []
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            last.append(len(s) - 1)
        h = self.hidden(x.to(self.device))[torch.arange(len(seqs)), torch.tensor(last)]
        return F.log_softmax(self.model.output(h).float(), dim=-1)

    @torch.no_grad()
    def seq_logprob(self, seqs):
        """Total log-prob of each full sequence (excluding its first token)."""
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L - 1)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s) - 1] = 1
        x = x.to(self.device)
        logp = F.log_softmax(self.model(x[:, :-1]).float(), dim=-1)
        tgt = logp.gather(-1, x[:, 1:, None]).squeeze(-1)
        return (tgt * mask.to(self.device)).sum(-1).tolist()

    def predict_words(self, lang, context, prefix="", k=3, beam=12, max_tokens=4):
        """Top-k next whole words (Latin scripts). A word = a ▁-initial token plus
        continuation tokens, scored as P(tokens) * P(word ends here)."""
        base = self.encode(lang, context)
        lp = self.next_logprobs([base])[0]
        cand = []
        for i in torch.topk(lp.masked_fill(~self.word_start, -1e9), 400).indices.tolist():
            text = self.pieces[i][1:]
            if _compatible(text, prefix):
                cand.append((base + [i], lp[i].item(), text))
            if len(cand) >= beam:
                break
        finished = {}
        for _ in range(max_tokens):
            if not cand:
                break
            lps = self.next_logprobs([c[0] for c in cand])
            end_lp = torch.logsumexp(lps.masked_fill(~self.terminator, -1e9), dim=-1)
            nxt = []
            for (ids, score, text), row, e in zip(cand, lps, end_lp.tolist()):
                if text and text.startswith(prefix) and WORD_RE.match(text):
                    total = score + e
                    if total > finished.get(text, -1e9):
                        finished[text] = total
                cont = row.masked_fill(self.terminator, -1e9)
                for j in torch.topk(cont, 8).indices.tolist():
                    t2 = text + self.pieces[j]
                    if _compatible(t2, prefix):
                        nxt.append((ids + [j], score + row[j].item(), t2))
            cand = sorted(nxt, key=lambda c: -c[1])[:beam]
        return [w for w, _ in sorted(finished.items(), key=lambda kv: -kv[1])[:k]]

    def _heal_mask(self, rest, word_start):
        key = (rest, word_start)
        cache = self.__dict__.setdefault("_heal_cache", {})
        if key not in cache:
            cache[key] = _heal_mask_impl(self.pieces, rest, word_start).to(self.device)
        return cache[key]

    def predict_continuations(self, lang, context, min_chars, k=3, beam=8, max_tokens=5):
        """Top-k continuation strings (Japanese), extended until >= min_chars.

        Token healing: a context cut mid-token (「今日」 when training always saw
        「今日は」 as one piece) puts the model off-distribution. So the last
        context token is dropped and the continuation is required to start with
        its text; that prefix is stripped from the returned strings. MyKey needs
        the same trick whenever the cursor isn't at a clean token boundary.
        """
        ids = self.encode(lang, context)
        heal = self.pieces[ids[-1]].replace("▁", "") if len(ids) > 1 else ""
        base = ids[:-1] if heal else ids
        at_start = len(base) == 1  # only the language tag left
        cand = [(base, 0.0, "")]
        done = []
        for step in range(max_tokens + (1 if heal else 0)):
            if not cand:
                break
            lps = self.next_logprobs([c[0] for c in cand])
            nxt = []
            for (seq, score, text), row in zip(cand, lps):
                if len(text) < len(heal):
                    # still healing: rank only tokens consistent with the dropped text
                    row = row.masked_fill(~self._heal_mask(heal[len(text):], at_start and not text), -1e9)
                for j in torch.topk(row, 4 * beam).indices.tolist():
                    if row[j] < -1e8:
                        break
                    p = self.pieces[j]
                    if j in self.special or (p.startswith("▁") and not (at_start and not text)):
                        continue
                    t2 = text + p.replace("▁", "")
                    if heal and not _compatible(t2, heal):
                        continue
                    item = (seq + [j], score + row[j].item(), t2)
                    if len(t2) - len(heal) >= min_chars and t2.startswith(heal):
                        done.append(item)
                    else:
                        nxt.append(item)
                    if len(nxt) >= 4 * beam:
                        break
            cand = sorted(nxt, key=lambda c: -c[1])[:beam]
        out = []
        for _, _, t in sorted(done, key=lambda c: -c[1]):
            t = t[len(heal):]
            if t not in out:
                out.append(t)
            if len(out) == k:
                break
        return out


def _heal_mask_impl(pieces, rest, word_start):
    mask = torch.zeros(len(pieces), dtype=torch.bool)
    for i, p in enumerate(pieces):
        if p is None or (p.startswith("<") and p.endswith(">")):
            continue
        if word_start != p.startswith("▁"):
            continue
        t = p.replace("▁", "")
        if t and _compatible(t, rest):
            mask[i] = True
    return mask


class OnnxLM(LM):
    """Same predictions, but logits come from the exported ONNX model — so the
    keyboard metrics can be measured on exactly what ships to the phone."""

    def __init__(self, checkpoint, tokenizer_path, onnx_path):
        super().__init__(checkpoint, tokenizer_path, "cpu")
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        c = self.cfg
        self.zero = np.zeros((c.n_layers, 1, c.n_kv_heads, c.max_seq_len, c.head_dim), np.float32)

    def _logits(self, seq):
        out = self.sess.run(None, {"tokens": np.asarray([seq], np.int64), "pos": np.array([0]),
                                   "past_k": self.zero, "past_v": self.zero})[0][0]
        return torch.from_numpy(out)

    @torch.no_grad()
    def next_logprobs(self, seqs):
        return torch.stack([F.log_softmax(self._logits(s)[-1].float(), -1) for s in seqs])

    @torch.no_grad()
    def seq_logprob(self, seqs):
        out = []
        for s in seqs:
            lp = F.log_softmax(self._logits(s[:-1]).float(), -1)
            out.append(lp.gather(-1, torch.tensor(s[1:])[:, None]).sum().item())
        return out


def _compatible(text, prefix):
    return text.startswith(prefix) or prefix.startswith(text)


# --------------------------------------------------------------- baseline

class Bigram:
    def __init__(self, sentences, tokenize):
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        for s in sentences:
            words = ["<s>"] + tokenize(s)
            self.uni.update(words[1:])
            for a, b in zip(words, words[1:]):
                self.bi[a][b] += 1
        self.top_uni = [w for w, _ in self.uni.most_common(5000)]

    def predict(self, prev, prefix="", k=3):
        out = [w for w, _ in self.bi.get(prev, Counter()).most_common() if w.startswith(prefix)][:k]
        for w in self.top_uni:
            if len(out) >= k:
                break
            if w.startswith(prefix) and w not in out:
                out.append(w)
        return out

    def score(self, prev, word):
        return math.log((self.bi.get(prev, {}).get(word, 0) + 0.1 * self.uni.get(word, 0) / max(1, len(self.uni)) + 1e-9))


# ----------------------------------------------------------- en / es eval

def latin_words(sentence):
    return [w.strip(".,!?¿¡\"“”«»:;()…") for w in sentence.split()]


def eval_latin(lm, lang, sentences, bigram, max_ks_sentences):
    stats = Counter()
    for si, s in enumerate(sentences):
        raw = s.split()
        words = latin_words(s)
        for i in range(1, len(words)):
            gold = words[i]
            if not gold or not WORD_RE.match(gold) or gold.isdigit():
                continue
            ctx = " ".join(raw[:i])
            prev = words[i - 1].lower() if words[i - 1] else "<s>"
            g = gold.lower()

            preds = [p.lower() for p in lm.predict_words(lang, ctx)]
            stats["nw_n"] += 1
            stats["nw_zenith"] += g in preds
            stats["nw_bigram"] += g in bigram.predict(prev)

            if len(gold) >= 2:
                for plen in range(1, min(3, len(gold) - 1) + 1):
                    pre = gold[:plen]
                    stats["pc_n"] += 1
                    stats["pc_zenith"] += g in [p.lower() for p in lm.predict_words(lang, ctx, prefix=pre)]
                    stats["pc_bigram"] += g in bigram.predict(prev, prefix=pre.lower())

            if si < max_ks_sentences:
                # keystroke savings: type letter by letter until gold shows in top-3
                stats["ks_total"] += len(gold) + 1  # + space
                for typed in range(0, len(gold)):
                    pre = gold[:typed]
                    if g in [p.lower() for p in lm.predict_words(lang, ctx, prefix=pre)]:
                        stats["ks_zenith"] += len(gold) - typed  # tap inserts word + space
                        break
                for typed in range(0, len(gold)):
                    if g in bigram.predict(prev, prefix=gold[:typed].lower()):
                        stats["ks_bigram"] += len(gold) - typed
                        break
    r = lambda a, b: stats[a] / max(1, stats[b])
    return {
        "next_word_top3": {"zenith": r("nw_zenith", "nw_n"), "bigram": r("nw_bigram", "nw_n"), "n": stats["nw_n"]},
        "prefix_top3": {"zenith": r("pc_zenith", "pc_n"), "bigram": r("pc_bigram", "pc_n"), "n": stats["pc_n"]},
        "keystroke_savings": {"zenith": r("ks_zenith", "ks_total"), "bigram": r("ks_bigram", "ks_total")},
    }


# --------------------------------------------------------------- ja eval

def kata_to_hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def load_mozc(cache_dir, readings=None):
    """reading -> surfaces in Mozc cost order. Pass `readings` to keep only those
    (the full dictionary is 1.3M entries — several hundred MB as Python dicts)."""
    path = os.path.join(cache_dir, "mozc_dictionary.tsv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as out:
            for i in range(10):
                with urllib.request.urlopen(MOZC_URL.format(i)) as r:
                    out.write(r.read().decode("utf-8"))
        os.replace(path + ".tmp", path)
    by_reading = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            reading, _, _, cost, surface = parts[:5]
            if readings is not None and reading not in readings:
                continue
            surface = unicodedata.normalize("NFKC", surface)
            cost = int(cost)
            if cost < by_reading[reading].get(surface, 1 << 30):
                by_reading[reading][surface] = cost
    return {r: [s for s, _ in sorted(d.items(), key=lambda kv: kv[1])] for r, d in by_reading.items()}


def ja_segment(tagger, sentence):
    out = []
    for w in tagger(sentence):
        kana = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        out.append((w.surface, kata_to_hira(kana) if kana and kana != "*" else None))
    return out


def eval_ja(lm, sentences, mozc, tagger, bigram, max_items):
    stats = Counter()
    for s in sentences:
        seg = ja_segment(tagger, s)
        for i, (surface, reading) in enumerate(seg):
            ctx = "".join(w for w, _ in seg[:i])
            prev = seg[i - 1][0] if i else "<s>"
            # kanji choice
            if stats["kc_n"] < max_items and reading and KANJI_RE.search(surface):
                cands = mozc.get(reading, [])
                if surface in cands and len(cands) >= 2:
                    cands = cands[:20]
                    scores = lm.seq_logprob([lm.encode("ja", ctx + c) for c in cands])
                    stats["kc_n"] += 1
                    stats["kc_zenith"] += cands[max(range(len(cands)), key=lambda j: scores[j])] == surface
                    stats["kc_mozc"] += cands[0] == surface
                    stats["kc_bigram"] += max(cands, key=lambda c: (bigram.score(prev, c), -cands.index(c))) == surface
            # next phrase
            if i >= 1 and stats["np_n"] < max_items and any(not unicodedata.category(c).startswith("P") for c in surface):
                preds = lm.predict_continuations("ja", ctx, min_chars=len(surface))
                stats["np_n"] += 1
                stats["np_zenith"] += any(p.startswith(surface) for p in preds)
                stats["np_bigram"] += surface in bigram.predict(prev)
        if stats["kc_n"] >= max_items and stats["np_n"] >= max_items:
            break
    r = lambda a, b: stats[a] / max(1, stats[b])
    return {
        "kanji_choice_top1": {"zenith": r("kc_zenith", "kc_n"), "mozc_order": r("kc_mozc", "kc_n"),
                              "bigram": r("kc_bigram", "kc_n"), "n": stats["kc_n"]},
        "next_phrase_top3": {"zenith": r("np_zenith", "np_n"), "bigram": r("np_bigram", "np_n"), "n": stats["np_n"]},
    }


# ------------------------------------------------------------------- main

def read_lines(path, n, seed=0, min_words=3):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.Random(seed).shuffle(lines)
    if min_words:
        lines = [l for l in lines if len(l.split()) >= min_words]
    return lines[:n]


def head_lines(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(line.strip())
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default="data_kb/tokenizer_kb.json")
    ap.add_argument("--text-dir", default="data_kb/text")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "ja"])
    ap.add_argument("--n-sentences", type=int, default=300)
    ap.add_argument("--ks-sentences", type=int, default=100)
    ap.add_argument("--ja-items", type=int, default=1000)
    ap.add_argument("--bigram-lines", type=int, default=300_000)
    ap.add_argument("--cache-dir", default="data_kb/eval_cache")
    ap.add_argument("--output", default=None, help="write results JSON here")
    ap.add_argument("--onnx", default=None, help="score with this exported ONNX model instead of PyTorch")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = OnnxLM(args.checkpoint, args.tokenizer, args.onnx) if args.onnx else LM(args.checkpoint, args.tokenizer, device)
    results = {"checkpoint": args.checkpoint, "step": lm.step}

    for lang in args.langs:
        val = os.path.join(args.text_dir, f"{lang}.os_{lang}.val.txt")
        train = os.path.join(args.text_dir, f"{lang}.os_{lang}.train.txt")
        if lang in ("en", "es"):
            sents = read_lines(val, args.n_sentences)
            bigram = Bigram(head_lines(train, args.bigram_lines), lambda s: [w.lower() for w in latin_words(s) if w])
            results[lang] = eval_latin(lm, lang, sents, bigram, args.ks_sentences)
        else:
            import fugashi

            tagger = fugashi.Tagger()
            sents = read_lines(val, args.n_sentences * 10, min_words=0)
            needed = {r for s in sents for _, r in ja_segment(tagger, s) if r}
            mozc = load_mozc(args.cache_dir, needed)
            n_bi = min(args.bigram_lines, 300_000)
            bigram = Bigram(head_lines(train, n_bi), lambda s: [w.surface for w in tagger(s)])
            results[lang] = eval_ja(lm, sents, mozc, tagger, bigram, args.ja_items)
        print(f"[{lang}] " + json.dumps(results[lang], indent=None))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
