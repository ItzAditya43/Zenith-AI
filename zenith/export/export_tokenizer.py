"""Write the Unigram tokenizer in a compact form MyKey can load without Python.

Format (UTF-8 text, one record per line):
  line 1:  ZENITH-UNIGRAM 1
  line 2:  JSON header {"vocab_size", "unk_id", "byte_fallback", "normalizer",
           "metaspace", "special": {name: id}, "byte_offset"}
  then:    <piece>\t<score> for every id in order (piece escaped: \\t \\n \\\\)

Encoding on the phone: NFKC-normalise, replace ' ' with '▁', prepend '▁', then
Viterbi over pieces by score; unknown chars fall back to their UTF-8 bytes
<0xNN> (ids byte_offset + byte). Specials are never produced by Viterbi.
"""
import argparse
import json

from tokenizers import Tokenizer


def escape(p):
    return p.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data_kb/tokenizer_kb.json")
    ap.add_argument("--output", default="export/zenith_kb.tok")
    args = ap.parse_args()

    spec = json.loads(Tokenizer.from_file(args.tokenizer).to_str())
    model = spec["model"]
    vocab = model["vocab"]  # [[piece, score], ...] in id order
    special = {p: i for i, (p, _) in enumerate(vocab) if p.startswith("<") and p.endswith(">") and not p.startswith("<0x")}
    header = {
        "vocab_size": len(vocab),
        "unk_id": model.get("unk_id"),
        "byte_fallback": model.get("byte_fallback", False),
        "normalizer": "NFKC",
        "metaspace": {"replacement": "▁", "prepend_scheme": "always"},
        "special": special,
        "byte_offset": next(i for i, (p, _) in enumerate(vocab) if p == "<0x00>"),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("ZENITH-UNIGRAM 1\n")
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for piece, score in vocab:
            f.write(f"{escape(piece)}\t{score:.6f}\n")
    print(f"Wrote {len(vocab)} pieces -> {args.output}")


if __name__ == "__main__":
    main()
