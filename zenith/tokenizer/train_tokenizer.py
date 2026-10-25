"""Train a tokenizer from scratch on a text corpus.

Two models:
  bpe      byte-level BPE (the original TinyStories tokenizer)
  unigram  SentencePiece-style Unigram with Metaspace (▁ marks word starts) and
           byte fallback — needed for Japanese, which has no spaces for a
           byte-level BPE pre-tokenizer to split on.
"""
import argparse

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.models import BPE, Unigram

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# Language tags MyKey sends with every request; <ru>/<zh> are reserved so those
# languages can be added later without changing the vocab layout.
LANG_TOKENS = ["<en>", "<es>", "<ja>", "<ru>", "<zh>"]
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]


def train_tokenizer(input_files, output_path, vocab_size=8192):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train(files=input_files, trainer=trainer)
    tokenizer.save(output_path)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) -> {output_path}")
    return tokenizer


def train_unigram_tokenizer(input_files, output_path, vocab_size=32000):
    tokenizer = Tokenizer(Unigram())
    # NFKC inside the tokenizer so training and the phone see identical text
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Metaspace(replacement="▁", prepend_scheme="always")])

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS + LANG_TOKENS + BYTE_TOKENS,
        unk_token="<unk>",
        max_piece_length=16,
        shrinking_factor=0.75,
        show_progress=True,
    )
    tokenizer.train(files=input_files, trainer=trainer)

    # Re-load with byte_fallback on: a rare kanji or emoji becomes its UTF-8
    # bytes (<0xE8><0x9B><0x99>) instead of <unk>.
    import json
    spec = json.loads(tokenizer.to_str())
    spec["model"]["byte_fallback"] = True
    tokenizer = Tokenizer.from_str(json.dumps(spec))
    tokenizer.save(output_path)
    print(f"Saved unigram tokenizer ({tokenizer.get_vocab_size()} tokens) -> {output_path}")
    return tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Text file(s) to train on")
    ap.add_argument("--output", required=True, help="Output tokenizer.json path")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--model", choices=["bpe", "unigram"], default="unigram")
    args = ap.parse_args()
    if args.model == "bpe":
        train_tokenizer(args.input, args.output, args.vocab_size or 8192)
    else:
        train_unigram_tokenizer(args.input, args.output, args.vocab_size or 32000)


if __name__ == "__main__":
    main()
