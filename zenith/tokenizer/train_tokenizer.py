"""Train a BPE tokenizer from scratch on a text corpus."""
import argparse

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
from tokenizers.models import BPE

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Text file(s) to train on")
    ap.add_argument("--output", required=True, help="Output tokenizer.json path")
    ap.add_argument("--vocab-size", type=int, default=8192)
    args = ap.parse_args()
    train_tokenizer(args.input, args.output, args.vocab_size)


if __name__ == "__main__":
    main()
