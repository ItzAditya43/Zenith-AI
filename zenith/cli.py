import argparse
import sys

import torch
from tokenizers import Tokenizer

from zenith.model import ZenithConfig, ZenithTransformer
from zenith.training.checkpoint import load_checkpoint
from zenith.inference.generate import generate


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ZenithConfig(**payload["config"])
    model = ZenithTransformer(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    tok = Tokenizer.from_file(tokenizer_path)
    return model, tok, cfg


def cmd_chat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)
    print(f"Zenith loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params) on {device}")
    print("Type a prompt and press enter. Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        for piece in generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k,
                               top_p=args.top_p, device=device, stream=True):
            print(piece, end="", flush=True)
        print()


def cmd_train(args):
    sys.argv = ["train.py", args.config]
    from zenith.training.train import main as train_main
    train_main()


def cmd_evaluate(args):
    sys.argv = ["evaluate.py", args.checkpoint, "--val-path", args.val_path, "--batch-size", str(args.batch_size)]
    from zenith.evaluation.evaluate import main as eval_main
    eval_main()


def cmd_tokenize(args):
    sys.argv = ["train_tokenizer.py", "--input", *args.input, "--output", args.output,
                "--vocab-size", str(args.vocab_size)]
    from zenith.tokenizer.train_tokenizer import main as tok_main
    tok_main()


def main():
    ap = argparse.ArgumentParser(prog="zenith")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model from a config file")
    p_train.add_argument("config")
    p_train.set_defaults(func=cmd_train)

    p_chat = sub.add_parser("chat", help="Interactive chat with a trained model")
    p_chat.add_argument("checkpoint")
    p_chat.add_argument("--tokenizer", required=True)
    p_chat.add_argument("--max-new-tokens", type=int, default=200)
    p_chat.add_argument("--temperature", type=float, default=0.8)
    p_chat.add_argument("--top-k", type=int, default=50)
    p_chat.add_argument("--top-p", type=float, default=0.95)
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint on a validation set")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("--val-path", required=True)
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.set_defaults(func=cmd_evaluate)

    p_tok = sub.add_parser("tokenize", help="Train a BPE tokenizer")
    p_tok.add_argument("--input", nargs="+", required=True)
    p_tok.add_argument("--output", required=True)
    p_tok.add_argument("--vocab-size", type=int, default=8192)
    p_tok.set_defaults(func=cmd_tokenize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
