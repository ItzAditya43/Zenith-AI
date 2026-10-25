# Zenith keyboard model: handoff for MyKey

These are the files from steps 1–5 of ZENITH_PLAN.pdf, ready for `engine/ZenithModel.kt`.

| File | What it is |
|---|---|
| `zenith_kb.onnx` | 18.3 MB int8 model, opset 17. Runs on ONNX Runtime Mobile, CPU provider. |
| `zenith_kb.tok` | Unigram tokenizer, plain text. It loads without Python (format below). |
| `tokenizer_kb.json` | The same tokenizer in Hugging Face `tokenizers` format, for reference and tests. |
| `keyboard_eval.json`, `parity.json`, `finish.log` | Eval numbers (see the README section "Zenith keyboard model"). |

## Model I/O

| Name | Type and shape | Meaning |
|---|---|---|
| `tokens` | int64 `[1, T]` | New tokens, T ≥ 1. Feed the whole context once, then 1 token per keystroke. |
| `pos` | int64 `[1]` | Position of `tokens[0]`, which equals the number of tokens already in the cache. |
| `past_k`, `past_v` | float32 `[7, 1, 1, 64, 64]` | KV cache: layers × batch × kv_heads × 64 slots × head_dim. Start with zeros. |
| `logits` | float32 `[1, T, 32000]` | Next-token logits at every input position. |
| `new_k`, `new_v` | same as `past_*` | Pass these back in as `past_*` on the next call. |

- **Window:** there are 64 positions in total. When `pos + T > 64`, reset: zero the cache, set `pos=0`, and re-feed `<lang>` plus about the last 47 tokens. Positions are absolute (RoPE), so you can't just slide the window.
- **Language tag:** every context starts with `<en>` (id 4), `<es>` (5) or `<ja>` (6). Ids 7 and 8 (`<ru>`, `<zh>`) are reserved. `<eos>` (2) separates messages and lines. No `<bos>` is used.
- **Speed:** about 2.0–2.4 ms per single-token step on the PC CPU, using one ORT session with the default thread count. Not yet measured on the Xiaomi 14 Civi.
- **Phone RAM, estimated:** the transformer layers are stored int8 and expanded to fp32 when the session loads (about 30 MB), plus the int8 head and embedding (about 10 MB) and the cache (about 0.2 MB).

## Tokenizer (`zenith_kb.tok`)

```
line 1: ZENITH-UNIGRAM 1
line 2: JSON {vocab_size, unk_id, byte_fallback, normalizer:"NFKC",
              metaspace:{replacement:"▁", prepend_scheme:"always"},
              special:{name:id}, byte_offset}
line 3+: <piece>\t<score>   (one per id, in id order; \t \n \\ escaped)
```

How to encode:
1. NFKC-normalise the text.
2. Replace `' '` with `'▁'` and prepend one `'▁'`.
3. Run Viterbi (highest total score) over the pieces, skipping special and `<0xNN>` ids.
4. For a character no piece covers, emit its UTF-8 bytes as `byte_offset + byte`.

A 40-line standalone implementation of this matched the HF tokenizer on 1,200 of 1,200 held-out lines (400 each of en, es, ja). Decode by concatenating the pieces, turning `▁` into a space, and joining `<0xNN>` byte runs back into UTF-8.

## Two gotchas found during eval

1. **Token healing (important for Japanese).** If the context ends mid-token, for example 「今日」 when training always saw 「今日は」 as one piece, the predictions go bad. Drop the last context token and constrain the first predicted tokens to start with its text. This changed Japanese next-phrase hits on test cases from 2/9 to 6/9. For English and Spanish, contexts that end in a space are clean, but the word being typed is the same situation: constrain the candidates to the typed letters instead of feeding the partial word.
2. **Word boundaries.** English and Spanish word starts are tokens beginning with `▁`. A word is complete when the next most likely token starts with `▁`, is punctuation, or is `<eos>`. `zenith/evaluation/keyboard_eval.py` (`predict_words`, `predict_continuations`) is a working reference for both.

## Kanji re-ranking

Score each Mozc candidate `c` as log P(`<ja>` + context + c) and compare the totals. This is what `keyboard_eval.py` measures as kanji choice. Per the plan, your own past choices still take priority.
