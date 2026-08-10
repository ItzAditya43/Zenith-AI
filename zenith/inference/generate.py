import torch
import torch.nn.functional as F


def sample_logits(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float("inf")

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        mask = cum_probs - probs > top_p
        sorted_logits[mask] = -float("inf")
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.95, device="cuda", stream=False):
    model.eval()
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    ids = [bos_id] + tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    kv_cache = model.new_kv_cache(device=device)

    # prefill
    logits = model(tokens, kv_cache=kv_cache, start_pos=0)
    next_logits = logits[:, -1, :]

    generated = []
    pos = tokens.shape[1]
    for _ in range(max_new_tokens):
        next_id = sample_logits(next_logits, temperature, top_k, top_p)
        token_id = next_id.item()
        if token_id == eos_id:
            break
        generated.append(token_id)
        if stream:
            yield tokenizer.decode([token_id])

        logits = model(next_id, kv_cache=kv_cache, start_pos=pos)
        next_logits = logits[:, -1, :]
        pos += 1

    if not stream:
        yield tokenizer.decode(generated)


def generate_text(model, tokenizer, prompt, **kwargs):
    return "".join(generate(model, tokenizer, prompt, stream=False, **kwargs))
