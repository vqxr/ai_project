import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Hello")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "ckpt.pt"
    cfg_path = run_dir / "config.json"
    if not ckpt_path.exists() or not cfg_path.exists():
        raise SystemExit("Missing ckpt.pt/config.json. Train first.")

    # tokenizer is stored with the dataset; infer it from config's data_dir.
    cfg_json = __import__("json").loads(cfg_path.read_text(encoding="utf-8"))
    data_dir = Path(cfg_json["data_dir"])
    tok_path = data_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()

    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ckpt.get("cfg", {})
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=int(mcfg.get("block_size", cfg_json.get("block_size", 512))),
        n_layer=int(mcfg.get("n_layer", cfg_json.get("n_layer", 8))),
        n_head=int(mcfg.get("n_head", cfg_json.get("n_head", 8))),
        n_embd=int(mcfg.get("n_embd", cfg_json.get("n_embd", 512))),
        dropout=float(mcfg.get("dropout", cfg_json.get("dropout", 0.0))),
    )
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    ids = tokenizer.encode(args.prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]

    for _ in range(args.max_new_tokens):
        x_cond = x[:, -cfg.block_size :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(1e-6, args.temperature)
        if args.top_k > 0:
            v, _ = torch.topk(logits, k=min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    out = tokenizer.decode(x[0].tolist())
    print(out)


if __name__ == "__main__":
    main()

