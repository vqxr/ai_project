import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from data import MemmapDataset, get_batch
from model import GPT, GPTConfig
from tokenizers import Tokenizer
from tqdm import tqdm


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_loss(model, train_mm, val_mm, cfg, batch_size, device, iters=50):
    model.eval()
    out = {}
    with torch.no_grad():
        for split, mm in [("train", train_mm), ("val", val_mm)]:
            losses = []
            for _ in range(iters):
                xb, yb = get_batch(mm, cfg.block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--resume", action="store_true")

    # Defaults tuned for 16GB unified memory.
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--micro_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=16)  # effective batch = 128 sequences

    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_path = data_dir / "tokenizer.json"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    meta_path = data_dir / "meta.json"
    if not tok_path.exists() or not train_path.exists() or not val_path.exists():
        raise SystemExit("Missing tokenizer.json/train.bin/val.bin. Run prepare_data.py first.")

    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()

    device = pick_device()
    print(f"Device: {device}")

    train_ds = MemmapDataset.open(train_path, meta_path=meta_path)
    val_ds = MemmapDataset.open(val_path, meta_path=meta_path)
    train_mm = train_ds.memmap()
    val_mm = val_ds.memmap()

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step0 = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step0 = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from step {step0}")

    # MPS autocast is supported; keep it conservative.
    use_amp = device.type in {"cuda", "mps"}
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def autocast_ctx():
        if not use_amp:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    run_cfg = vars(args) | {"vocab_size": vocab_size, "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2) + "\n", encoding="utf-8")

    t0 = time.time()
    pbar = tqdm(range(step0, args.max_steps), desc="train")
    for step in pbar:
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum_steps):
            xb, yb = get_batch(train_mm, cfg.block_size, args.micro_batch_size, device)
            with autocast_ctx():
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum_steps
            loss_accum += loss.item()

            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if device.type == "cuda":
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            toks_per_step = cfg.block_size * args.micro_batch_size * args.grad_accum_steps
            pbar.set_postfix(loss=f"{loss_accum:.4f}", tok_s=f"{toks_per_step / max(1e-9, elapsed):.0f}")
            t0 = time.time()

        if (step + 1) % args.eval_every == 0:
            losses = estimate_loss(model, train_mm, val_mm, cfg, batch_size=args.micro_batch_size, device=device, iters=30)
            val_loss = losses["val"]
            msg = f"step {step+1}: train {losses['train']:.4f} | val {val_loss:.4f} | ppl {math.exp(min(20, val_loss)):.2f}"
            print("\n" + msg)
            (out_dir / "metrics.jsonl").open("a", encoding="utf-8").write(json.dumps({"step": step + 1, **losses}) + "\n")
            best_val = min(best_val, val_loss)

        if (step + 1) % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(), "step": step + 1, "best_val": best_val, "cfg": cfg.__dict__},
                ckpt_path,
            )

    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict(), "step": args.max_steps, "best_val": best_val, "cfg": cfg.__dict__},
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
