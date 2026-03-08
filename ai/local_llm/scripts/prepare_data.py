import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def _iter_text_files(text_dir: Path):
    for p in sorted(text_dir.rglob("*.txt")):
        if p.is_file():
            yield p


def _iter_pdf_files(text_dir: Path):
    for p in sorted(text_dir.rglob("*.pdf")):
        if p.is_file():
            yield p


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _read_pdf_text(path: Path) -> str:
    # Best-effort extraction; for highest quality, convert PDFs to clean text yourself.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def build_tokenizer(corpus_paths, out_path: Path, vocab_size: int):
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    tok.train([str(p) for p in corpus_paths], trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def encode_to_memmap(tokenizer: Tokenizer, text: str, out_bin: Path, dtype=np.uint16):
    ids = tokenizer.encode(text).ids
    arr = np.asarray(ids, dtype=dtype)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(arr.shape[0],))
    mm[:] = arr[:]
    mm.flush()
    return arr.shape[0]

def write_ids_to_memmap(ids, out_bin: Path, dtype=np.uint16):
    arr = np.asarray(ids, dtype=dtype)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_bin, mode="w+", dtype=dtype, shape=(arr.shape[0],))
    mm[:] = arr[:]
    mm.flush()
    return arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir", type=str, required=True, help="Root dir containing .txt (and optionally .pdf)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for tokenizer + bin files")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--include_pdfs", action="store_true", help="Also extract text from PDFs found under text_dir")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_paths = list(_iter_text_files(text_dir))
    pdf_paths = list(_iter_pdf_files(text_dir)) if args.include_pdfs else []
    corpus_paths = txt_paths + pdf_paths

    if not corpus_paths:
        raise SystemExit(f"No .txt files found under {text_dir} (and include_pdfs={args.include_pdfs}).")

    # If PDFs are included, we extract them to temporary .txt files in out_dir for tokenizer training.
    extracted_paths = []
    if pdf_paths:
        extracted_dir = out_dir / "_extracted_pdfs"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for p in tqdm(pdf_paths, desc="Extract PDFs"):
            extracted = extracted_dir / (p.stem + ".txt")
            if not extracted.exists():
                extracted.write_text(_clean_text(_read_pdf_text(p)), encoding="utf-8", errors="ignore")
            extracted_paths.append(extracted)
        corpus_paths = txt_paths + extracted_paths

    tokenizer_path = out_dir / "tokenizer.json"
    print(f"Training tokenizer on {len(corpus_paths)} files -> {tokenizer_path}")
    tok = build_tokenizer(corpus_paths, tokenizer_path, vocab_size=args.vocab_size)

    # Concatenate into one big stream (simple + effective baseline).
    # For large corpora, you may want a multi-file shard format later.
    texts = []
    for p in tqdm(corpus_paths, desc="Read text"):
        try:
            texts.append(_clean_text(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            continue
    full_text = "\n\n".join(t for t in texts if t)

    # Split into train/val by tokens to avoid text-length bias.
    ids = tok.encode(full_text).ids
    n = len(ids)
    if n < 10_000:
        raise SystemExit(f"Corpus too small after tokenization ({n} tokens). Add more text.")

    split = int(n * (1.0 - args.val_ratio))
    train_ids = ids[:split]
    val_ids = ids[split:]

    # Choose smallest dtype that can hold vocab size.
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    print(f"Writing train tokens: {len(train_ids):,} -> {train_bin} ({dtype.__name__})")
    write_ids_to_memmap(train_ids, train_bin, dtype=dtype)

    print(f"Writing val tokens:   {len(val_ids):,} -> {val_bin} ({dtype.__name__})")
    write_ids_to_memmap(val_ids, val_bin, dtype=dtype)

    meta = {
        "vocab_size": vocab_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_files": len(corpus_paths),
        "num_tokens_total": n,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    meta_path = out_dir / "meta.json"
    import json

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
