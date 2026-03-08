# Local LLM (from scratch) on Mac (16GB)

This project trains a small GPT-style transformer **from scratch** on a local text corpus (your papers/books after you convert them to `.txt`).

It is designed to be workable on a MacBook Air with **16GB unified RAM** by using:
- a small-ish model (default ~30M params)
- a memory-mapped token dataset (doesn’t load everything into RAM)
- gradient accumulation (small microbatches)
- Apple Silicon acceleration when available (`mps`)

## 0) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r local_llm/requirements.txt
```

## 1) Put your data in

Place plain text files here:

```
local_llm/data/text/**/*.txt
```

If your sources are PDFs, convert them to text first (recommended), or use `prepare_data.py`'s optional PDF extraction.

## 2) Build tokenizer + token dataset

```bash
python3 local_llm/scripts/prepare_data.py \
  --text_dir local_llm/data/text \
  --out_dir local_llm/data/processed \
  --vocab_size 16000
```

Outputs:
- `local_llm/data/processed/tokenizer.json`
- `local_llm/data/processed/train.bin`
- `local_llm/data/processed/val.bin`

## 3) Train

```bash
python3 local_llm/scripts/train.py \
  --data_dir local_llm/data/processed \
  --out_dir local_llm/runs/run1
```

Resume:

```bash
python3 local_llm/scripts/train.py \
  --data_dir local_llm/data/processed \
  --out_dir local_llm/runs/run1 \
  --resume
```

## 4) Sample text

```bash
python3 local_llm/scripts/sample.py \
  --run_dir local_llm/runs/run1 \
  --prompt "In this paper, we propose" \
  --max_new_tokens 200
```

## Notes (practical limits)

- Training “a new model” locally is realistic for **toy → small** models. You can still make something useful for *your* domain, but it won’t match large frontier models.
- If you want the best “answer from your library” experience, add RAG later (we can do that after the base model trains).
