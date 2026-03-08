#!/usr/bin/env python3
"""
Batch-convert all PDFs in a directory to plain text files.

Usage:
    python scripts/pdf_to_txt.py --input papers/ --output papers_txt/
    python scripts/pdf_to_txt.py --input single_file.pdf --output out/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import pymupdf  # PyMuPDF >= 1.24 exposes as 'pymupdf'
except ImportError:
    try:
        import fitz as pymupdf  # older PyMuPDF versions use 'fitz'
    except ImportError:
        sys.exit("PyMuPDF is required. Install with: pip install pymupdf")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF, page by page."""
    doc = pymupdf.open(str(pdf_path))
    pages: list[str] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def convert_directory(input_path: Path, output_dir: Path, *, recursive: bool = True) -> dict:
    """Walk input_path, convert every .pdf to .txt in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    errors: list[str] = []

    if input_path.is_file():
        pdf_files = [input_path]
    elif recursive:
        pdf_files = sorted(input_path.rglob("*.pdf"))
    else:
        pdf_files = sorted(input_path.glob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)
            if not text.strip():
                skipped += 1
                print(f"  SKIP (empty): {pdf_file}")
                continue

            # Mirror subdirectory structure inside output_dir
            if input_path.is_file():
                rel = pdf_file.name
            else:
                rel = pdf_file.relative_to(input_path)
            out_file = output_dir / Path(str(rel)).with_suffix(".txt")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(text, encoding="utf-8")

            converted += 1
            print(f"  OK: {pdf_file} -> {out_file}")
        except Exception as e:
            errors.append(f"{pdf_file}: {e}")
            print(f"  ERR: {pdf_file}: {e}")

    return {"converted": converted, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert PDFs to plain text for the Evo Swarm knowledge pipeline."
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory or single PDF file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for .txt files")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        sys.exit(f"Input path does not exist: {input_path}")

    print(f"Converting PDFs: {input_path} -> {output_dir}")
    t0 = time.time()
    result = convert_directory(input_path, output_dir, recursive=not args.no_recursive)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Converted: {result['converted']}")
    print(f"  Skipped:   {result['skipped']}")
    print(f"  Errors:    {len(result['errors'])}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"    - {err}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
