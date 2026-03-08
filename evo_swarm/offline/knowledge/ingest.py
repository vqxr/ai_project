from __future__ import annotations

import os
from collections.abc import Iterable

from evo_swarm.offline.config import OfflineSwarmConfig
from evo_swarm.offline.knowledge.store import KnowledgeStore, is_probably_text_file, sha256_file

# Optional PDF support via PyMuPDF
try:
    import pymupdf  # type: ignore[import-untyped]
    _HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore[no-redef, import-untyped]
        _HAS_PYMUPDF = True
    except ImportError:
        _HAS_PYMUPDF = False


def read_pdf_text(path: str) -> str:
    """Extract all text from a PDF using PyMuPDF, page by page."""
    if not _HAS_PYMUPDF:
        raise ImportError("pymupdf is required for PDF ingestion. Install with: pip install pymupdf")
    doc = pymupdf.open(path)
    pages: list[str] = []
    for page_num in range(len(doc)):
        text = str(doc[page_num].get_text("text"))  # type: ignore[attr-defined]
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def read_file_text(path: str) -> str:
    """Read text from a file, dispatching to PDF extraction when needed."""
    _, ext = os.path.splitext(path.lower())
    if ext == ".pdf":
        return read_pdf_text(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[int, str]]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be >= 0 and < chunk_chars")

    chunks: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def iter_paths(root: str) -> Iterable[str]:
    if os.path.isfile(root):
        yield root
        return

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def ingest_path(store: KnowledgeStore, config: OfflineSwarmConfig, root: str) -> dict:
    """
    Ingest .txt/.md/.rst/.tex/.pdf files into the knowledge store.
    PDFs are extracted to text via PyMuPDF.
    """
    ingested = 0
    skipped = 0
    errors: list[str] = []

    for path in iter_paths(root):
        if not is_probably_text_file(path):
            skipped += 1
            continue
        try:
            text = read_file_text(path)
            digest = sha256_file(path)
            doc_id = store.upsert_document(path=os.path.abspath(path), sha256=digest)
            chunks = chunk_text(
                text=text,
                chunk_chars=config.chunk_chars,
                overlap_chars=config.chunk_overlap_chars,
            )
            store.replace_chunks(doc_id=doc_id, chunks=chunks)
            ingested += 1
        except Exception as e:
            errors.append(f"{path}: {e}")

    return {"ingested": ingested, "skipped": skipped, "errors": errors}

