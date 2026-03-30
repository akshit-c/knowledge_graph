from __future__ import annotations
import hashlib
from pathlib import Path
from datetime import datetime
import shutil
from backend.services.summarizer import summarize_document
from backend.services.memory_db import init_db, upsert_document, insert_chunks
import sqlite3

MEMORY_ROOT = Path(__file__).resolve().parents[1] / "memory_store"
FILES_DIR = MEMORY_ROOT / "files"
DOCS_DIR = MEMORY_ROOT / "documents"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "memory.sqlite"

# Deterministic chunking: stable, simple, reproducible
CHUNK_CHARS = 900
OVERLAP_CHARS = 150

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def _read_pdf(path: Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n\n".join(parts)

def _read_docx(path: Path) -> str:
    import docx
    d = docx.Document(str(path))
    return "\n".join([p.text for p in d.paragraphs if p.text.strip()])

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(file_path)
    if ext == ".docx":
        return _read_docx(file_path)
    if ext == ".txt":
        return _read_txt(file_path)
    raise ValueError(f"Unsupported file type: {ext}")

def normalize_text(t: str) -> str:
    # Stable whitespace normalization helps reproducibility
    return " ".join(t.split()).strip()

def save_summary(doc_id: str, summary: str):
    """Save document summary to SQLite database."""
    if not summary:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE documents SET summary=? WHERE doc_id=?", (summary, doc_id))
    con.commit()
    con.close()

def chunk_text(text: str) -> list[dict]:
    chunks = []
    i = 0
    n = len(text)
    idx = 0
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append({
                "chunk_index": idx,
                "text": chunk,
                "start_char": i,
                "end_char": end,
            })
            idx += 1
        i = end - OVERLAP_CHARS
        if i < 0:
            i = 0
        if end == n:
            break
    return chunks

def ingest_file(tmp_upload_path: Path, original_filename: str) -> dict:
    """
    Creates a canonical, auditable memory record from an uploaded file.
    Returns: {doc_id, chunk_count, stored_file_path, stored_text_path}
    """
    init_db()
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ext = tmp_upload_path.suffix.lower()
    file_hash = sha256_file(tmp_upload_path)

    # Deterministic doc_id from content hash
    doc_id = f"doc_{file_hash[:16]}"

    stored_file = FILES_DIR / f"{doc_id}{ext}"
    shutil.copy2(tmp_upload_path, stored_file)

    raw_text = extract_text(stored_file)
    text = normalize_text(raw_text)

    stored_text = DOCS_DIR / f"{doc_id}.txt"
    stored_text.write_text(text, encoding="utf-8")

    chunks = chunk_text(text)
    # Deterministic chunk_id = hash(doc_id + chunk_index + boundaries)
    chunk_rows = []
    for c in chunks:
        cid_src = f"{doc_id}|{c['chunk_index']}|{c['start_char']}|{c['end_char']}"
        chunk_id = "ch_" + hashlib.sha256(cid_src.encode("utf-8")).hexdigest()[:16]
        chunk_rows.append({
            "chunk_id": chunk_id,
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "start_char": c["start_char"],
            "end_char": c["end_char"],
        })

    upsert_document({
        "doc_id": doc_id,
        "original_filename": original_filename,
        "file_ext": ext,
        "sha256": file_hash,
        "created_at": datetime.utcnow().isoformat(),
        "stored_file_path": str(stored_file),
        "stored_text_path": str(stored_text),
        "chunk_count": len(chunk_rows),
    })

    # 1) create summary from full extracted text (read from stored file)
    full_text = stored_text.read_text(encoding="utf-8", errors="ignore")
    summary = summarize_document(full_text)

    # 2) store into sqlite
    save_summary(doc_id, summary)

    insert_chunks(doc_id, chunk_rows)

    return {
        "doc_id": doc_id,
        "chunk_count": len(chunk_rows),
        "stored_file_path": str(stored_file),
        "stored_text_path": str(stored_text),
    }
