import os
import pickle
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------- CONFIG (adjust if your folders differ) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # MindLayer/
BACKEND_DIR = PROJECT_ROOT / "backend"

UPLOADS_DIR = BACKEND_DIR / "uploads"          # where uploaded files are saved
DATA_DIR = BACKEND_DIR / "data"                # where we save faiss + metadata
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking params (simple + effective baseline)
CHUNK_CHARS = 900
OVERLAP_CHARS = 150
# ----------------------------------------------------------


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n\n".join(text_parts)


def read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return ""


def chunk_text(text: str):
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - OVERLAP_CHARS
        if i < 0:
            i = 0
        if end == n:
            break
    return chunks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    if not UPLOADS_DIR.exists():
        raise FileNotFoundError(f"Uploads directory not found: {UPLOADS_DIR}")

    files = []
    for ext in ("*.pdf", "*.docx", "*.txt"):
        files.extend(sorted(UPLOADS_DIR.glob(ext)))

    if not files:
        raise RuntimeError(f"No files found in {UPLOADS_DIR}. Upload at least one PDF/DOCX/TXT first.")

    print(f"Found {len(files)} files in {UPLOADS_DIR}")

    all_chunks = []
    metadata = []

    for f in files:
        try:
            text = extract_text(f)
        except Exception as e:
            print(f"Skipping {f.name} (parse error): {e}")
            continue

        chunks = chunk_text(text)
        print(f"{f.name}: {len(chunks)} chunks")

        ts = datetime.fromtimestamp(f.stat().st_mtime).isoformat()

        for j, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadata.append({
                "id": len(metadata),
                "text": ch,
                "source_file": f.name,
                "chunk_index": j,
                "timestamp": ts,
            })

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check your file parsing.")

    print(f"Total chunks: {len(all_chunks)}")

    # Embed
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    emb = embedder.encode(all_chunks, show_progress_bar=True)
    emb = np.array(emb, dtype=np.float32)

    # Use cosine similarity via normalized inner product
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Saved:")
    print(" -", FAISS_INDEX_PATH)
    print(" -", METADATA_PATH)
    print("FAISS vectors:", index.ntotal)


if __name__ == "__main__":
    main()
