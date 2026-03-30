from pathlib import Path
import pickle
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

DB_PATH = BACKEND_DIR / "data" / "memory.sqlite"
INDEX_DIR = BACKEND_DIR / "memory_store" / "indexes"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute("""
        SELECT chunk_id, doc_id, chunk_index, text, start_char, end_char
        FROM chunks
        ORDER BY doc_id, chunk_index
    """).fetchall()
    con.close()

    if not rows:
        raise RuntimeError("No chunks found in SQLite. Upload at least one file first.")

    texts = [r["text"] for r in rows]

    # Metadata we return as sources in /query
    metadata = []
    for r in rows:
        metadata.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "start_char": r["start_char"],
            "end_char": r["end_char"],
        })

    print(f"Embedding {len(texts)} chunks using {EMBED_MODEL}...")

    embedder = SentenceTransformer(EMBED_MODEL)
    emb = embedder.encode(texts, show_progress_bar=True)
    emb = np.array(emb, dtype=np.float32)

    # cosine similarity via normalized inner product
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Saved:")
    print(" -", FAISS_PATH)
    print(" -", META_PATH)
    print("Vectors:", index.ntotal, "Dim:", dim)

if __name__ == "__main__":
    main()
