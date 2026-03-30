from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BACKEND_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BACKEND_DIR / "memory_store" / "indexes"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.pkl"

_EMBEDDER = None
_INDEX = None
_META = None

def _load():
    global _EMBEDDER, _INDEX, _META
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

    if _INDEX is None or _META is None:
        if not FAISS_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"Missing FAISS artifacts. Expected:\n{FAISS_PATH}\n{META_PATH}\n"
                "Run: python backend/scripts/build_faiss_from_sqlite.py"
            )

        _INDEX = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            _META = pickle.load(f)

def search(query: str, top_k: int = 5):
    _load()

    q = _EMBEDDER.encode([query]).astype(np.float32)
    faiss.normalize_L2(q)

    scores, idxs = _INDEX.search(q, top_k)

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        m = _META[i]
        results.append({
            "score": float(score),
            "chunk_id": m.get("chunk_id"),
            "doc_id": m.get("doc_id"),
            "chunk_index": m.get("chunk_index"),
            "text": m.get("text"),
        })
    return results
