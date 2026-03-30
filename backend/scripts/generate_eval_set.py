#!/usr/bin/env python3
"""
Generate a research-grade evaluation set (100 queries) automatically from your ingested PDFs.

Design goals:
- 100 total examples
- Balanced query types (definition, list, comparison, why/how, factual)
- Mix of answerable and unanswerable (default 60/40)
- Every answerable example includes evidence (doc_id, chunk_id) and gold answer
- Unanswerable examples have expected_answer = NOT_IN_MEMORY

Usage:
  python backend/scripts/generate_eval_set.py --n 100 --pos 60 --neg 40
"""

import argparse
import csv
import json
import os
import random
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NOT_IN_MEMORY = "NOT_IN_MEMORY"

DB_PATH_DEFAULT = Path("backend/data/memory.sqlite")
OUT_DIR_DEFAULT = Path("backend/outputs/eval")

# Ensure repo root is on sys.path so `import backend...` works
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import your local LLM generate (MLX/Ollama wrapper)
# Assumption: you already have backend/services/local_llm_mlx.py with generate(prompt)->str
from backend.services.local_llm_mlx import generate


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _safe_json_extract(text: str) -> Optional[dict]:
    """
    Extract a JSON object from model output robustly.
    """
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object substring
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _table_has_column(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {r[1] for r in rows}
    return col in cols


def _pick_existing_table(cur: sqlite3.Cursor, candidates: List[str]) -> Optional[str]:
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for c in candidates:
        if c in tables:
            return c
    return None


@dataclass
class ChunkRow:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def load_chunks(db_path: Path) -> Tuple[List[ChunkRow], Dict[str, Dict[str, Any]]]:
    """
    Loads chunks + document metadata (summary + filename if present).
    Returns: (chunks, docs_meta_by_id)
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    docs_table = _pick_existing_table(cur, ["documents", "document", "docs"])
    chunks_table = _pick_existing_table(cur, ["chunks", "chunk"])
    if not docs_table or not chunks_table:
        raise RuntimeError(f"Could not find expected tables. Found docs_table={docs_table}, chunks_table={chunks_table}")

    # documents columns we expect from your schema dump:
    # doc_id, original_filename, file_ext, sha256, created_at, stored_file_path, stored_text_path, chunk_count, summary
    doc_cols = [r[1] for r in cur.execute(f"PRAGMA table_info({docs_table})").fetchall()]

    docs = {}
    doc_rows = cur.execute(f"SELECT * FROM {docs_table}").fetchall()
    for row in doc_rows:
        rec = dict(zip(doc_cols, row))
        did = rec.get("doc_id")
        if not did:
            continue
        docs[did] = rec

    # chunks columns (robust)
    chunk_cols = [r[1] for r in cur.execute(f"PRAGMA table_info({chunks_table})").fetchall()]
    # We need chunk_id/doc_id/chunk_index/text
    required = ["chunk_id", "doc_id", "chunk_index", "text"]
    for r in required:
        if r not in chunk_cols:
            raise RuntimeError(f"Missing column '{r}' in table '{chunks_table}'. Columns: {chunk_cols}")

    rows = cur.execute(f"SELECT chunk_id, doc_id, chunk_index, text FROM {chunks_table}").fetchall()
    con.close()

    chunks = []
    for (chunk_id, doc_id, chunk_index, text) in rows:
        if not text or not str(text).strip():
            continue
        chunks.append(ChunkRow(
            chunk_id=str(chunk_id),
            doc_id=str(doc_id),
            chunk_index=int(chunk_index),
            text=str(text).strip()
        ))

    # Basic sanity
    if not chunks:
        raise RuntimeError("No chunks found in DB. Ingest PDFs first.")

    return chunks, docs


def make_question_from_chunk(chunk: ChunkRow, query_type: str) -> Optional[dict]:
    """
    Uses the local LLM to create ONE answerable QA pair from the chunk.
    Output JSON schema:
      { "question": "...", "answer": "...", "answer_type": "short|list|sentence" }
    """
    prompt = f"""
You are generating a benchmark QA dataset for a memory-grounded assistant.
The assistant must answer ONLY using the provided context.

TASK:
Create exactly ONE question of type: {query_type}
The question MUST be answerable using ONLY the context.
Then provide a gold answer that is STRICTLY grounded in the context.
If the context does not support this query type, return: {{"skip": true}}

RULES:
- Output ONLY valid JSON.
- The "answer" should be concise and directly supported by the context.
- Prefer extracting phrases/sentences from the context (minimal paraphrase).
- No external facts.

Return JSON with keys:
- question (string)
- answer (string)
- answer_type (string: "short"|"list"|"sentence")
- skip (boolean, optional)

CONTEXT:
{chunk.text}
""".strip()

    raw = generate(prompt)
    js = _safe_json_extract(raw)
    if not js or js.get("skip") is True:
        return None

    q = (js.get("question") or "").strip()
    a = (js.get("answer") or "").strip()

    if not q or not a:
        return None
    if a.upper() == NOT_IN_MEMORY:
        # shouldn't happen for positives
        return None

    return {
        "question": q,
        "expected_answer": a,
        "is_answerable": 1,
        "query_type": query_type,
        "evidence": [{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id}],
    }


def make_unanswerable_question(topic_hint: str) -> Optional[dict]:
    """
    Create an unanswerable question: plausible but likely not in memory.
    Output:
      { question: "...", expected_answer: "NOT_IN_MEMORY", is_answerable: 0 }
    """
    prompt = f"""
You are generating "unanswerable" benchmark queries for a memory-grounded assistant.
The assistant must output exactly: {NOT_IN_MEMORY} if the answer is not in memory.

TASK:
Write ONE question that sounds reasonable given the topic hint below,
BUT would typically require external knowledge (names, dates, current facts, statistics, biographies, CEOs, prices, etc).
Avoid asking about generic definitions that might appear in documents.

RULES:
- Output ONLY valid JSON.
- The question must NOT be answerable from a typical research PDF set unless that exact fact was stated.
- Keep it short and natural.

Return JSON:
- question (string)

TOPIC HINT:
{topic_hint}
""".strip()

    raw = generate(prompt)
    js = _safe_json_extract(raw)
    if not js:
        return None
    q = (js.get("question") or "").strip()
    if not q:
        return None

    return {
        "question": q,
        "expected_answer": NOT_IN_MEMORY,
        "is_answerable": 0,
        "query_type": "unanswerable_external",
        "evidence": [],
    }


def build_eval_set(
    db_path: Path,
    out_dir: Path,
    n_total: int = 100,
    n_pos: int = 60,
    seed: int = 42
) -> Tuple[Path, Path]:
    random.seed(seed)

    chunks, docs = load_chunks(db_path)

    # Prefer “rich” chunks (length filter) to make good questions
    rich_chunks = [c for c in chunks if 400 <= len(c.text) <= 1800]
    if len(rich_chunks) < max(20, n_pos):
        rich_chunks = chunks  # fallback

    # Query type mix (paper-friendly)
    query_types = [
        "definition",
        "factual",
        "list",
        "comparison",
        "why/how",
    ]

    # Allocate positives across types evenly
    per_type = max(1, n_pos // len(query_types))
    pos_targets = {t: per_type for t in query_types}
    # distribute remainder
    remainder = n_pos - per_type * len(query_types)
    for t in query_types:
        if remainder <= 0:
            break
        pos_targets[t] += 1
        remainder -= 1

    examples: List[Dict[str, Any]] = []
    used_questions = set()

    # Generate positives
    random.shuffle(rich_chunks)
    idx = 0

    for qtype, target in pos_targets.items():
        made = 0
        tries = 0
        while made < target and tries < 500:
            tries += 1
            if idx >= len(rich_chunks):
                idx = 0
                random.shuffle(rich_chunks)
            chunk = rich_chunks[idx]
            idx += 1

            ex = make_question_from_chunk(chunk, qtype)
            if not ex:
                continue

            q = ex["question"]
            if q.lower() in used_questions:
                continue

            used_questions.add(q.lower())
            examples.append(ex)
            made += 1

    # Generate negatives
    n_neg = n_total - len([e for e in examples if e["is_answerable"] == 1])
    if n_neg < 0:
        # trim if over-generated
        examples = examples[:n_total]
        n_neg = 0

    # Use doc summaries / filenames as topic hints
    topic_hints = []
    for did, meta in docs.items():
        s = (meta.get("summary") or "").strip()
        fn = (meta.get("original_filename") or "").strip()
        hint = s if s else fn
        if hint:
            topic_hints.append(hint)

    # fallback to random chunk snippets
    if not topic_hints:
        topic_hints = [c.text[:200] for c in random.sample(chunks, min(30, len(chunks)))]

    made_neg = 0
    tries = 0
    while made_neg < n_neg and tries < 500:
        tries += 1
        hint = random.choice(topic_hints)
        ex = make_unanswerable_question(hint)
        if not ex:
            continue
        q = ex["question"]
        if q.lower() in used_questions:
            continue
        used_questions.add(q.lower())
        examples.append(ex)
        made_neg += 1

    # Final trim/pad (should end exactly n_total)
    random.shuffle(examples)
    examples = examples[:n_total]

    # Add IDs + metadata
    for i, ex in enumerate(examples, start=1):
        ex["qid"] = f"Q{i:03d}"
        ex["created_at"] = _now_iso()
        ex["source"] = "auto_from_pdfs"
        # Optional: add doc summary for answerable ones (helps later analysis)
        if ex["is_answerable"] == 1 and ex.get("evidence"):
            did = ex["evidence"][0]["doc_id"]
            ex["doc_summary"] = (docs.get(did, {}).get("summary") or "").strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"eval_{n_total}.jsonl"
    csv_path = out_dir / f"eval_{n_total}.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # CSV (flatten)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "is_answerable", "query_type", "question", "expected_answer", "evidence_doc_id", "evidence_chunk_id"])
        for ex in examples:
            ev_doc = ""
            ev_chunk = ""
            if ex.get("evidence"):
                ev_doc = ex["evidence"][0].get("doc_id", "")
                ev_chunk = ex["evidence"][0].get("chunk_id", "")
            w.writerow([ex["qid"], ex["is_answerable"], ex["query_type"], ex["question"], ex["expected_answer"], ev_doc, ev_chunk])

    return jsonl_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=str(DB_PATH_DEFAULT))
    ap.add_argument("--out", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pos", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db = Path(args.db)
    out = Path(args.out)
    n_total = args.n
    n_pos = args.pos

    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    jsonl_path, csv_path = build_eval_set(db, out, n_total=n_total, n_pos=n_pos, seed=args.seed)

    print("✅ Generated evaluation set")
    print(" - JSONL:", jsonl_path)
    print(" - CSV:  ", csv_path)

    # quick stats
    pos = 0
    neg = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex["is_answerable"] == 1:
                pos += 1
            else:
                neg += 1
    print(f"Stats: total={pos+neg} answerable={pos} unanswerable={neg}")


if __name__ == "__main__":
    main()

