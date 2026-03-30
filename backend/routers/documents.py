from fastapi import APIRouter, HTTPException
import sqlite3
from pathlib import Path

router = APIRouter()
DB_PATH = Path("backend/data/memory.sqlite")

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

@router.get("/documents/recent")
def recent_documents(limit: int = 10):
    con = db()
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT doc_id, original_filename, file_ext, created_at, chunk_count, summary
        FROM documents
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    con.close()
    return {"documents": [dict(r) for r in rows]}

@router.get("/documents/{doc_id}")
def get_document(doc_id: str):
    con = db()
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT doc_id, original_filename, file_ext, created_at, chunk_count, summary
        FROM documents
        WHERE doc_id = ?
        """,
        (doc_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return dict(row)
