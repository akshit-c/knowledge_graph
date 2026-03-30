from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "memory.sqlite"

def conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c

def init_db() -> None:
    c = conn()
    cur = c.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        original_filename TEXT NOT NULL,
        file_ext TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        stored_file_path TEXT NOT NULL,
        stored_text_path TEXT NOT NULL,
        chunk_count INTEGER NOT NULL,
        summary TEXT
    )
    """)
    
    # Migration: handle existing databases
    cur.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cur.fetchall()]
    
    # If old 'filename' column exists, rename it to 'original_filename'
    if "filename" in columns and "original_filename" not in columns:
        try:
            # Try RENAME COLUMN (SQLite 3.25.0+)
            cur.execute("ALTER TABLE documents RENAME COLUMN filename TO original_filename")
            c.commit()
        except sqlite3.OperationalError:
            # Fallback for older SQLite: recreate table
            cur.execute("""
                CREATE TABLE documents_new (
                    doc_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    stored_file_path TEXT NOT NULL,
                    stored_text_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    summary TEXT
                )
            """)
            cur.execute("""
                INSERT INTO documents_new 
                (doc_id, original_filename, file_ext, sha256, created_at, stored_file_path, stored_text_path, chunk_count, summary)
                SELECT doc_id, filename, file_ext, sha256, created_at, stored_file_path, stored_text_path, chunk_count, NULL
                FROM documents
            """)
            cur.execute("DROP TABLE documents")
            cur.execute("ALTER TABLE documents_new RENAME TO documents")
            c.commit()
    
    # Add summary column if it doesn't exist
    if "summary" not in columns:
        cur.execute("ALTER TABLE documents ADD COLUMN summary TEXT")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
    c.commit()
    c.close()

def upsert_document(row: Dict[str, Any]) -> None:
    c = conn()
    cur = c.cursor()
    cur.execute("""
    INSERT INTO documents (doc_id, original_filename, file_ext, sha256, created_at, stored_file_path, stored_text_path, chunk_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(doc_id) DO UPDATE SET
      original_filename=excluded.original_filename,
      file_ext=excluded.file_ext,
      sha256=excluded.sha256,
      created_at=excluded.created_at,
      stored_file_path=excluded.stored_file_path,
      stored_text_path=excluded.stored_text_path,
      chunk_count=excluded.chunk_count
    """, (
        row["doc_id"], row["original_filename"], row["file_ext"], row["sha256"], row["created_at"],
        row["stored_file_path"], row["stored_text_path"], row["chunk_count"]
    ))
    c.commit()
    c.close()

def insert_chunks(doc_id: str, chunk_rows: list[Dict[str, Any]]) -> None:
    c = conn()
    cur = c.cursor()
    for r in chunk_rows:
        cur.execute("""
        INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, text, start_char, end_char)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (r["chunk_id"], doc_id, r["chunk_index"], r["text"], r["start_char"], r["end_char"]))
    c.commit()
    c.close()
