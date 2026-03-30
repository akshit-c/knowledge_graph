import os
import sqlite3
from pathlib import Path
from neo4j import GraphDatabase

DB_PATH = Path("backend/data/memory.sqlite")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mindlayer123")

def get_sqlite_tables(con):
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at {DB_PATH.resolve()}")

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    tables = get_sqlite_tables(con)
    print("SQLite tables:", tables)

    if "documents" not in tables:
        raise RuntimeError("Expected table 'documents' not found in SQLite.")
    if "chunks" not in tables:
        raise RuntimeError("Expected table 'chunks' not found in SQLite.")

    docs = cur.execute("SELECT * FROM documents").fetchall()
    chunks = cur.execute("SELECT * FROM chunks").fetchall()

    print(f"Found documents: {len(docs)}")
    print(f"Found chunks: {len(chunks)}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def upsert_document(tx, d_row):
        # Convert sqlite3.Row -> plain dict so we can use .get safely
        d = dict(d_row)
        tx.run(
            """
            MERGE (doc:Document {doc_id: $doc_id})
            SET doc.original_filename = $original_filename,
                doc.file_ext = $file_ext,
                doc.sha256 = $sha256,
                doc.created_at = $created_at,
                doc.stored_file_path = $stored_file_path,
                doc.stored_text_path = $stored_text_path,
                doc.chunk_count = $chunk_count,
                doc.summary = $summary
            """,
            doc_id=d["doc_id"],
            original_filename=d.get("original_filename"),
            file_ext=d.get("file_ext"),
            sha256=d.get("sha256"),
            created_at=d.get("created_at"),
            stored_file_path=d.get("stored_file_path"),
            stored_text_path=d.get("stored_text_path"),
            chunk_count=d.get("chunk_count"),
            summary=d.get("summary"),
        )

    def upsert_chunk_and_link(tx, c_row):
        # Convert sqlite3.Row -> plain dict so we can use .get safely
        c = dict(c_row)
        tx.run(
            """
            MATCH (doc:Document {doc_id: $doc_id})
            MERGE (ch:Chunk {chunk_id: $chunk_id})
            SET ch.doc_id = $doc_id,
                ch.chunk_index = $chunk_index,
                ch.text = $text,
                ch.created_at = $created_at
            MERGE (doc)-[:HAS_CHUNK]->(ch)
            """,
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            chunk_index=c.get("chunk_index"),
            text=c.get("text"),
            created_at=c.get("created_at"),
        )

    with driver.session() as session:
        # Documents first
        for d in docs:
            session.execute_write(upsert_document, d)

        # Then chunks + edges
        for c in chunks:
            session.execute_write(upsert_chunk_and_link, c)

    driver.close()
    con.close()

    print("Done: SQLite -> Neo4j loaded successfully.")
    print("Try in Neo4j Browser:")
    print("MATCH (d:Document) RETURN d LIMIT 5;")
    print("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d.doc_id, count(c) LIMIT 10;")

if __name__ == "__main__":
    main()
