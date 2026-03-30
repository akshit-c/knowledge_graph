from fastapi import APIRouter, File, UploadFile
from backend.services.embedder import embed_and_store, search_similar
from backend.services.parser import parse_file
from backend.services.ingest import ingest_file  # NEW
from datetime import datetime
from pydantic import BaseModel
from pathlib import Path
import tempfile
from backend.services.parser import parse_file_path



router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Processing upload: {file.filename}")

        # 1) Save uploaded file bytes to a temp file (so ingest can hash + store deterministically)
        suffix = Path(file.filename).suffix.lower() or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())

        # 2) Canonical ingestion (writes to memory_store + sqlite + deterministic chunks)
        ingest_info = ingest_file(tmp_path, file.filename)
        print(f"Ingested to MemoryStore: {ingest_info}")

        # 3) Keep your existing pipeline: parse + embed (for current FAISS runtime behavior)
        #    (Parse again is fine for now; later we will reuse stored extracted text for single-pass.)
        content, file_type = parse_file_path(tmp_path)

        print(f"Parsed file - Type: {file_type}, Content length: {len(content)}")

        if not content or not content.strip():
            return {
                "filename": file.filename,
                "type": file_type,
                "error": "File appears to be empty or could not be parsed",
                "embedded_chunks": 0,
                "doc_id": ingest_info.get("doc_id"),
            }

        metadata = {
            "filename": file.filename,
            "type": file_type,
            "upload_time": datetime.utcnow().isoformat(),
            "doc_id": ingest_info.get("doc_id"),  # NEW: stable ID
        }

        num_chunks = embed_and_store(content, metadata)

        print(f"Upload complete - Filename: {file.filename}, Chunks: {num_chunks}")

        return {
            "filename": file.filename,
            "type": file_type,
            "upload_time": metadata["upload_time"],
            "doc_id": metadata["doc_id"],
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "embedded_chunks": num_chunks,
            "content_length": len(content),
            "stored_file_path": ingest_info.get("stored_file_path"),
            "stored_text_path": ingest_info.get("stored_text_path"),
            "canonical_chunk_count": ingest_info.get("chunk_count"),
        }

    except Exception as e:
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return {
            "filename": file.filename if file else "unknown",
            "error": str(e),
            "embedded_chunks": 0
        }

@router.post("/search/")
async def search_documents(search_query: SearchQuery):
    results = search_similar(search_query.query, search_query.top_k)
    return {
        "query": search_query.query,
        "results": results
    }
