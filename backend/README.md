# MindLayer Backend

FastAPI backend for the MindLayer application with document processing and AI query capabilities.

## Quick Start

### 1. Navigate to the backend directory
```bash
cd backend
```

### 2. Activate the virtual environment
```bash
source env/bin/activate
```

### 3. Make sure you have a `.env` file
Create a `.env` file in the `backend` directory with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Start the server

**Option A: Development mode (with auto-reload)**
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option B: Production mode**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Option C: Using uvicorn directly**
```bash
uvicorn main:app --reload --port 8000
```

## Server will be available at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Available Endpoints

### Health Check
- `GET /` - Check if server is running

### Document Upload
- `POST /upload/` - Upload and embed documents (PDF, DOCX, TXT)

### Search
- `POST /search/` - Semantic search through uploaded documents

### Query
- `POST /query/` - Ask questions about uploaded documents using Gemini AI
- `GET /memory-status/` - Check memory status and document count

## Example Usage

### Check if server is running:
```bash
curl http://localhost:8000/
```

### Check memory status:
```bash
curl http://localhost:8000/memory-status/
```

### Upload a document:
```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@document.pdf"
```

### Search documents:
```bash
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "search term", "top_k": 5}'
```

### Ask a question:
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

## Troubleshooting

### Port already in use
If port 8000 is already in use, you can change it:
```bash
python -m uvicorn main:app --reload --port 8080
```

### Virtual environment not activating
Make sure you're in the `backend` directory and the `env` folder exists.

### Module not found errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Development Notes

- The server uses CORS middleware configured for `http://localhost:5173` (Vite default port)
- Documents are stored in memory (not persisted to disk)
- Vector embeddings are calculated using sentence-transformers
- AI queries are handled by Google's Gemini API

