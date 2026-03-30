# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

MindLayer is a personal AI memory system with a React 19 frontend and FastAPI backend. The system allows users to upload documents (PDF, DOCX, TXT), which are then embedded using sentence transformers for semantic search and AI-powered querying via Google's Gemini API.

**Key architectural principle**: Documents are stored in-memory only (not persisted to disk). The embedder service maintains two in-memory lists: `vectors` (384-dimensional embeddings) and `metadata_store` (chunks with metadata).

## Architecture

### Backend Structure
- **main.py**: FastAPI app with CORS configured for `http://localhost:5173`
- **routers/**: API endpoints organized by functionality
  - `upload.py`: File upload and search endpoints (`/upload/`, `/search/`)
  - `query.py`: AI query endpoints (`/query/`, `/memory-status/`, `/gemini-status/`)
  - `search.py`: Search functionality
- **services/**: Core business logic
  - `parser.py`: File parsing for PDF, DOCX, TXT using pdfplumber, python-docx
  - `embedder.py`: Embedding generation (all-MiniLM-L6-v2) and in-memory vector storage

### Frontend Structure
- **src/App.jsx**: Main app with FileUpload and MemoryChat components
- **src/components/**: React components
  - `FileUpload.jsx`: Document upload interface
  - `MemoryChat.jsx`: Chat interface for querying uploaded documents

### Key Technical Details
- Embeddings: 384-dimensional vectors using `all-MiniLM-L6-v2` model
- Chunking strategy: 512 characters per chunk (no overlap)
- Similarity: Cosine similarity for semantic search
- AI model: Google Gemini API (gemini-pro)

## Common Development Commands

### Backend

#### Start backend server (development with auto-reload)
```bash
cd backend
source env/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Start backend server (production)
```bash
cd backend
source env/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Install backend dependencies
```bash
cd backend
source env/bin/activate
pip install fastapi uvicorn python-multipart sentence-transformers pdfplumber python-docx numpy python-dotenv google-generativeai
```

#### API documentation available at
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Frontend

#### Start frontend development server
```bash
cd frontend
npm run dev
```

#### Build frontend for production
```bash
cd frontend
npm run build
```

#### Lint frontend code
```bash
cd frontend
npm run lint
```

#### Preview production build
```bash
cd frontend
npm run preview
```

#### Install frontend dependencies
```bash
cd frontend
npm install
```

## Environment Setup

### Backend Environment Variables
Create `backend/.env` with:
```env
GEMINI_API_KEY=your_api_key_here
```

The Gemini API key is required for AI-powered query functionality. Without it, document upload and semantic search will work, but the `/query/` endpoint will fail.

## Initial Setup

### First-time setup
1. Clone repository and navigate to project root
2. Set up backend:
   ```bash
   cd backend
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install fastapi uvicorn python-multipart sentence-transformers pdfplumber python-docx numpy python-dotenv google-generativeai
   ```
3. Create `backend/.env` with Gemini API key
4. Set up frontend:
   ```bash
   cd frontend
   npm install
   ```

### Running the application
Start both servers in separate terminals:

Terminal 1 (backend):
```bash
cd backend
source env/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):
```bash
cd frontend
npm run dev
```

Access application at http://localhost:5173

## API Endpoints

- `GET /` - Health check
- `POST /upload/` - Upload documents (multipart/form-data with "file" field)
- `POST /search/` - Semantic search (JSON: `{query: string, top_k: number}`)
- `POST /query/` - AI-powered question answering (JSON: `{question: string, top_k: number}`)
- `GET /memory-status/` - Check uploaded documents and chunk count
- `GET /gemini-status/` - Verify Gemini API configuration

## Technology Stack

### Backend
- Python 3.12+
- FastAPI (REST API framework)
- Sentence Transformers (all-MiniLM-L6-v2 for embeddings)
- PDFPlumber (PDF parsing)
- python-docx (DOCX parsing)
- NumPy (vector operations)
- Google Generative AI (Gemini API)

### Frontend
- React 19
- Vite (build tool and dev server)
- Tailwind CSS v4
- ESLint with React hooks and React refresh plugins

## Important Notes

### Memory Persistence
Documents and embeddings are stored **in-memory only**. Restarting the backend server clears all uploaded documents. This is by design for the current development phase.

### CORS Configuration
Backend CORS is hardcoded to allow `http://localhost:5173`. If running frontend on a different port, update `main.py` CORS configuration.

### Model Loading
The sentence transformer model (`all-MiniLM-L6-v2`) is downloaded on first run. Initial startup may take longer while the model is cached locally.

### Error Handling in query.py
The `/query/` endpoint includes fallback logic to try multiple Gemini model names and can dynamically discover available models if preferred names fail.
