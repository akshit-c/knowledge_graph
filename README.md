# Knowledge Graph Based Personalized Learning & Logical Decision Making

A personal AI memory system that provides ownership of your digital memory with reflection over time, emotional intelligence, and semantic understanding.

## Overview

It is an advanced personal knowledge management system that goes beyond traditional note-taking applications. Unlike ChatGPT or Notion AI, MindLayer provides complete ownership of your memory with sophisticated reflection capabilities, emotional intelligence, and semantic understanding.

## Key Features

- **Memory Ownership**: Complete control over your personal data
- **Temporal Reflection**: Track insights and growth over time
- **Emotional Intelligence**: Sentiment analysis and mood tracking
- **Semantic Understanding**: Advanced natural language processing
- **Flexible Deployment**: Cloud or local deployment options
- **Multi-format Support**: PDF, DOCX, Markdown file processing
- **Intelligent Search**: Semantic queries and natural language chat
- **Decision Support**: AI-powered reasoning and insights

## Technology Stack

### Frontend
- React 19
- Vite
- Tailwind CSS
- Modern JavaScript (ES6+)

### Backend
- Python 3.12+
- FastAPI
- Sentence Transformers
- PDF Plumber (PDF parsing)
- Python-docx (DOCX parsing)
- NumPy

## Getting Started

### Prerequisites

- **Node.js** (v20.19.0 or higher)
- **Python** (3.12 or higher)
- **Git**

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/akshit-c/MindLayer.git
cd mindlayer
```

2. **Set up the Backend:**

   Navigate to the backend directory and create a virtual environment:
   ```bash
   cd backend
   python -m venv env
   ```

   Activate the virtual environment:
   ```bash
   # On macOS/Linux:
   source env/bin/activate
   
   # On Windows:
   env\Scripts\activate
   ```

   Install Python dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart sentence-transformers pdfplumber python-docx numpy python-dotenv
   ```

3. **Set up the Frontend:**

   Navigate to the frontend directory:
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the Backend Server:**

   From the `backend` directory (with virtual environment activated):
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   The backend will be available at `http://localhost:8000`

2. **Start the Frontend Development Server:**

   From the `frontend` directory:
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

3. **Access the Application:**

   Open your browser and navigate to `http://localhost:5173` to use the MindLayer application.

### API Endpoints

The backend provides the following endpoints:

- `GET /` - Health check
- `POST /upload/` - Upload and process documents (PDF, DOCX, TXT)
- `POST /search/` - Search through uploaded documents
- `GET /query/` - Query the memory system

### File Upload Support

MindLayer currently supports the following file formats:
- **PDF** (.pdf) - Extracts text from PDF documents
- **DOCX** (.docx) - Extracts text from Word documents  
- **TXT** (.txt) - Plain text files

### Development

#### Backend Development
- The backend uses FastAPI for the REST API
- Document processing is handled by `services/parser.py`
- Semantic search is implemented in `services/embedder.py`
- All routes are organized in the `routers/` directory

#### Frontend Development
- Built with React 19 and Vite
- Uses Tailwind CSS for styling
- Components are in the `src/components/` directory

### Environment Variables

Create a `.env` file in the backend directory if needed:
```env
# Add any environment variables here
# For example, API keys or configuration settings
```

## Build Roadmap

### Phase 1: Vision & Planning (Weeks 1-2)
- Project setup and architecture design
- Feature scope definition
- Technology stack selection

### Phase 2: UI & Architecture (Weeks 3-4)
- Frontend layout development
- Backend API scaffolding
- Database schema design

### Phase 3: Parsing Engine (Weeks 5-6)
- PDF document processing
- DOCX file parsing
- Markdown content extraction

### Phase 4: Semantic Layer (Weeks 7-8)
- Embedding generation
- Vector database integration (FAISS/ChromaDB)
- Semantic indexing

### Phase 5: Search Interface (Weeks 9-10)
- Semantic query processing
- Natural language chat interface
- Advanced search algorithms

### Phase 6: Reasoning & Sentiment (Weeks 11-12)
- Decision support systems
- Tone and sentiment analysis
- Contextual understanding

### Phase 7: Insights Engine (Weeks 13-14)
- Mood tracking and analysis
- Automated summaries
- Interactive pinboard

### Phase 8: Final Build (Weeks 15-16)
- Comprehensive testing
- Performance optimization
- Documentation completion

## Project Team

- **Akshit Chaudhary** - A2305222548
- **Sarvjeet Sambyal** - A2305222541  
- **Karan Singh** - A2305222546

## References

- [OpenAI Research](https://openai.com/research)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers)
- [Weaviate Vector Database](https://weaviate.io)
- [LangChain Framework](https://python.langchain.com)
- [FAISS Vector Search](https://github.com/facebookresearch/faiss)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

## Philosophy

"Don't just talk to AI — co-evolve with it."

MindLayer represents a new paradigm in personal AI systems, where users maintain complete ownership of their digital memory while leveraging advanced AI capabilities for reflection, understanding, and growth.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

---

*MindLayer: Your Personal AI Memory System*
