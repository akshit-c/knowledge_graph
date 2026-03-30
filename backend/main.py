from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import upload, search, query
from backend.routers.local_llm import router as local_llm_router
from dotenv import load_dotenv
from backend.routers import documents
from backend.routers import chat






load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(upload.router)
app.include_router(search.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(local_llm_router)

@app.get("/")
def root():
    return {"message": "MindLayer backend is running"}

