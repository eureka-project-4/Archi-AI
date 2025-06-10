from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import chat, admin
from app.core.rag_manager import RAGManager
# from dotenv import load_dotenv
# load_dotenv(override=True)
rag_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_manager
    print("AI 서버 시작 중...")
    rag_manager = RAGManager()
    rag_manager.initialize()
    print("AI 서버 초기화 완료")
    yield
    print("AI 서버 종료")

app = FastAPI(
    title="AI 처리 서버",
    description="채팅 메시지 AI 처리 및 응답 생성",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.get("/")
async def root():
    return {"message": "AI 처리 서버가 실행 중입니다"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "service": "ai-server",
        "environment": settings.ENVIRONMENT,
        "rag_initialized": rag_manager is not None and rag_manager.rag_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)