import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import admin
from app.core.rag_manager import RAGManager
from app.services.consumer import stream_consumer

rag_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_manager
    print("🚀 AI 서버 시작 중...")
    
    # RAG 매니저 초기화
    rag_manager = RAGManager()
    rag_manager.initialize()
    print("RAG 매니저 초기화 완료")
    
    # Redis Streams 컨슈머 시작
    consumer_task = asyncio.create_task(stream_consumer.start_consuming())
    print("Redis Streams 컨슈머 시작")
    
    yield
    
    # 종료 시 정리
    print("AI 서버 종료 중...")
    stream_consumer.stop()
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    print("AI 서버 종료 완료")

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

# 관리자 API만 포함
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
        "rag_initialized": rag_manager is not None and rag_manager.rag_chain is not None,
        "consumer_running": stream_consumer.running
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)