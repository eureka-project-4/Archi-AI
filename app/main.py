#!/usr/bin/env python3
# app/main.py - 강제 출력 버전

import sys
import os
import asyncio
import signal
from pathlib import Path
from app.services.consumer import stream_consumer , image_stream_consumer
# 출력 강제 플러시
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def force_print(msg):
    """강제 출력 함수"""
    print(msg, flush=True)
    sys.stdout.flush()

force_print("🔥🔥🔥 MAIN.PY 시작됨! 🔥🔥🔥")
force_print(f"Python 버전: {sys.version}")
force_print(f"현재 디렉토리: {os.getcwd()}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.config import settings
    force_print("✅ settings 임포트 성공")
except Exception as e:
    force_print(f"❌ settings 임포트 실패: {e}")

try:
    from app.api import admin, chat
    force_print("✅ API 모듈 임포트 성공")
except Exception as e:
    force_print(f"❌ API 모듈 임포트 실패: {e}")

try:
    from app.core.rag_manager import RAGManager
    force_print("✅ RAG 매니저 임포트 성공")
except Exception as e:
    force_print(f"❌ RAG 매니저 임포트 실패: {e}")

try:
    from app.services.consumer import stream_consumer
    force_print("✅ 컨슈머 임포트 성공")
except Exception as e:
    force_print(f"❌ 컨슈머 임포트 실패: {e}")

# 전역 변수
rag_manager = None
consumer_task = None

def initialize_system():
    """시스템 초기화"""
    global rag_manager
    
    force_print("🚀🚀🚀 시스템 초기화 시작 🚀🚀🚀")
    
    try:
        # 경로 확인
        pricing_data_dir = Path("app/data/pricing")
        force_print(f"📁 데이터 경로 체크: {pricing_data_dir}")
        force_print(f"📁 경로 존재 여부: {pricing_data_dir.exists()}")
        
        if pricing_data_dir.exists():
            csv_files = list(pricing_data_dir.glob("*.csv"))
            force_print(f"📄 발견된 CSV: {[f.name for f in csv_files]}")
            settings.PRICING_DATA_DIR = str(pricing_data_dir)
        else:
            force_print("❌ pricing 디렉토리 없음")
        
        # RAG 매니저 초기화
        force_print("🤖 RAG 매니저 생성 중...")
        rag_manager = RAGManager()
        force_print("🤖 RAG 매니저 초기화 중...")
        rag_manager.initialize()
        force_print("✅✅✅ RAG 매니저 초기화 완료!")
        
        return True
        
    except Exception as e:
        force_print(f"❌❌❌ 초기화 실패: {e}")
        import traceback
        force_print(f"상세 오류: {traceback.format_exc()}")
        return False

async def start_consumer():
    """컨슈머 시작"""
    global consumer_task
    global image_consumer_task
    try:
        force_print("🎧 컨슈머 시작 시도...")
        
        if await stream_consumer.connect_redis():
            force_print("✅ Redis 연결 성공")
        else:
            force_print("❌ Redis 연결 실패")
            return False
        if await image_consumer_task.connect_redis():
            force_print("✅ Redis 연결 성공")
        else:
            force_print("❌ Redis 연결 실패")
            return False
        consumer_task = asyncio.create_task(stream_consumer.start_consuming())
        image_consumer_task = asyncio.create_task(image_stream_consumer.start_consuming())
        force_print("✅ 컨슈머 태스크 생성됨")
        
        return True
        
    except Exception as e:
        force_print(f"❌ 컨슈머 시작 실패: {e}")
        return False

# 앱 생성 전 초기화
force_print("⚡ 앱 생성 전 시스템 초기화...")
init_success = initialize_system()
force_print(f"⚡ 초기화 결과: {'성공' if init_success else '실패'}")

app = FastAPI(
    title="AI 처리 서버",
    description="채팅 메시지 AI 처리 및 응답 생성", 
    version="1.0.0"
)

force_print("📱 FastAPI 앱 생성됨")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    force_print("📡📡📡 STARTUP 이벤트 실행됨! 📡📡📡")
    await start_consumer()
    force_print("📡 STARTUP 완료")

@app.on_event("shutdown")
async def shutdown_event():
    global consumer_task
    force_print("🛑 SHUTDOWN 이벤트 실행됨")
    
    try:
        if stream_consumer:
            stream_consumer.stop()
        
        if consumer_task and not consumer_task.done():
            consumer_task.cancel()
            try:
                await asyncio.wait_for(consumer_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                force_print("⚠️ 컨슈머 강제 종료")
        
        force_print("✅ 종료 완료")
        
    except Exception as e:
        force_print(f"❌ 종료 오류: {e}")

# API 라우터
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "AI 처리 서버가 실행 중입니다",
        "rag_system": "OK" if rag_manager else "ERROR",
        "consumer": "OK" if stream_consumer and stream_consumer.running else "UNKNOWN",
        "initialization": "success" if init_success else "failed"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rag_initialized": rag_manager is not None,
        "consumer_running": stream_consumer.running if stream_consumer else False,
        "data_dir": getattr(settings, 'PRICING_DATA_DIR', 'unknown')
    }

@app.get("/debug")
async def debug_info():
    """디버깅 정보"""
    force_print("🔍 디버그 엔드포인트 호출됨")
    
    return {
        "rag_manager": rag_manager is not None,
        "rag_manager_type": type(rag_manager).__name__ if rag_manager else None,
        "consumer_running": stream_consumer.running if stream_consumer else False,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "pricing_dir_exists": Path("app/data/pricing").exists()
    }

force_print("🎯 모든 설정 완료, 서버 시작 대기 중...")

if __name__ == "__main__":
    import uvicorn
    force_print("🚀 직접 실행 모드")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)