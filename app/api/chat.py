from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

class VerifiedChatResponse(BaseModel):
    response: str
    used_knowledge: list[str] = []
    verification_status: Optional[str] = None
    mentioned_plans: list[str] = []
    confidence_score: float = 1.0
    message_type: str = "chat"

class VerificationReportRequest(BaseModel):
    user_id: str
    message: str

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """기본 채팅 메시지 처리 (기존 엔드포인트 유지)"""
    
    from app.main import rag_manager

    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    chat_history_str = ""
    for chat in request.chat_history or []:
        chat_history_str += f"사용자: {chat.user}\n"
        chat_history_str += f"챗봇: {chat.assistant}\n"
    
    result = rag_manager.chat(
        user_id=request.user_id,
        message=request.message,
        chat_history=chat_history_str
    )
    
    return ChatResponse(
        response=result.get("response", ""),
        used_knowledge=result.get("used_knowledge", [])
    )

@router.post("/chat/verified", response_model=VerifiedChatResponse)
async def process_verified_chat(request: ChatRequest):
    """검증 기능이 포함된 채팅 메시지 처리"""
    
    from app.main import rag_manager

    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        result = rag_manager.chat_with_verification(request.user_id, request.message)
        
        return VerifiedChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류: {str(e)}")

@router.post("/verification-report")
async def generate_verification_report(request: VerificationReportRequest):
    """특정 메시지에 대한 상세 검증 보고서 생성"""
    
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        report = rag_manager.generate_verification_report(request.user_id, request.message)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 보고서 생성 중 오류: {str(e)}")

@router.get("/plan-database/info")
async def get_plan_database_info():
    """요금제 데이터베이스 정보 조회"""
    
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        return rag_manager.get_plan_database_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요금제 정보 조회 중 오류: {str(e)}")

@router.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """특정 사용자의 대화 통계 조회"""
    
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        stats = rag_manager.get_user_statistics(user_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")