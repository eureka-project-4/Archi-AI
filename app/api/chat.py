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
    verification_results: dict = {}
    verification_method: str = ""

class VerificationReportRequest(BaseModel):
    user_id: str
    message: str

@router.post("/chat", response_model=VerifiedChatResponse)
async def process_chat(request: ChatRequest):
    """기본 채팅 메시지 처리 - 모든 채팅에 검증 적용"""
    
    from app.main import rag_manager

    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        result = rag_manager.chat_with_verification(request.user_id, request.message)
        
        return VerifiedChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류: {str(e)}")

@router.post("/chat/verified", response_model=VerifiedChatResponse)
async def process_verified_chat(request: ChatRequest):
    """검증 기능이 포함된 채팅 메시지 처리 (명시적 엔드포인트)"""
    
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

@router.get("/debug/plan-verification/{plan_name}")
async def debug_plan_verification(plan_name: str):
    """요금제 검증 디버깅 엔드포인트"""
    
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    try:
        if not rag_manager.csv_verifier:
            return {"error": "CSV 검증 시스템이 없습니다"}
        
        verification = rag_manager.csv_verifier.verify_plan_exists(plan_name)
        
        return {
            "plan_name": plan_name,
            "verification_result": verification,
            "exists": verification['exists'],
            "confidence": verification['confidence'],
            "match_type": verification['match_type'],
            "matched_plan": verification['matched_plan']['name'] if verification['matched_plan'] else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 디버깅 중 오류: {str(e)}")