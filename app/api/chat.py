from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """채팅 메시지 처리"""
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
    
    # 채팅 히스토리 포맷팅
    chat_history_str = ""
    if request.chat_history:
        for chat in request.chat_history:
            chat_history_str += f"사용자: {chat.get('user', '')}\n"
            chat_history_str += f"챗봇: {chat.get('assistant', '')}\n"
    
    # AI 처리
    result = rag_manager.chat(
        user_id=request.user_id,
        message=request.message,
        chat_history=chat_history_str
    )
    
    return ChatResponse(
        response=result["response"],
        used_knowledge=result.get("used_knowledge", [])
    )