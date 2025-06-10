# app/models/chat.py

from typing import List, Optional
from pydantic import BaseModel

class ChatHistoryItem(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    message: str
    user_id: str
    chat_history: Optional[List[ChatHistoryItem]] = []

class ChatResponse(BaseModel):
    response: str
    used_knowledge: List[str]
    
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# 기존 모델
class ChatRequest(BaseModel):
    user_id: str
    message: str
    chat_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    used_knowledge: List[str] = []

# 확장 모델
class VerifiedChatRequest(ChatRequest):
    enable_verification: bool = True

class VerifiedChatResponse(ChatResponse):
    verification_status: Optional[str] = None
    mentioned_plans: List[str] = []
    confidence_score: float = 1.0
    message_type: str = "chat"
    verification_results: Optional[Dict[str, Any]] = None

class LoginRequest(BaseModel):
    username: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    is_new_user: bool
    username: str
    conversation_count: int

class VerificationReportRequest(BaseModel):
    user_id: str
    message: str

class UserStatsResponse(BaseModel):
    username: str
    total_conversations: int
    has_summary: bool
    summary_length: int
    first_visit: str
    last_login: str
    current_session_messages: int

class PlanDatabaseInfo(BaseModel):
    total_plans: int
    total_entries: int
    sample_plans: List[str]

class MessageStats(BaseModel):
    username: str
    total_messages: int
    chat_messages: int
    suggestion_messages: int
    average_confidence: float
    suggestion_ratio: float