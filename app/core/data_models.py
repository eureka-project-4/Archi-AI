from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    CHAT = "chat"       
    SUGGESTION = "suggestion"  

@dataclass
class ChatEntry:
    timestamp: str
    human: str
    ai: str
    message_type: MessageType = MessageType.CHAT
    mentioned_plans: List[str] = None
    
    def __post_init__(self):
        if self.mentioned_plans is None:
            self.mentioned_plans = []
    
    @classmethod
    def create_now(cls, human_input: str, ai_response: str, message_type: MessageType = MessageType.CHAT, 
                   mentioned_plans: List[str] = None):
        return cls(
            timestamp=datetime.now().isoformat(),
            human=human_input,
            ai=ai_response,
            message_type=message_type,
            mentioned_plans=mentioned_plans or []
        )

@dataclass
class UserMemory:
    username: str
    chat_history: List[dict]
    conversation_summary: str
    last_login: str
    total_conversations: int
    created_at: str
    
    @classmethod
    def create_new(cls, username: str):
        now = datetime.now().isoformat()
        return cls(
            username=username,
            chat_history=[],
            conversation_summary="",
            last_login=now,
            total_conversations=0,
            created_at=now
        )

class VerificationResult:
    """CSV 검증 결과 (단순 Dict 대신 타입 힌트용)"""
    exists: bool
    confidence: float
    match_type: str 
    matched_plan: Optional[dict]
    original_input: str