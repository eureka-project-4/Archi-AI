from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    CHAT = "chat"       
    SUGGESTION = "suggestion"  

@dataclass
class HallucinationCheck:
    plan_exists: bool
    confidence_score: float
    matched_plan: Optional[str] = None
    discrepancies: List[str] = None
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.discrepancies is None:
            self.discrepancies = []
        if self.evidence is None:
            self.evidence = []

@dataclass
class VerificationResult:
    response: str
    mentioned_plans: List[str]
    verification_results: dict
    overall_confidence: float
    verification_status: str
    has_verification_issues: bool

@dataclass
class ChatEntry:
    timestamp: str
    human: str
    ai: str
    message_type: MessageType = MessageType.CHAT  # 기본값은 일반 대화
    mentioned_plans: List[str] = None  # 언급된 요금제들
    confidence_score: Optional[float] = None  # 추천 신뢰도
    
    def __post_init__(self):
        if self.mentioned_plans is None:
            self.mentioned_plans = []
    
    @classmethod
    def create_now(cls, human_input: str, ai_response: str, message_type: MessageType = MessageType.CHAT, 
                   mentioned_plans: List[str] = None, confidence_score: Optional[float] = None):
        return cls(
            timestamp=datetime.now().isoformat(),
            human=human_input,
            ai=ai_response,
            message_type=message_type,
            mentioned_plans=mentioned_plans or [],
            confidence_score=confidence_score
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