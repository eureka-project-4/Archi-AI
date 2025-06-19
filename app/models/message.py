from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class MessageType(Enum):
    USER_MESSAGE = "USER_MESSAGE" #사용자 메시지
    SUGGESTION = "SUGGESTION" #평범한 추천
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE" #성향 업데이트
    PROACTIVE_SUGGESTION = "PROACTIVE_SUGGESTION" # 선제안 스케줄링
    GENERAL_RESPONSE = "GENERAL_RESPONSE" #평범한 대화
    FILTERED_MESSAGE = "FILTERED_MESSAGE" #욕설 필터링
    BLOCKED_MESSAGE = "BLOCKED_MESSAGE" #없는 요금제
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS" #이미지 분석

class AuthMetadata(BaseModel):
    age_code: str = Field(alias="ageCode")
    tag_code: str = Field(alias="tagCode")
    
    class Config:
        populate_by_name = True 

class MessagePayload(BaseModel):
    user_id: str = Field(alias="userId")
    content: str
    message_id: str = Field(alias="messageId")
    type: Optional[MessageType] = None
    
    class Config:
        populate_by_name = True

class AiPromptMessage(BaseModel):
    payload: MessagePayload
    metadata: AuthMetadata