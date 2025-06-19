from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class MessageType(Enum):
    USER_MESSAGE = "USER_MESSAGE"
    SUGGESTION = "SUGGESTION"
    KEYWORD_RECOMMENDATION = "KEYWORD_RECOMMENDATION"
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE"
    PROACTIVE_SUGGESTION = "PROACTIVE_SUGGESTION"
    GENERAL_RESPONSE = "GENERAL_RESPONSE"
    FILTERED_MESSAGE = "FILTERED_MESSAGE"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"

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
