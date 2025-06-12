from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    SUGGESTION = "SUGGESTION"
    KEYWORD_RECOMMENDATION = "KEYWORD_RECOMMENDATION"
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE"
    PROACTIVE_SUGGESTION = "PROACTIVE_SUGGESTION"
    GENERAL_RESPONSE = "GENERAL_RESPONSE"
    FILTERED_MESSAGE = "FILTERED_MESSAGE"

class SenderType(str, Enum):
    USER = "USER"
    BOT = "BOT"


class AuthMetadata(BaseModel):
    tag_code: int = Field(alias="tagCode")
    age_code: int = Field(alias="ageCode")

    class Config:
        allow_population_by_field_name = True

class ChatMessage(BaseModel):
    message_id: str = Field(alias="messageId")
    user_id: str = Field(alias="userId")
    content: str
    type: MessageType
    sender: SenderType
    timestamp: datetime

    class Config:
        allow_population_by_field_name = True

class AiPromptMessage(BaseModel):
    metadata: AuthMetadata
    payload: ChatMessage

    class Config:
        allow_population_by_field_name = True