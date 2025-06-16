<<<<<<< HEAD
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class MessageType(Enum):
=======
from pydantic import BaseModel, Field
from typing import Optional, Union
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
>>>>>>> bed46aa58795337f2d2f6cb617ed016fdd58dd0c
    USER_MESSAGE = "USER_MESSAGE"
    SUGGESTION = "SUGGESTION"
    KEYWORD_RECOMMENDATION = "KEYWORD_RECOMMENDATION"
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE"
    PROACTIVE_SUGGESTION = "PROACTIVE_SUGGESTION"
    GENERAL_RESPONSE = "GENERAL_RESPONSE"
    FILTERED_MESSAGE = "FILTERED_MESSAGE"
<<<<<<< HEAD
    BLOCKED_MESSAGE = "BLOCKED_MESSAGE"

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
=======

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
    user_id: Union[str, int] = Field(alias="userId")  # str과 int 모두 허용
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
>>>>>>> bed46aa58795337f2d2f6cb617ed016fdd58dd0c
