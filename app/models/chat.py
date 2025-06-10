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