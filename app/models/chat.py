# app/models/chat.py

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    reply: str
    model: str = "gpt-3.5-turbo"