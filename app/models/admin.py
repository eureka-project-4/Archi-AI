# app/models/admin.py

from pydantic import BaseModel
from typing import List, Optional

class UpdateVectorRequest(BaseModel):
    file_paths: List[str]

class UpdateVectorResponse(BaseModel):
    success: bool
    message: str
    chunks_created: Optional[int] = None