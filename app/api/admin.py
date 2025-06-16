from fastapi import APIRouter, HTTPException
from app.models.admin import UpdateVectorResponse
from app.database import export_tables_to_csv
import os

from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class UpdateVectorResponse(BaseModel):
    success: bool
    message: str
    chunks_created: Optional[int] = None

@router.post("/update-vectors", response_model=UpdateVectorResponse)
async def update_vectors():
   """벡터스토어 업데이트"""
   from app.main import rag_manager
   
   if rag_manager is None:
       raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
   
   export_tables_to_csv()
   csv_dir = "./app/data/pricing"
   file_paths = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
   result = rag_manager.update_vectorstore(file_paths)
   
   return UpdateVectorResponse(**result)

@router.get("/plan-database/info")
async def get_plan_database_info():
   """요금제 데이터베이스 정보 조회"""
   from app.main import rag_manager
   
   if rag_manager is None:
       raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
   
   return rag_manager.get_plan_database_info()

@router.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
   """사용자 대화 통계 조회"""
   from app.main import rag_manager
   
   if rag_manager is None:
       raise HTTPException(status_code=500, detail="AI 시스템이 초기화되지 않았습니다.")
   
   stats = rag_manager.get_user_statistics(user_id)
   if "error" in stats:
       raise HTTPException(status_code=404, detail=stats["error"])
   return stats