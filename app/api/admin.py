from fastapi import APIRouter, HTTPException
from app.models.admin import UpdateVectorResponse
from app.database import export_tables_to_csv
import os

router = APIRouter()

@router.post("/update-vectors", response_model=UpdateVectorResponse)
async def update_vectors():
    """📦 DB → CSV → 벡터스토어 업데이트"""
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")

    # 1️⃣ DB 테이블을 CSV로 export
    export_tables_to_csv()

    # 2️⃣ export된 파일 경로 수집
    csv_dir = "./app/data/pricing"
    file_paths = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]

    # 3️⃣ 벡터스토어 업데이트
    result = rag_manager.update_vectorstore(file_paths)

    return UpdateVectorResponse(
        success=result["success"],
        message=result["message"],
        chunks_created=result.get("chunks_created")
    )