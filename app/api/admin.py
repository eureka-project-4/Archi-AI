from fastapi import APIRouter, HTTPException
from app.models.admin import UpdateVectorRequest, UpdateVectorResponse

router = APIRouter()

@router.post("/update-vectors", response_model=UpdateVectorResponse)
async def update_vectors(request: UpdateVectorRequest):
    """벡터스토어 업데이트"""
    from app.main import rag_manager
    
    if rag_manager is None:
        raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    result = rag_manager.update_vectorstore(request.file_paths)
    
    return UpdateVectorResponse(
        success=result["success"],
        message=result["message"],
        chunks_created=result.get("chunks_created")
    )