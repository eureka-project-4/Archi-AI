
from fastapi import APIRouter, UploadFile, HTTPException
from app.core.img_analyze.image_analyze import ImageAnalyzer
from app.models.preference_tags import PREFERENCE_TAGS, CONTENT_TAGS, LIFE_TAGS, SELF_TAGS, HOBBY_TAGS, CONSUME_TAGS, FEW_SHOT
import base64

router = APIRouter()
analyzer = ImageAnalyzer()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile):
    """
    사용자가 업로드한 배터리 스크린샷을 분석하여
    - 1문장 성향 요약
    - 통신 성향 + 선호 태그 추출
    """
    try:
        # 1) 파일 읽고 Base64 인코딩
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # 2) ImageAnalyzer 호출
        result = analyzer.analyze_image_and_tags(base64_image)

        ALL_TAGS = PREFERENCE_TAGS
        tags = result.get("tags", [])

        # 유효하지 않은 태그만 추출
        invalid_tags = [tag for tag in tags if tag not in ALL_TAGS]

        if invalid_tags:
            print(f"유효하지 않은 태그 발견: {invalid_tags}")

        result["tags"] = [tag for tag in tags if tag in ALL_TAGS]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

