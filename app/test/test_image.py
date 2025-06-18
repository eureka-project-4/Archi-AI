# 라우터 호출, 이미지는 로컬에서 업로드(file path = "C:W") 

import requests
from app.services.img_analyze.image_router import analyze_image  


file_path = r"/Users/lbk6661/Desktop/sample1.jpg"

async def test_direct_call():
    # 파일 열기 (binary)
    with open(file_path, "rb") as f:
        # Starlette UploadFile 은 file-like 객체 필요
        upload_file = StarletteUploadFile(filename="sample1.jpg", file=f)

        # 라우터 함수 직접 호출
        result = await analyze_image(upload_file)
        print(result)

asyncio.run(test_direct_call())
