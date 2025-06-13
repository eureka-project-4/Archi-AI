import asyncio
from app.services.ai_classifier import ai_classifier
from app.services.content_filter import content_filter
from app.models.message import AuthMetadata
import json
from app.services.bad_word_filter import content_filter2
async def test_components():
    # 1. AI Classifier 테스트
    print("=== AI Classifier 테스트 ===")
    metadata = AuthMetadata(ageCode="004", tagCode="NORMAL")
    
    test_cases = [
        "5만원 이하 요금제 추천해줘",
        "요즘 데이터를 많이 써서 무제한으로 바꾸고 싶어",
        "안녕하세요",
        "영화관이랑 카페 관련 멤버십 추천해줘"
    ]
    
    for content in test_cases:
        msg_type = await ai_classifier.classify_message_type(content, metadata)
        print(f"입력: '{content}' → 타입: {msg_type}")
    
    # 2. Content Filter 테스트
    print("\n=== Content Filter 테스트 ===")
    test_contents = [
        "정상적인 메시지입니다",
        "시발 이게 뭐야",
        "니 @ㅐ미 창!녀",
        "깜둥이 노예놈아",
        "엉덩이 만지고 싶다",
        "짱개 미개해~",
        "좆같은새끼야",
        "fuck you"
    ]
    
    for content in test_contents:
        # Raw response를 위해 client 직접 호출
        try:
            response=content_filter2.detect_profanity(content)
            
            print(f"\n내용: '{content}'")
            print(response)
            
            
            
        except Exception as e:
            print(f"오류: {e}")

if __name__ == "__main__":
    asyncio.run(test_components())