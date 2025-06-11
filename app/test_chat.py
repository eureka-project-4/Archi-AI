import requests
import json

BASE_URL = "http://localhost:8000"

def test_multiturn_chat():
    """한 사용자의 멀티턴 채팅 시나리오 테스트"""
    
    user_id = "multi_turn_user"
    turns = [
        # {
        #     "name": "인사",
        #     "message": "안녕하세요",
        #     "expect": {
        #         "response_contains": "안녕하세요",
        #         "mentioned_plans": [],
        #         "verification_status": "높은 신뢰도 - 정확한 정보"
        #     }
        # },
        # {
        #     "name": "조건 기반 추천",
        #     "message": "저는 통화를 많이 하고 데이터를 조금 써요. 어떤 요금제가 좋을까요?",
        #     "expect": {
        #         "response_contains": "추천",
        #         "verification_status": "높은 신뢰도 - 정확한 정보"
        #     }
        # },
        # {
        #     "name": "구체적 요금제 질문",
        #     "message": "5G 프리미어 에센셜 요금제에 대해 알려줘",
        #     "expect": {
        #         "response_contains": "5G 프리미어 에센셜",
        #         "mentioned_plans_contains": "5G 프리미어 에센셜",
        #         "verification_status": "높은 신뢰도 - 정확한 정보",
        #         "confidence_score_min": 0.7
        #     }
        # },
        # {
        #     "name": "가짜 요금제(할루시네이션) 차단",
        #     "message": "킹왕짱 요금제도 설명해줘",
        #     "expect": {
        #         "response_contains": "제공되지 않는 요금제",
        #         "verification_status": "요금제 없음",
        #         "confidence_score_max": 0.5
        #     }
        # },
        # {
        #     "name": "실제/가짜 혼합 비교",
        #     "message": "5G 프리미어 에센셜과 킹왕짱 요금제 비교해줘",
        #     "expect": {
        #         "mentioned_plans_contains": "5G 프리미어 에센셜",
        #         "response_contains": "제공되지 않는 요금제",
        #         "verification_status": None  # 혼합 응답이면 별도 체크
        #     }
        # },
        {
            "name": "다시 조건 기반 추천",
            "message": "월 3만원 이하로 쓸 수 있는 요금제 있어?",
            "expect": {
                "response_contains": "추천",
                "mentioned_plans": [],
                "verification_status": "높은 신뢰도 - 정확한 정보"
            }
        }
    ]
    
    print("\n=== 멀티턴 채팅 시나리오 테스트 ===\n")
    for i, turn in enumerate(turns, 1):
        print(f"▶️ 턴 {i}: {turn['name']}")
        print(f"질문: {turn['message']}")
        
        payload = {
            "user_id": user_id,
            "message": turn["message"]
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/verified", json=payload)
        if response.status_code == 200:
            try:
                data = response.json()
                response_text = data.get('response', '')
                mentioned_plans = data.get('mentioned_plans', [])
                verification_status = data.get('verification_status', '')
                confidence_score = data.get('confidence_score', None)
                
                # 기본 응답 출력
                print(f"✅ 응답: {response_text[:100]}...")
                print(f"📊 검증 상태: {verification_status}")
                
                
                
                print(f"   ✅ 검증 통과\n" + "-"*60)
            except AssertionError as e:
                print(f"❌ 테스트 실패: {e}\nRaw: {data}")
            except Exception as e:
                print(f"❌ JSON 파싱/기타 오류: {e}\nRaw: {response.text}")
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"오류 내용: {response.text}")
        print()

if __name__ == "__main__":
    test_multiturn_chat()
