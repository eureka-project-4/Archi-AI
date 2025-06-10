import requests
import json

BASE_URL = "http://localhost:8000"

def test_scenarios():
    """실제 사용 시나리오 테스트"""
    
    scenarios = [
        {
            "name": "일반 인사",
            "user_id": "kim_user", 
            "message": "안녕하세요"
        },
        {
            "name": "요금제 문의", 
            "user_id": "kim_user",
            "message": "저는 데이터를 많이 써요. 어떤 요금제가 좋을까요?"
        },
        {
            "name": "구체적 요금제 질문",
            "user_id": "kim_user", 
            "message": "5G 프리미어 에센셜 요금제에 대해 알려주세요"
        },
        {
            "name": "예산 기반 추천",
            "user_id": "park_user",
            "message": "월 5만원 이하로 쓸 수 있는 요금제 추천해주세요"
        },
        {
            "name": "비교 요청",
            "user_id": "park_user",
            "message": "5G 스탠다드와 5G 프리미어 에센셜 차이점이 뭔가요?"
        },
        {
            "name": "가족 요금제 문의",
            "user_id": "lee_family",
            "message": "가족 4명이 쓸 수 있는 요금제 있나요?"
        }
    ]
    
    print("=== 실제 사용 시나리오 테스트 ===\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🎯 시나리오 {i}: {scenario['name']}")
        print(f"사용자: {scenario['user_id']}")
        print(f"질문: {scenario['message']}")
        
        # 검증 기능 포함 채팅 사용 (이게 정상 작동함)
        payload = {
            "user_id": scenario["user_id"],
            "message": scenario["message"]
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/verified", json=payload)
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ 응답: {data['response'][:100]}...")
                print(f"📊 검증 상태: {data.get('verification_status', 'N/A')}")
                print(f"📋 언급된 요금제: {data.get('mentioned_plans', [])}")
                print(f"🎯 신뢰도: {data.get('confidence_score', 'N/A')}")
                print(f"💬 메시지 타입: {data.get('message_type', 'N/A')}")
            except Exception as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"오류 내용: {response.text}")
        
        print("-" * 80)
        print()

def test_plan_mentions():
    """요금제 언급 감지 테스트"""
    
    test_messages = [
        "5G 프리미어 에센셜이 좋다고 들었어요",
        "유쓰 5G 스탠다드 에센셜 가격이 궁금해요", 
        "5G 키즈 45 요금제는 어떤가요?",
        "존재하지않는요금제는 어떤가요?",
        "킹왕짱 요금제는 어떤가요?"
    ]
    
    print("=== 요금제 언급 감지 테스트 ===\n")
    
    for i, message in enumerate(test_messages, 1):
        print(f"🔍 테스트 {i}: {message}")
        
        payload = {
            "user_id": "test_mentions",
            "message": message
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/verified", json=payload)
        
        if response.status_code == 200:
            try:
                data = response.json()
                mentioned = data.get('mentioned_plans', [])
                confidence = data.get('confidence_score', 1.0)
                
                if mentioned:
                    print(f"✅ 감지된 요금제: {mentioned}")
                    print(f"📊 신뢰도: {confidence:.1%}")
                else:
                    print(f"ℹ️ 요금제 언급 없음")
                
                print(f"💬 응답: {data['response'][:80]}...")
                
            except Exception as e:
                print(f"❌ 오류: {e}")
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
        
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_scenarios()
    test_plan_mentions()