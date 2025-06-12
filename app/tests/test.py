import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_chat_basic(user_id="test_user", message="안녕하세요"):
    """기본 채팅 테스트 (/api/chat)"""
    payload = {
        "user_id": user_id,
        "message": message
    }
    response = requests.post(f"{BASE_URL}/api/chat", json=payload)
    print(f"\n=== 기본 채팅 테스트 ===")
    print(f"User: {user_id}")
    print(f"Input: {message}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"Response: {json_response.get('response', 'No response field')}")
            print(f"Used Knowledge: {json_response.get('used_knowledge', [])}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

def test_chat_verified(user_id="test_user", message="월 3만원 이하 요금제 추천해주세요"):
    """검증 기능 포함 채팅 테스트 (/api/chat)"""
    payload = {
        "user_id": user_id,
        "message": message
    }
    response = requests.post(f"{BASE_URL}/api/chat", json=payload)
    print(f"\n=== 검증 채팅 테스트 ===")
    print(f"User: {user_id}")
    print(f"Input: {message}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"Response: {json_response.get('response', 'No response field')}")
            print(f"Verification Status: {json_response.get('verification_status', 'N/A')}")
            print(f"Mentioned Plans: {json_response.get('mentioned_plans', [])}")
            print(f"Confidence Score: {json_response.get('confidence_score', 'N/A')}")
            print(f"Message Type: {json_response.get('message_type', 'N/A')}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

def test_plan_database_info():
    """요금제 데이터베이스 정보 조회"""
    response = requests.get(f"{BASE_URL}/api/plan-database/info")
    print(f"\n=== 요금제 DB 정보 ===")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"DB Info: {json_response}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

def test_user_stats(user_id="test_user"):
    """사용자 통계 조회"""
    response = requests.get(f"{BASE_URL}/api/users/{user_id}/stats")
    print(f"\n=== 사용자 통계 ({user_id}) ===")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"User Stats: {json_response}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

def test_verification_report(user_id="test_user", message="5G 프리미어 에센셜 요금제 추천해드릴게요"):
    """검증 보고서 생성"""
    payload = {
        "user_id": user_id,
        "message": message
    }
    response = requests.post(f"{BASE_URL}/api/verification-report", json=payload)
    print(f"\n=== 검증 보고서 테스트 ===")
    print(f"User: {user_id}")
    print(f"Input: {message}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"Verification Report: {json_response}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

def test_admin_update_vectors():
    """관리자 - 벡터 업데이트"""
    response = requests.post(f"{BASE_URL}/admin/update-vectors")
    print(f"\n=== 관리자 - 벡터 업데이트 ===")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(f"Update Result: {json_response}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("=== 올바른 API 경로로 테스트 시작 ===")
    
    # 기본 테스트
    test_health()
    
    # 요금제 DB 정보 확인
    test_plan_database_info()
    
    # 기본 채팅 테스트 (500 에러 디버깅용)
    test_chat_basic("user1", "안녕하세요")
    
    # 검증 기능 포함 채팅 테스트
    test_chat_verified("user1", "월 3만원 이하 요금제 추천해주세요")
    
    # 사용자 통계
    test_user_stats("user1")
    
    # 검증 보고서
    test_verification_report("user1", "5G 프리미어 에센셜 요금제가 좋습니다")
    
    # 관리자 기능
    test_admin_update_vectors()
    
    print("\n=== 테스트 완료 ===")