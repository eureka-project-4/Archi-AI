import requests
import json
from typing import Dict, List, Any

BASE_URL = "http://localhost:8000"

def test_hallucination_prevention():
    """할루시네이션 방지 시스템 테스트"""
    
    test_cases = [
        {
            "name": "킹왕짱 요금제 문의",
            "user_id": "test_user",
            "message": "킹왕짱 요금제에 대해 설명해줘",
            "should_block": True,
            "description": "존재하지 않는 요금제는 차단되어야 함"
        },
        {
            "name": "슈퍼울트라 요금제 문의", 
            "user_id": "test_user",
            "message": "슈퍼울트라맥시멈 요금제 추천해주세요",
            "should_block": True,
            "description": "가짜 요금제명은 차단되어야 함"
        },
        {
            "name": "메가킹왕짱 플랜 문의",
            "user_id": "test_user",
            "message": "메가킹왕짱 플랜은 얼마인가요?",
            "should_block": True,
            "description": "창의적인 가짜 요금제명도 차단되어야 함"
        },
        {
            "name": "실제 요금제 문의 (5G)",
            "user_id": "test_user", 
            "message": "5G 프리미어 에센셜 요금제는 어떤가요?",
            "should_block": False,
            "description": "실제 존재하는 요금제는 정상 처리되어야 함"
        },
        {
            "name": "일반 인사",
            "user_id": "test_user",
            "message": "안녕하세요",
            "should_block": False,
            "description": "일반적인 대화는 정상 처리되어야 함"
        },
        {
            "name": "요금제 추천 요청",
            "user_id": "test_user",
            "message": "월 5만원대 요금제 추천해주세요",
            "should_block": False,
            "description": "일반적인 추천 요청은 정상 처리되어야 함"
        }
    ]
    
    print("할루시네이션 방지 시스템 테스트 시작")
    print("=" * 60)
    
    passed_tests = 0
    failed_tests = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n시나리오 {i}: {test_case['name']}")
        print(f"사용자: {test_case['user_id']}")
        print(f"질문: {test_case['message']}")
        
        payload = {
            "user_id": test_case["user_id"],
            "message": test_case["message"]
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                ai_response = data.get('response', '')
                confidence = data.get('confidence_score', 1.0)
                verification_status = data.get('verification_status', '')
                mentioned_plans = data.get('mentioned_plans', [])
                message_type = data.get('message_type', '')
                
                print(f"응답: {ai_response[:150]}{'...' if len(ai_response) > 150 else ''}")
                print(f"검증 상태: {verification_status}")
                print(f"언급된 요금제: {mentioned_plans}")
                print(f"신뢰도: {confidence}")
                print(f"메시지 타입: {message_type}")
                
                test_passed = evaluate_test_result(test_case, data)
                
                if test_passed:
                    passed_tests += 1
                    print("테스트 통과")
                else:
                    failed_tests.append({
                        "scenario": test_case['name'],
                        "reason": get_failure_reason(test_case, data)
                    })
                    print("테스트 실패")
                    
            else:
                print(f"HTTP 오류: {response.status_code}")
                print(f"오류 내용: {response.text}")
                failed_tests.append({
                    "scenario": test_case['name'],
                    "reason": f"HTTP {response.status_code} 오류"
                })
                
        except Exception as e:
            print(f"예외 발생: {e}")
            failed_tests.append({
                "scenario": test_case['name'],
                "reason": f"예외 발생: {str(e)}"
            })
        
        print("-" * 50)
    
    total_tests = len(test_cases)
    print(f"\n테스트 결과: {passed_tests}/{total_tests} 통과")
    
    if failed_tests:
        print("\n실패한 테스트:")
        for failure in failed_tests:
            print(f"- {failure['scenario']}: {failure['reason']}")
    
    return passed_tests == total_tests

def evaluate_test_result(test_case: Dict, response_data: Dict) -> bool:
    """테스트 결과 평가"""
    ai_response = response_data.get('response', '')
    confidence = response_data.get('confidence_score', 1.0)
    
    if test_case['should_block']:
        if "현재 제공되지 않는 요금제" in ai_response or "찾을 수 없습니다" in ai_response:
            return True
        
        fake_plan_names = ["킹왕짱", "슈퍼울트라", "메가킹왕짱"]
        for fake_name in fake_plan_names:
            if fake_name in test_case['message'] and fake_name in ai_response:
                return False
        
        if confidence < 0.5:
            return True
            
        return False
    else:
        if "현재 제공되지 않는 요금제" in ai_response:
            return False
        return True

def get_failure_reason(test_case: Dict, response_data: Dict) -> str:
    """테스트 실패 이유 분석"""
    ai_response = response_data.get('response', '')
    confidence = response_data.get('confidence_score', 1.0)
    
    if test_case['should_block']:
        if confidence >= 0.5:
            return "가짜 요금제가 차단되지 않음 (높은 신뢰도)"
        fake_plan_names = ["킹왕짱", "슈퍼울트라", "메가킹왕짱"]
        for fake_name in fake_plan_names:
            if fake_name in test_case['message'] and fake_name in ai_response:
                return f"응답에 가짜 요금제명 '{fake_name}' 포함"
        return "가짜 요금제가 적절히 차단되지 않음"
    else:
        if "현재 제공되지 않는 요금제" in ai_response:
            return "실제 요금제 또는 일반 질문이 잘못 차단됨"
        return "알 수 없는 오류"

def test_confidence_scores():
    """신뢰도 점수 테스트"""
    print("\n신뢰도 점수 테스트")
    print("=" * 60)
    
    test_queries = [
        ("킹왕짱 요금제", 0.0, 0.3),
        ("5G 프리미어 에센셜", 0.8, 1.0),
        ("프리미어", 0.6, 0.9),
        ("슈퍼맥시멈울트라", 0.0, 0.3),
        ("데이터 무제한 요금제", 0.5, 1.0)
    ]
    
    for query, min_expected, max_expected in test_queries:
        response = requests.post(f"{BASE_URL}/api/chat", json={
            "user_id": "confidence_test",
            "message": f"{query} 요금제에 대해 알려주세요"
        })
        
        if response.status_code == 200:
            data = response.json()
            confidence = data.get('confidence_score', 1.0)
            
            print(f"쿼리: '{query}'")
            print(f"신뢰도: {confidence:.2f} (예상 범위: {min_expected:.2f} ~ {max_expected:.2f})")
            
            if min_expected <= confidence <= max_expected:
                print("통과")
            else:
                print("실패 - 예상 범위를 벗어남")
        else:
            print(f"쿼리 '{query}' 실패: HTTP {response.status_code}")
        
        print("-" * 30)

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n엣지 케이스 테스트")
    print("=" * 60)
    
    edge_cases = [
        {
            "name": "오타가 있는 실제 요금제",
            "message": "5G 프리미어 에센셜요금제 알려주세요",
            "expected": "정상 처리"
        },
        {
            "name": "요금제명 없이 질문",
            "message": "제일 비싼 요금제는 뭔가요?",
            "expected": "정상 처리"
        },
        {
            "name": "여러 요금제 동시 언급",
            "message": "킹왕짱 요금제랑 5G 프리미어 에센셜 비교해주세요",
            "expected": "킹왕짱만 차단"
        },
        {
            "name": "영어 요금제명",
            "message": "Super Ultra Maximum plan 추천해주세요",
            "expected": "차단"
        }
    ]
    
    for case in edge_cases:
        print(f"\n테스트: {case['name']}")
        print(f"입력: {case['message']}")
        print(f"예상: {case['expected']}")
        
        response = requests.post(f"{BASE_URL}/api/chat", json={
            "user_id": "edge_test",
            "message": case['message']
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"응답: {data.get('response', '')[:100]}...")
            print(f"신뢰도: {data.get('confidence_score', 1.0):.2f}")
        else:
            print(f"오류: HTTP {response.status_code}")

def test_performance():
    """성능 테스트"""
    import time
    
    print("\n성능 테스트")
    print("=" * 60)
    
    queries = [
        "킹왕짱 요금제 추천해주세요",
        "5G 프리미어 에센셜 요금제 정보",
        "월 3만원대 요금제 있나요?",
        "안녕하세요"
    ]
    
    total_time = 0
    successful_requests = 0
    
    for i, query in enumerate(queries * 3):
        start_time = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}/api/chat", json={
                "user_id": "performance_test",
                "message": query
            })
            
            if response.status_code == 200:
                successful_requests += 1
                
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if i % len(queries) == 0:
                print(f"요청 {i+1}: {elapsed:.2f}초")
                
        except Exception as e:
            print(f"요청 {i+1} 실패: {e}")
    
    avg_time = total_time / (len(queries) * 3)
    print(f"\n평균 응답 시간: {avg_time:.2f}초")
    print(f"성공률: {successful_requests}/{len(queries)*3}")

if __name__ == "__main__":
    print("할루시네이션 방지 시스템 종합 테스트")
    print("=" * 60)
    
    success = test_hallucination_prevention()
    
    test_confidence_scores()
    
    test_edge_cases()
    
    test_performance()
    
    print("\n" + "=" * 60)
    if success:
        print("모든 기본 테스트가 통과했습니다!")
    else:
        print("일부 테스트가 실패했습니다. 시스템 점검이 필요합니다.")