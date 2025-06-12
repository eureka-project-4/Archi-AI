def test_verification_directly():
    """검증 시스템만 단독 테스트"""
    from app.core.csv_verification_system import CSVVerificationSystem
    
    csv_verifier = CSVVerificationSystem("./app/data/pricing/plans.csv")
    
    # 테스트 케이스
    test_cases = [
        "5G 프리미어 에센셜",
        "5G 스탠다드", 
        "존재하지않는요금제",
        "5G 프리미"  # 부분 매칭 테스트
    ]
    
    print("🔍 CSV 검증 시스템 직접 테스트:")
    print("=" * 40)
    
    for plan_name in test_cases:
        result = csv_verifier.verify_plan_exists(plan_name)
        print(f"\n요금제: {plan_name}")
        print(f"존재: {result['exists']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"매칭 타입: {result['match_type']}")
        if result['matched_plan']:
            print(f"매칭된 요금제: {result['matched_plan']['name']}")
            print(f"가격: {result['matched_plan']['price']:,}원")
if __name__ == "__main__":
    # 전체 시스템 테스트
 
    
    print("\n" + "="*60 + "\n")
    
    # 검증 시스템만 테스트
    test_verification_directly()