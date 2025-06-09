# Archi-AI
pip install -r requirements.txt

archi_chat2.py는
최근 5개만 불러와서 사용
archi_chat3.py는
대화 10개 이상 시 요약기능 사용. -> 최근 대화와 합쳐서 사용.
 
data_models.py
"""
데이터 모델 정의
할루시네이션 검증 및 챗봇 관련 데이터 클래스들
"""
HallucinationCheck:
    """할루시네이션 검증 결과 데이터 클래스"""
VerificationResult:
    """검증 결과를 담는 데이터 클래스"""
ChatEntry:
    """대화 기록 엔트리"""
UserMemory:
    """사용자 메모리 데이터"""
