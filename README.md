# AI 요금제 추천 챗봇 시스템

통신사 요금제 추천을 위한 AI 챗봇 시스템입니다. RAG(Retrieval-Augmented Generation)와 CSV 기반 검증 시스템을 통해 정확하고 신뢰할 수 있는 요금제 추천 서비스를 제공합니다.

## 주요 기능

### 🤖 AI 챗봇
- **자연어 대화**: 사용자와 자연스러운 한국어 대화
- **개인화 추천**: 사용자의 통화량, 데이터 사용량, 예산 기반 맞춤 추천
- **대화 기록 관리**: 사용자별 대화 이력 저장 및 연속 대화 지원

### 🔍 검증 시스템
- **CSV 직접 검증**: 96개 요금제 데이터베이스 기반 실시간 검증
- **할루시네이션 방지**: AI 응답의 요금제 정보 정확성 자동 검증
- **신뢰도 측정**: 응답별 신뢰도 스코어 제공 (0.0 ~ 1.0)
- **상세 검증 보고서**: 언급된 요금제별 세부 검증 결과

### 📊 RAG 시스템
- **벡터 기반 검색**: 요금제 정보의 의미적 유사도 검색
- **실시간 업데이트**: 새로운 데이터 추가 시 벡터스토어 자동 업데이트
- **컨텍스트 인식**: 대화 맥락을 고려한 정확한 응답 생성

### 👥 사용자 관리
- **세션 관리**: 사용자별 독립적인 대화 세션
- **메모리 시스템**: 장기 대화 요약 및 중요 정보 유지
- **통계 제공**: 사용자별 대화 통계 및 활동 분석

## 시스템 구조

```
├── app/
│   ├── main.py                     # FastAPI 서버 메인
│   ├── config.py                   # 설정 파일
│   ├── api/                        # API 라우터
│   │   ├── chat.py                 # 채팅 API
│   │   └── admin.py                # 관리자 API
│   └── core/                       # 핵심 시스템
│       ├── csv_verification_system.py  # CSV 검증 시스템
│       ├── rag_manager.py              # RAG 시스템 관리
│       ├── memory_manager.py           # 메모리 관리
│       ├── message_classifier.py       # 메시지 분류
│       └── data_models.py              # 데이터 모델
├── data/
│   └── pricing/                    # 요금제 CSV 데이터
└── user_memories/                  # 사용자 대화 기록
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install fastapi uvicorn pandas fuzzywuzzy langchain-openai langchain-community faiss-cpu
```

### 2. 환경 변수 설정

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 데이터 준비

`data/pricing/` 디렉토리에 다음 형식의 CSV 파일들을 배치:

- `plans.csv`: 요금제 정보 (plan_name, price, month_data, call_usage, message_usage, benefit 등)
- `coupons.csv`: 쿠폰 정보 (선택사항)
- `services.csv`: 서비스 정보 (선택사항)

### 4. 서버 실행

```bash
python -m uvicorn main:app --reload
```

또는

```bash
python main.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

## API 문서

### 주요 엔드포인트

#### 채팅 API
- `POST /api/chat/verified` - 검증 기능 포함 채팅 (권장)
- `GET /api/plan-database/info` - 요금제 데이터베이스 정보
- `GET /api/users/{user_id}/stats` - 사용자 통계
- `POST /api/verification-report` - 검증 보고서 생성

#### 관리자 API
- `POST /admin/update-vectors` - 벡터스토어 업데이트

#### 시스템 API
- `GET /health` - 서버 상태 확인
- `GET /docs` - Swagger UI 문서

### API 사용 예시

#### 기본 채팅

```bash
curl -X POST "http://localhost:8000/api/chat/verified" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "월 5만원 이하 요금제 추천해주세요"
  }'
```

#### 응답 예시

```json
{
  "response": "월 5만원 이하 요금제를 추천드리겠습니다...",
  "verification_status": "높은 신뢰도 - 정확한 정보",
  "mentioned_plans": ["5G 스탠다드", "유쓰 5G 스탠다드 에센셜"],
  "confidence_score": 0.95,
  "message_type": "suggestion",
  "used_knowledge": ["요금제 정보 관련 문서..."]
}
```

## 검증 시스템

### 신뢰도 등급
- **0.9 이상**: 높은 신뢰도 - 정확한 정보
- **0.7 이상**: 보통 신뢰도 - 대체로 정확
- **0.5 이상**: 낮은 신뢰도 - 일부 불일치 가능
- **0.5 미만**: 매우 낮은 신뢰도 - 정보 확인 필요

### 검증 매칭 타입
- `exact`: 정확한 일치
- `high_similarity`: 높은 유사도 (0.8 이상)
- `medium_similarity`: 중간 유사도 (0.6-0.8)
- `low_similarity`: 낮은 유사도 (0.4-0.6)
- `no_match`: 매칭 없음

## 테스트

### 테스트 스크립트 실행

```bash
python test_client.py
```

### 주요 테스트 시나리오
1. 기본 인사 및 응답
2. 예산 기반 요금제 추천
3. 특정 요금제 문의
4. 요금제 비교 요청
5. 할루시네이션 검증

## 현재 데이터베이스 상태

- **총 요금제 수**: 96개
- **검색 키 수**: 242개
- **주요 요금제**: 5G 프리미어 에센셜, 5G 키즈 45, 5G 프리미어 슈퍼, 유쓰 5G 스탠다드 에센셜, 5G 스탠다드

## 개발 정보

### 기술 스택
- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: OpenAI GPT-3.5-turbo, LangChain
- **Vector Store**: FAISS
- **Data Processing**: Pandas, FuzzyWuzzy
- **Memory**: JSON 기반 파일 저장

### 설정 가능한 옵션
- OpenAI 모델 선택
- 검색 결과 개수 (RETRIEVAL_K)
- 청크 크기 (CHUNK_SIZE)
- 메모리 관리 임계값
- 신뢰도 임계값

## 문제 해결

### 자주 발생하는 문제

1. **500 Internal Server Error**
   - OpenAI API 키 확인
   - CSV 파일 경로 확인
   - 메모리 디렉토리 권한 확인

2. **요금제 인식 안됨**
   - CSV 파일 형식 확인
   - 요금제명 정확성 확인
   - 벡터스토어 업데이트 실행

3. **메모리 저장 실패**
   - `user_memories/` 디렉토리 생성
   - 디렉토리 쓰기 권한 확인

### 로그 확인

```bash
# 서버 실행 시 콘솔에서 다음 메시지들 확인:
✅ CSV 검증 시스템 로드: data/pricing
✅ 기존 벡터스토어 로드됨
✅ RAG 체인 설정 완료
```

## 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

This project is licensed under the MIT License.

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.