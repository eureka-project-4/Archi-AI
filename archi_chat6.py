import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
import difflib
from fuzzywuzzy import fuzz

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(override=True)

@dataclass
class HallucinationCheck:
    """할루시네이션 검증 결과 데이터 클래스"""
    plan_exists: bool
    confidence_score: float
    matched_plan: Optional[str] = None
    discrepancies: List[str] = None
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.discrepancies is None:
            self.discrepancies = []
        if self.evidence is None:
            self.evidence = []

@dataclass
class UserProfile:
    """사용자 프로필 데이터 클래스"""
    username: str
    max_budget: Optional[int] = None
    preferred_budget: Optional[int] = None
    monthly_data_usage: Optional[int] = None  # GB 단위
    call_minutes: Optional[int] = None  # 분 단위
    sms_count: Optional[int] = None
    user_type: Optional[str] = None  # 학생, 직장인, 가족, 시니어 등
    family_size: Optional[int] = None
    age: Optional[int] = None
    priority_features: List[str] = None  # 중요하게 생각하는 기능들
    current_plan: Optional[str] = None
    satisfaction_score: Optional[int] = None  # 1-10 만족도
    pain_points: List[str] = None  # 현재 요금제의 불만사항
    usage_patterns: Dict[str, Any] = None  # 사용 패턴 분석
    preferences: Dict[str, Any] = None  # 선호도
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.priority_features is None:
            self.priority_features = []
        if self.pain_points is None:
            self.pain_points = []
        if self.usage_patterns is None:
            self.usage_patterns = {}
        if self.preferences is None:
            self.preferences = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

@dataclass
class RecommendationHistory:
    """추천 기록 데이터 클래스"""
    timestamp: str
    recommended_plan: str
    reason: str
    user_response: Optional[str] = None
    accepted: Optional[bool] = None
    feedback: Optional[str] = None

class PersonalizedPricingAgent:
    def __init__(self, openai_api_key: str, pricing_data_file: str, memory_dir: str = "user_memories"):
        """
        개인화된 요금제 추천 에이전트 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            pricing_data_file: 요금제 정보가 담긴 텍스트 파일 경로
            memory_dir: 사용자 메모리 파일들이 저장될 디렉토리
        """
        # OpenAI API 키 설정
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500
        )
        
        # 분석용 LLM (temperature 낮음)
        self.analysis_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000
        )
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # 디렉토리 설정
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.profiles_dir = self.memory_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        self.export_dir = Path("chat_history")
        self.export_dir.mkdir(exist_ok=True)
        
        # 현재 사용자 정보
        self.current_user: Optional[str] = None
        self.user_profile: Optional[UserProfile] = None
        self.chat_history: List[Dict[str, Any]] = []
        self.conversation_summary: str = ""
        self.recommendation_history: List[RecommendationHistory] = []
        
        # 대화 관리 설정
        self.max_conversation_length = 10
        self.summary_threshold = 8
        
        # RAG 시스템 초기화
        self.setup_rag_system(pricing_data_file)
        
        # 할루시네이션 검증 시스템
        self.plan_database = {}  # 요금제 정보 데이터베이스
        self.extract_plan_information()
        
        # 챗봇 체인 생성
        self.setup_chatbot_chain()
        
        # 분석 체인 생성
        self.setup_analysis_chains()
        
        # 할루시네이션 검증 체인 생성
        self.setup_hallucination_check_chains()
    
    def setup_rag_system(self, pricing_data_file: str):
        """RAG 시스템 설정"""
        try:
            loader = TextLoader(pricing_data_file, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print(f"RAG 시스템 초기화 완료: {len(splits)}개의 문서 청크 생성됨")
            
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류 발생: {e}")
            self.retriever = None
    
    def extract_plan_information(self):
        """RAG 문서에서 요금제 정보를 구조화하여 추출"""
        try:
            if not self.retriever:
                print("⚠️ RAG 시스템이 초기화되지 않아 요금제 정보를 추출할 수 없습니다.")
                return
            
            # 모든 문서에서 요금제 정보 검색
            all_docs = self.vectorstore.similarity_search("요금제", k=20)
            
            # 요금제 정보 파싱 체인 생성
            extraction_prompt = ChatPromptTemplate.from_template("""
            다음 요금제 정보에서 모든 요금제의 정확한 정보를 JSON 형태로 추출해주세요.
            
            텍스트: {text}
            
            다음 형태로 응답해주세요:
            {{
                "plans": [
                    {{
                        "name": "정확한 요금제명",
                        "monthly_fee": 월요금(숫자만),
                        "data": "데이터 용량",
                        "calls": "통화 정보",
                        "sms": "문자 정보",
                        "features": ["특징1", "특징2"],
                        "benefits": ["혜택1", "혜택2"],
                        "target_users": ["대상 사용자1", "대상 사용자2"]
                    }}
                ]
            }}
            
            JSON만 응답하세요.
            """)
            
            extraction_chain = extraction_prompt | self.analysis_llm | StrOutputParser()
            
            # 모든 문서에서 요금제 정보 추출
            all_text = "\n".join([doc.page_content for doc in all_docs])
            
            response = extraction_chain.invoke({"text": all_text})
            
            import json
            extracted_data = json.loads(response)
            
            # 요금제 데이터베이스 구축
            for plan in extracted_data.get("plans", []):
                plan_name = plan.get("name", "").strip()
                if plan_name:
                    self.plan_database[plan_name.lower()] = plan
                    
                    # 별명이나 줄임말도 추가
                    name_parts = plan_name.split()
                    for part in name_parts:
                        if len(part) > 2:
                            self.plan_database[part.lower()] = plan
            
            print(f"✅ 요금제 데이터베이스 구축 완료: {len(self.plan_database)}개 항목")
            
        except Exception as e:
            print(f"⚠️ 요금제 정보 추출 중 오류: {e}")
            self.plan_database = {}
    
    def setup_chatbot_chain(self):
        """개인화된 챗봇 체인 설정"""
        system_prompt = """
        당신은 개인 맞춤형 통신사 요금제 추천 전문 에이전트입니다.
        사용자의 개인 프로필과 과거 대화 내용을 기반으로 최적화된 서비스를 제공하세요.
        
        **현재 사용자: {current_user}**
        
        **사용자 프로필:**
        {user_profile}
        
        **개인화 지침:**
        1. 사용자의 예산 범위를 항상 고려하세요
        2. 데이터 사용 패턴과 통화/문자 습관을 분석하세요
        3. 사용자 유형(학생, 직장인, 가족, 시니어)에 맞는 추천을 하세요
        4. 과거 추천 기록과 피드백을 참고하세요
        5. 사용자의 불만사항을 해결하는 방향으로 추천하세요
        6. 가성비와 사용자 만족도를 균형있게 고려하세요
        
        **대화 스타일:**
        - 친근하고 전문적인 톤을 유지하세요
        - 개인의 상황을 이해한다는 느낌을 주세요
        - 추천 이유를 구체적으로 설명하세요
        - 필요시 프로필 정보를 수집하세요
        
        **컨텍스트 정보:**
        {context}
        
        **이전 대화 내용:**
        {chat_history}
        
        **추천 기록:**
        {recommendation_history}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        if self.retriever:
            question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        else:
            self.rag_chain = self.prompt | self.llm | StrOutputParser()
    
    def setup_analysis_chains(self):
        """분석용 체인들 설정"""
        # 프로필 추출 체인
        profile_prompt = ChatPromptTemplate.from_template("""
        다음 대화에서 사용자의 요금제 관련 정보를 추출해주세요.
        
        대화 내용: {conversation}
        
        다음 정보들을 JSON 형태로 추출해주세요:
        - max_budget: 최대 예산 (원)
        - preferred_budget: 선호 예산 (원)
        - monthly_data_usage: 월 데이터 사용량 (GB)
        - call_minutes: 월 통화 시간 (분)
        - user_type: 사용자 유형 (학생/직장인/가족/시니어/기타)
        - family_size: 가족 구성원 수
        - age: 나이
        - priority_features: 중요하게 생각하는 기능들 (리스트)
        - current_plan: 현재 요금제
        - pain_points: 현재 요금제 불만사항 (리스트)
        
        확실하지 않은 정보는 null로 설정하세요.
        JSON만 응답하세요.
        """)
        
        self.profile_extraction_chain = profile_prompt | self.analysis_llm | StrOutputParser()
        
        # 추천 체인
        recommendation_prompt = ChatPromptTemplate.from_template("""
        사용자 프로필과 요금제 정보를 바탕으로 최적의 요금제를 추천해주세요.
        
        사용자 프로필:
        {user_profile}
        
        요금제 정보:
        {plans_info}
        
        추천 기준:
        1. 예산 범위 내에서 최적의 가성비
        2. 데이터 사용량에 맞는 적절한 데이터 용량
        3. 사용자 유형별 특화 혜택
        4. 과거 불만사항 해결
        
        다음 형태로 응답해주세요:
        {{
            "recommended_plan": "추천 요금제명",
            "reason": "추천 이유 (구체적으로)",
            "pros": ["장점1", "장점2", "장점3"],
            "cons": ["단점1", "단점2"] (있다면),
            "monthly_cost": 예상 월 요금,
            "savings": "현재 대비 절약액 (있다면)",
            "alternative_plans": ["대안1", "대안2"]
        }}
        
        JSON만 응답하세요.
        """)
        
        self.recommendation_chain = recommendation_prompt | self.analysis_llm | StrOutputParser()
    
    def setup_hallucination_check_chains(self):
        """할루시네이션 검증용 체인들 설정"""
        # 요금제 존재 확인 체인
        verification_prompt = ChatPromptTemplate.from_template("""
        다음 추천된 요금제가 제공된 요금제 목록에 실제로 존재하는지 확인해주세요.
        
        추천된 요금제: {recommended_plan}
        추천 내용: {recommendation_content}
        
        실제 요금제 목록:
        {actual_plans}
        
        다음을 확인해주세요:
        1. 요금제명이 정확히 일치하는가?
        2. 언급된 가격 정보가 정확한가?
        3. 언급된 데이터 용량이 정확한가?
        4. 언급된 혜택들이 실제로 존재하는가?
        
        JSON 형태로 응답해주세요:
        {{
            "plan_exists": true/false,
            "exact_match": true/false,
            "closest_match": "가장 유사한 실제 요금제명",
            "discrepancies": ["차이점1", "차이점2"],
            "confidence_score": 0.0-1.0
        }}
        """)
        
        self.verification_chain = verification_prompt | self.analysis_llm | StrOutputParser()
        
        # 사실성 검증 체인
        fact_check_prompt = ChatPromptTemplate.from_template("""
        추천된 요금제 정보와 실제 문서의 내용을 비교하여 사실성을 검증해주세요.
        
        추천 정보: {recommendation}
        실제 문서 내용: {source_documents}
        
        다음 항목들의 정확성을 평가해주세요:
        - 요금제명
        - 월 요금
        - 데이터 용량
        - 통화/문자 조건
        - 혜택 및 특징
        
        JSON으로 응답:
        {{
            "accuracy_score": 0.0-1.0,
            "verified_facts": ["검증된 사실1", "검증된 사실2"],
            "false_claims": ["잘못된 정보1", "잘못된 정보2"],
            "missing_info": ["누락된 정보1", "누락된 정보2"]
        }}
        """)
        
        self.fact_check_chain = fact_check_prompt | self.analysis_llm | StrOutputParser()
    
    def extract_profile_info(self, conversation_text: str) -> Dict[str, Any]:
        """대화에서 프로필 정보 추출"""
        try:
            response = self.profile_extraction_chain.invoke({
                "conversation": conversation_text
            })
            
            # JSON 파싱
            import json
            profile_data = json.loads(response)
            
            # None 값들을 실제 None으로 변환
            for key, value in profile_data.items():
                if value == "null" or value == "None":
                    profile_data[key] = None
            
            return profile_data
            
        except Exception as e:
            print(f"프로필 정보 추출 중 오류: {e}")
            return {}
    
    def update_user_profile(self, new_info: Dict[str, Any]):
        """사용자 프로필 업데이트"""
        if not self.user_profile:
            return
        
        for key, value in new_info.items():
            if value is not None and hasattr(self.user_profile, key):
                if key in ['priority_features', 'pain_points']:
                    # 리스트 타입은 기존 값과 병합
                    existing_list = getattr(self.user_profile, key) or []
                    if isinstance(value, list):
                        # 중복 제거하면서 병합
                        combined = list(set(existing_list + value))
                        setattr(self.user_profile, key, combined)
                else:
                    setattr(self.user_profile, key, value)
        
        # 업데이트 시간 갱신
        self.user_profile.updated_at = datetime.now().isoformat()
    
    def get_personalized_recommendation(self) -> Dict[str, Any]:
        """개인화된 요금제 추천 생성"""
        if not self.user_profile:
            return {"error": "사용자 프로필이 없습니다."}
        
        try:
            # 요금제 정보 검색
            if self.retriever:
                docs = self.retriever.get_relevant_documents("요금제 추천")
                plans_info = "\n".join([doc.page_content for doc in docs])
            else:
                plans_info = "요금제 정보를 불러올 수 없습니다."
            
            # 프로필을 문자열로 변환
            profile_str = self.format_user_profile()
            
            response = self.recommendation_chain.invoke({
                "user_profile": profile_str,
                "plans_info": plans_info
            })
            
            import json
            recommendation = json.loads(response)
            
            # 추천 기록에 저장
            rec_history = RecommendationHistory(
                timestamp=datetime.now().isoformat(),
                recommended_plan=recommendation.get("recommended_plan", "Unknown"),
                reason=recommendation.get("reason", "No reason provided")
            )
            self.recommendation_history.append(rec_history)
            
            return recommendation
            
        except Exception as e:
            print(f"추천 생성 중 오류: {e}")
            return {"error": f"추천 생성 실패: {e}"}
    
    def check_plan_hallucination(self, recommended_plan: str, recommendation_content: str) -> HallucinationCheck:
        """추천된 요금제의 할루시네이션 여부 검증"""
        try:
            # 1. 정확한 이름 매칭
            plan_name_lower = recommended_plan.lower().strip()
            exact_match = plan_name_lower in self.plan_database
            
            # 2. 퍼지 매칭으로 유사한 요금제 찾기
            best_match = None
            best_score = 0
            
            for db_plan_name in self.plan_database.keys():
                # 여러 유사도 알고리즘 사용
                ratio_score = fuzz.ratio(plan_name_lower, db_plan_name) / 100
                partial_score = fuzz.partial_ratio(plan_name_lower, db_plan_name) / 100
                token_score = fuzz.token_sort_ratio(plan_name_lower, db_plan_name) / 100
                
                # 가중 평균
                combined_score = (ratio_score * 0.4 + partial_score * 0.3 + token_score * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = db_plan_name
            
            # 3. RAG 문서에서 직접 검증
            if self.retriever:
                verification_docs = self.retriever.get_relevant_documents(
                    f"{recommended_plan} 요금제 정보"
                )
                source_content = "\n".join([doc.page_content for doc in verification_docs])
            else:
                source_content = "RAG 시스템을 사용할 수 없습니다."
            
            # 4. LLM을 통한 상세 검증
            actual_plans = "\n".join([
                f"- {name}: {info.get('monthly_fee', 'N/A')}원, {info.get('data', 'N/A')}, {info.get('features', [])}"
                for name, info in list(self.plan_database.items())[:10]  # 상위 10개만
            ])
            
            verification_result = self.verification_chain.invoke({
                "recommended_plan": recommended_plan,
                "recommendation_content": recommendation_content,
                "actual_plans": actual_plans
            })
            
            import json
            verification_data = json.loads(verification_result)
            
            # 5. 사실성 검증
            fact_check_result = self.fact_check_chain.invoke({
                "recommendation": recommendation_content,
                "source_documents": source_content
            })
            
            fact_data = json.loads(fact_check_result)
            
            # 6. 종합 평가
            plan_exists = exact_match or best_score > 0.8
            confidence_score = max(
                1.0 if exact_match else best_score,
                verification_data.get("confidence_score", 0),
                fact_data.get("accuracy_score", 0)
            )
            
            # 불일치 사항 수집
            discrepancies = []
            discrepancies.extend(verification_data.get("discrepancies", []))
            discrepancies.extend(fact_data.get("false_claims", []))
            
            # 증거 수집
            evidence = []
            evidence.extend(fact_data.get("verified_facts", []))
            if best_match and best_score > 0.5:
                evidence.append(f"가장 유사한 요금제: {best_match} (유사도: {best_score:.2f})")
            
            return HallucinationCheck(
                plan_exists=plan_exists,
                confidence_score=confidence_score,
                matched_plan=best_match if best_score > 0.5 else None,
                discrepancies=discrepancies,
                evidence=evidence
            )
            
        except Exception as e:
            print(f"⚠️ 할루시네이션 검증 중 오류: {e}")
            return HallucinationCheck(
                plan_exists=False,
                confidence_score=0.0,
                discrepancies=[f"검증 오류: {e}"]
            )
    
    def get_verified_recommendation(self) -> Dict[str, Any]:
        """할루시네이션 검증이 포함된 개인화된 요금제 추천"""
        if not self.user_profile:
            return {"error": "사용자 프로필이 없습니다."}
        
        try:
            # 1. 기본 추천 생성
            recommendation = self.get_personalized_recommendation()
            
            if "error" in recommendation:
                return recommendation
            
            # 2. 할루시네이션 검증
            recommended_plan = recommendation.get("recommended_plan", "")
            recommendation_content = json.dumps(recommendation, ensure_ascii=False)
            
            hallucination_check = self.check_plan_hallucination(
                recommended_plan, recommendation_content
            )
            
            # 3. 검증 결과를 추천에 포함
            recommendation["verification"] = {
                "plan_exists": hallucination_check.plan_exists,
                "confidence_score": hallucination_check.confidence_score,
                "matched_plan": hallucination_check.matched_plan,
                "discrepancies": hallucination_check.discrepancies,
                "evidence": hallucination_check.evidence,
                "verification_status": self._get_verification_status(hallucination_check)
            }
            
            # 4. 신뢰도가 낮으면 대안 제시
            if hallucination_check.confidence_score < 0.7:
                recommendation["warning"] = "⚠️ 추천된 요금제 정보의 정확성이 확실하지 않습니다."
                
                if hallucination_check.matched_plan:
                    recommendation["suggested_alternative"] = hallucination_check.matched_plan
                    recommendation["alternative_reason"] = f"'{hallucination_check.matched_plan}'가 더 정확한 요금제명일 수 있습니다."
            
            return recommendation
            
        except Exception as e:
            print(f"검증된 추천 생성 중 오류: {e}")
            return {"error": f"검증된 추천 생성 실패: {e}"}
    
    def _get_verification_status(self, check: HallucinationCheck) -> str:
        """검증 상태를 문자열로 반환"""
        if check.confidence_score >= 0.9:
            return "✅ 높은 신뢰도 - 정확한 정보"
        elif check.confidence_score >= 0.7:
            return "🟡 보통 신뢰도 - 대체로 정확"
        elif check.confidence_score >= 0.5:
            return "🟠 낮은 신뢰도 - 일부 불일치 가능"
        else:
            return "❌ 매우 낮은 신뢰도 - 정보 확인 필요"
    
    def get_plan_database_info(self) -> Dict[str, Any]:
        """요금제 데이터베이스 정보 반환"""
        return {
            "total_plans": len(set(plan["name"] for plan in self.plan_database.values() if "name" in plan)),
            "total_entries": len(self.plan_database),
            "sample_plans": list(set(plan.get("name", "Unknown") for plan in list(self.plan_database.values())[:5]))
        }
    
    def analyze_conversation_for_profile(self):
        """현재 대화 내용을 분석하여 프로필 업데이트"""
        if not self.chat_history:
            return
        
        # 최근 대화 내용 수집
        recent_conversations = self.chat_history[-3:]  # 최근 3개 대화
        conversation_text = ""
        
        for conv in recent_conversations:
            conversation_text += f"사용자: {conv['human']}\n"
            conversation_text += f"챗봇: {conv['ai']}\n\n"
        
        # 프로필 정보 추출
        extracted_info = self.extract_profile_info(conversation_text)
        
        if extracted_info:
            self.update_user_profile(extracted_info)
            print(f"✅ 프로필 정보가 업데이트되었습니다: {list(extracted_info.keys())}")
    
    def format_user_profile(self) -> str:
        """사용자 프로필을 문자열로 포맷팅"""
        if not self.user_profile:
            return "프로필 정보가 없습니다."
        
        profile_parts = [f"사용자명: {self.user_profile.username}"]
        
        if self.user_profile.max_budget:
            profile_parts.append(f"최대 예산: {self.user_profile.max_budget:,}원")
        if self.user_profile.preferred_budget:
            profile_parts.append(f"선호 예산: {self.user_profile.preferred_budget:,}원")
        if self.user_profile.monthly_data_usage:
            profile_parts.append(f"월 데이터 사용량: {self.user_profile.monthly_data_usage}GB")
        if self.user_profile.user_type:
            profile_parts.append(f"사용자 유형: {self.user_profile.user_type}")
        if self.user_profile.family_size:
            profile_parts.append(f"가족 구성원 수: {self.user_profile.family_size}명")
        if self.user_profile.current_plan:
            profile_parts.append(f"현재 요금제: {self.user_profile.current_plan}")
        if self.user_profile.priority_features:
            profile_parts.append(f"중요 기능: {', '.join(self.user_profile.priority_features)}")
        if self.user_profile.pain_points:
            profile_parts.append(f"불만사항: {', '.join(self.user_profile.pain_points)}")
        
        return "\n".join(profile_parts)
    
    def format_recommendation_history(self) -> str:
        """추천 기록을 문자열로 포맷팅"""
        if not self.recommendation_history:
            return "이전 추천 기록이 없습니다."
        
        history_parts = []
        for i, rec in enumerate(self.recommendation_history[-3:], 1):  # 최근 3개만
            history_parts.append(f"{i}. {rec.recommended_plan}")
            history_parts.append(f"   이유: {rec.reason}")
            if rec.user_response:
                history_parts.append(f"   사용자 반응: {rec.user_response}")
        
        return "\n".join(history_parts)
    
    def get_profile_file_path(self, username: str) -> Path:
        """프로필 파일 경로 생성"""
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        if not safe_username:
            safe_username = "unknown_user"
        return self.profiles_dir / f"{safe_username}_profile.json"
    
    def load_user_profile(self, username: str):
        """사용자 프로필 로드"""
        profile_file = self.get_profile_file_path(username)
        
        if profile_file.exists():
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                self.user_profile = UserProfile(**profile_data)
                
                # 추천 기록도 로드
                if 'recommendation_history' in profile_data:
                    self.recommendation_history = []
                    for rec_data in profile_data['recommendation_history']:
                        self.recommendation_history.append(RecommendationHistory(**rec_data))
                
                print(f"✅ {username}님의 프로필을 불러왔습니다.")
                return True
                
            except Exception as e:
                print(f"⚠️ 프로필 로드 중 오류: {e}")
                self.user_profile = UserProfile(username=username)
                return False
        else:
            self.user_profile = UserProfile(username=username)
            print(f"👋 {username}님의 새 프로필을 생성합니다.")
            return False
    
    def save_user_profile(self):
        """사용자 프로필 저장"""
        if not self.user_profile:
            return False
        
        try:
            profile_file = self.get_profile_file_path(self.user_profile.username)
            
            # 프로필 데이터 준비
            profile_data = asdict(self.user_profile)
            
            # 추천 기록 추가
            profile_data['recommendation_history'] = [
                asdict(rec) for rec in self.recommendation_history
            ]
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"⚠️ 프로필 저장 중 오류: {e}")
            return False
    
    def get_memory_file_path(self, username: str) -> Path:
        """대화 메모리 파일 경로 생성"""
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        if not safe_username:
            safe_username = "unknown_user"
        return self.memory_dir / f"{safe_username}_memory.json"
    
    def load_user_memory(self, username: str) -> bool:
        """사용자 대화 메모리 로드"""
        self.current_user = username
        memory_file = self.get_memory_file_path(username)
        
        # 프로필 로드
        profile_exists = self.load_user_profile(username)
        
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                    
                self.chat_history = user_data.get('chat_history', [])
                self.conversation_summary = user_data.get('conversation_summary', "")
                
                print(f"✅ {username}님의 대화 기록을 불러왔습니다.")
                print(f"   - 저장된 대화 수: {len(self.chat_history)}개")
                print(f"   - 프로필 존재: {'예' if profile_exists else '아니오'}")
                return True
                
            except Exception as e:
                print(f"⚠️ 메모리 파일 로드 중 오류 발생: {e}")
                self.chat_history = []
                self.conversation_summary = ""
                return False
        else:
            self.chat_history = []
            self.conversation_summary = ""
            print(f"👋 {username}님, 새로운 대화를 시작합니다!")
            return False
    
    def save_user_memory(self) -> bool:
        """사용자 메모리 및 프로필 저장"""
        if not self.current_user:
            return False
        
        try:
            # 대화 메모리 저장
            memory_file = self.get_memory_file_path(self.current_user)
            
            user_data = {
                'username': self.current_user,
                'chat_history': self.chat_history,
                'conversation_summary': self.conversation_summary,
                'last_login': datetime.now().isoformat(),
                'total_conversations': len(self.chat_history),
                'created_at': datetime.now().isoformat() if not memory_file.exists() else None
            }
            
            # 기존 파일이 있으면 created_at 유지
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    user_data['created_at'] = existing_data.get('created_at', datetime.now().isoformat())
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            # 프로필 저장
            profile_saved = self.save_user_profile()
            
            return profile_saved
            
        except Exception as e:
            print(f"⚠️ 메모리 저장 중 오류 발생: {e}")
            return False
    
    def chat(self, user_input: str) -> str:
        """개인화된 챗봇 대화"""
        if not self.current_user:
            return "⚠️ 먼저 사용자 이름을 설정해주세요. login_user() 메소드를 사용하세요."
        
        try:
            # 대화 기록 포맷팅
            chat_history_str = self.format_chat_history()
            profile_str = self.format_user_profile()
            rec_history_str = self.format_recommendation_history()
            
            if self.retriever:
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "current_user": self.current_user,
                    "user_profile": profile_str,
                    "recommendation_history": rec_history_str
                })
                
                if isinstance(response, dict) and "answer" in response:
                    ai_response = response["answer"]
                else:
                    ai_response = str(response)
            else:
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "context": "요금제 정보 파일을 로드할 수 없습니다.",
                    "current_user": self.current_user,
                    "user_profile": profile_str,
                    "recommendation_history": rec_history_str
                })
                ai_response = response
            
            # 대화 기록에 추가
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": user_input,
                "ai": ai_response
            }
            self.chat_history.append(conversation_entry)
            
            # 프로필 정보 분석 및 업데이트
            self.analyze_conversation_for_profile()
            
            # 대화가 임계점을 넘으면 요약 수행
            if len(self.chat_history) > self.max_conversation_length:
                self.summarize_old_conversations()
            
            # 자동 저장
            self.save_user_memory()
            
            return ai_response
            
        except Exception as e:
            error_msg = f"챗봇 응답 생성 중 오류가 발생했습니다: {e}"
            print(error_msg)
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    def generate_recommendation_report(self) -> str:
        """할루시네이션 검증이 포함된 개인화된 추천 보고서 생성"""
        recommendation = self.get_verified_recommendation()
        
        if "error" in recommendation:
            return f"추천 생성 실패: {recommendation['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("🎯 검증된 개인 맞춤형 요금제 추천 보고서")
        report.append("=" * 60)
        report.append(f"📅 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"👤 사용자: {self.current_user}")
        report.append("")
        
        # 사용자 프로필 요약
        report.append("📊 사용자 프로필 요약")
        report.append("-" * 30)
        if self.user_profile.max_budget:
            report.append(f"💰 최대 예산: {self.user_profile.max_budget:,}원")
        if self.user_profile.monthly_data_usage:
            report.append(f"📱 월 데이터 사용량: {self.user_profile.monthly_data_usage}GB")
        if self.user_profile.user_type:
            report.append(f"👥 사용자 유형: {self.user_profile.user_type}")
        report.append("")
        
        # 추천 요금제
        report.append("🏆 추천 요금제")
        report.append("-" * 30)
        report.append(f"📋 요금제명: {recommendation.get('recommended_plan', 'N/A')}")
        report.append(f"💵 예상 월 요금: {recommendation.get('monthly_cost', 'N/A')}원")
        report.append("")
        
        # 🆕 검증 결과 섹션
        if "verification" in recommendation:
            verification = recommendation["verification"]
            report.append("🔍 신뢰성 검증 결과")
            report.append("-" * 30)
            report.append(f"📈 신뢰도 점수: {verification['confidence_score']:.1%}")
            report.append(f"🎯 검증 상태: {verification['verification_status']}")
            report.append(f"✅ 요금제 존재: {'예' if verification['plan_exists'] else '아니오'}")
            
            if verification['matched_plan']:
                report.append(f"🔗 매칭된 요금제: {verification['matched_plan']}")
            
            if verification['evidence']:
                report.append("🔍 검증 근거:")
                for evidence in verification['evidence']:
                    report.append(f"  • {evidence}")
            
            if verification['discrepancies']:
                report.append("⚠️ 발견된 불일치:")
                for discrepancy in verification['discrepancies']:
                    report.append(f"  • {discrepancy}")
            
            report.append("")
        
        # 경고 메시지
        if "warning" in recommendation:
            report.append("⚠️ 주의사항")
            report.append("-" * 30)
            report.append(recommendation["warning"])
            
            if "suggested_alternative" in recommendation:
                report.append(f"💡 제안: {recommendation['suggested_alternative']}")
                report.append(f"   {recommendation.get('alternative_reason', '')}")
            
            report.append("")
        
        # 추천 이유
        report.append("✅ 추천 이유")
        report.append("-" * 30)
        report.append(recommendation.get('reason', 'N/A'))
        report.append("")
        
        # 장점
        if 'pros' in recommendation and recommendation['pros']:
            report.append("👍 주요 장점")
            report.append("-" * 30)
            for i, pro in enumerate(recommendation['pros'], 1):
                report.append(f"{i}. {pro}")
            report.append("")
        
        # 단점
        if 'cons' in recommendation and recommendation['cons']:
            report.append("👎 고려사항")
            report.append("-" * 30)
            for i, con in enumerate(recommendation['cons'], 1):
                report.append(f"{i}. {con}")
            report.append("")
        
        # 대안 요금제
        if 'alternative_plans' in recommendation and recommendation['alternative_plans']:
            report.append("🔄 대안 요금제")
            report.append("-" * 30)
            for i, alt in enumerate(recommendation['alternative_plans'], 1):
                report.append(f"{i}. {alt}")
            report.append("")
        
        # 절약 효과
        if 'savings' in recommendation and recommendation['savings']:
            report.append("💸 절약 효과")
            report.append("-" * 30)
            report.append(recommendation['savings'])
            report.append("")
        
        return "\n".join(report)
    
    def summarize_old_conversations(self) -> str:
        """오래된 대화들을 요약"""
        if len(self.chat_history) <= self.summary_threshold:
            return self.conversation_summary
        
        try:
            conversations_to_summarize = self.chat_history[:-self.summary_threshold]
            
            conversation_text = ""
            for conv in conversations_to_summarize:
                conversation_text += f"사용자: {conv['human']}\n"
                conversation_text += f"챗봇: {conv['ai']}\n\n"
            
            summary_prompt = ChatPromptTemplate.from_template("""
            다음은 {username}님과의 개인화된 요금제 상담 대화 내용입니다. 
            사용자의 특성과 요구사항을 중심으로 요약해주세요:
            
            1. 사용자의 예산 및 사용 패턴
            2. 추천했던 요금제들과 사용자 반응
            3. 사용자의 선호도와 불만사항
            4. 개인화 정보 (가족 구성, 나이, 직업 등)
            5. 기타 중요한 맥락 정보
            
            대화 내용:
            {conversation_text}
            
            요약 (한국어로 간결하게):
            """)
            
            if self.conversation_summary:
                conversation_text = f"[이전 대화 요약]\n{self.conversation_summary}\n\n[새로운 대화]\n{conversation_text}"
            
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            new_summary = summary_chain.invoke({
                "username": self.current_user,
                "conversation_text": conversation_text
            })
            
            self.chat_history = self.chat_history[-self.summary_threshold:]
            self.conversation_summary = new_summary
            
            print(f"💾 대화 요약 완료: {len(conversations_to_summarize)}개 대화가 요약되었습니다.")
            
            return new_summary
            
        except Exception as e:
            print(f"⚠️ 대화 요약 중 오류 발생: {e}")
            return self.conversation_summary
    
    def format_chat_history(self) -> str:
        """대화 기록을 문자열로 포맷팅 (요약 포함)"""
        formatted_parts = []
        
        if self.conversation_summary:
            formatted_parts.append("[이전 대화 요약]")
            formatted_parts.append(self.conversation_summary)
            formatted_parts.append("\n[최근 대화]")
        
        if not self.chat_history:
            if not self.conversation_summary:
                formatted_parts.append("이전 대화 내용이 없습니다.")
        else:
            for entry in self.chat_history:
                formatted_parts.append(f"사용자: {entry['human']}")
                formatted_parts.append(f"챗봇: {entry['ai']}")
        
        return "\n".join(formatted_parts)
    
    def login_user(self, username: str) -> Dict[str, Any]:
        """사용자 로그인 및 개인화 설정"""
        if not username or not username.strip():
            return {
                "success": False,
                "message": "올바른 사용자 이름을 입력해주세요.",
                "is_new_user": False
            }
        
        username = username.strip()
        is_existing_user = self.load_user_memory(username)
        
        return {
            "success": True,
            "message": f"안녕하세요, {username}님!" + 
                      (" 개인화된 서비스를 제공하겠습니다." if is_existing_user 
                       else " 새로운 프로필을 생성하여 맞춤 서비스를 시작하겠습니다."),
            "is_new_user": not is_existing_user,
            "username": username,
            "conversation_count": len(self.chat_history),
            "has_profile": bool(self.user_profile and 
                              (self.user_profile.max_budget or self.user_profile.user_type))
        }
    
    def logout_user(self) -> bool:
        """현재 사용자 로그아웃"""
        if self.current_user:
            save_success = self.save_user_memory()
            
            if save_success:
                print(f"👋 {self.current_user}님, 안녕히 가세요! 개인화 정보가 저장되었습니다.")
            else:
                print(f"⚠️ {self.current_user}님의 정보 저장 중 오류가 발생했습니다.")
            
            self.current_user = None
            self.user_profile = None
            self.chat_history = []
            self.conversation_summary = ""
            self.recommendation_history = []
            return save_success
        else:
            print("현재 로그인된 사용자가 없습니다.")
            return True
    
    def get_user_profile_summary(self) -> Dict[str, Any]:
        """사용자 프로필 요약 정보 반환"""
        if not self.user_profile:
            return {"error": "프로필이 없습니다."}
        
        return {
            "username": self.user_profile.username,
            "max_budget": self.user_profile.max_budget,
            "preferred_budget": self.user_profile.preferred_budget,
            "monthly_data_usage": self.user_profile.monthly_data_usage,
            "user_type": self.user_profile.user_type,
            "family_size": self.user_profile.family_size,
            "current_plan": self.user_profile.current_plan,
            "priority_features": self.user_profile.priority_features,
            "pain_points": self.user_profile.pain_points,
            "profile_completeness": self.calculate_profile_completeness(),
            "total_recommendations": len(self.recommendation_history),
            "created_at": self.user_profile.created_at,
            "updated_at": self.user_profile.updated_at
        }
    
    def calculate_profile_completeness(self) -> float:
        """프로필 완성도 계산 (0-100%)"""
        if not self.user_profile:
            return 0.0
        
        total_fields = 9  # 주요 필드 개수
        filled_fields = 0
        
        fields_to_check = [
            'max_budget', 'monthly_data_usage', 'user_type', 
            'current_plan', 'priority_features', 'pain_points',
            'family_size', 'age', 'preferred_budget'
        ]
        
        for field in fields_to_check:
            value = getattr(self.user_profile, field, None)
            if value is not None:
                if isinstance(value, list) and len(value) > 0:
                    filled_fields += 1
                elif not isinstance(value, list):
                    filled_fields += 1
        
        return round((filled_fields / total_fields) * 100, 1)
    
    def update_recommendation_feedback(self, plan_name: str, feedback: str, accepted: bool = None):
        """추천에 대한 사용자 피드백 업데이트"""
        for rec in reversed(self.recommendation_history):
            if rec.recommended_plan == plan_name:
                rec.user_response = feedback
                rec.accepted = accepted
                rec.feedback = feedback
                break
        
        # 프로필에 피드백 반영
        if not accepted and feedback:
            if feedback not in self.user_profile.pain_points:
                self.user_profile.pain_points.append(feedback)
        
        self.save_user_memory()
    
    def get_smart_questions(self) -> List[str]:
        """프로필 완성도를 높이기 위한 스마트 질문 생성"""
        if not self.user_profile:
            return ["먼저 로그인을 해주세요."]
        
        questions = []
        
        if not self.user_profile.max_budget:
            questions.append("💰 월 통신비 예산은 어느 정도로 생각하고 계신가요?")
        
        if not self.user_profile.monthly_data_usage:
            questions.append("📱 평소 한 달에 데이터를 얼마나 사용하시나요?")
        
        if not self.user_profile.user_type:
            questions.append("👥 학생, 직장인, 가족 중 어디에 해당하시나요?")
        
        if not self.user_profile.current_plan:
            questions.append("📋 현재 사용하고 계신 요금제가 있나요?")
        
        if not self.user_profile.priority_features:
            questions.append("🎯 통신 서비스에서 가장 중요하게 생각하는 것은 무엇인가요? (데이터, 통화, 가격, 혜택 등)")
        
        if self.user_profile.user_type == "가족" and not self.user_profile.family_size:
            questions.append("👨‍👩‍👧‍👦 가족 구성원은 몇 명이신가요?")
        
        return questions[:3]  # 최대 3개까지만 반환
    
    def list_all_users(self) -> List[Dict[str, Any]]:
        """저장된 모든 사용자 목록 반환 (프로필 정보 포함)"""
        users = []
        
        for memory_file in self.memory_dir.glob("*_memory.json"):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                username = user_data.get('username', 'Unknown')
                
                # 프로필 정보 로드
                profile_file = self.get_profile_file_path(username)
                profile_info = {}
                
                if profile_file.exists():
                    try:
                        with open(profile_file, 'r', encoding='utf-8') as pf:
                            profile_data = json.load(pf)
                            profile_info = {
                                'max_budget': profile_data.get('max_budget'),
                                'user_type': profile_data.get('user_type'),
                                'current_plan': profile_data.get('current_plan'),
                                'total_recommendations': len(profile_data.get('recommendation_history', []))
                            }
                    except:
                        pass
                
                user_info = {
                    'username': username,
                    'total_conversations': user_data.get('total_conversations', 0),
                    'last_login': user_data.get('last_login', 'Unknown'),
                    'created_at': user_data.get('created_at', 'Unknown'),
                    **profile_info
                }
                
                users.append(user_info)
                
            except Exception as e:
                print(f"사용자 파일 읽기 오류 ({memory_file}): {e}")
        
        return sorted(users, key=lambda x: x['last_login'], reverse=True)
    
    def delete_user_data(self, username: str) -> bool:
        """사용자의 모든 데이터 삭제 (메모리 + 프로필)"""
        try:
            memory_file = self.get_memory_file_path(username)
            profile_file = self.get_profile_file_path(username)
            
            deleted_files = []
            
            if memory_file.exists():
                memory_file.unlink()
                deleted_files.append("대화 기록")
            
            if profile_file.exists():
                profile_file.unlink()
                deleted_files.append("프로필")
            
            if deleted_files:
                print(f"✅ {username}님의 {', '.join(deleted_files)}이 삭제되었습니다.")
                return True
            else:
                print(f"⚠️ {username}님의 데이터를 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"⚠️ 데이터 삭제 중 오류 발생: {e}")
            return False
    
    # 기존 메소드들 (get_chat_history, clear_chat_history, save_chat_history 등)은 유지
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """대화 기록 반환"""
        return self.chat_history
    
    def clear_chat_history(self):
        """현재 사용자의 대화 기록 초기화"""
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        self.chat_history = []
        self.conversation_summary = ""
        self.save_user_memory()
        print(f"{self.current_user}님의 대화 기록이 초기화되었습니다.")
    
    def save_chat_history(self, filename: str = None):
        """대화 기록을 별도 파일로 저장"""
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        if filename is None:
            safe_username = "".join(c for c in self.current_user if c.isalnum() or c in ('-', '_')).strip()
            filename = self.export_dir / f"{safe_username}_chat_export.json"
        
        try:
            export_data = {
                'username': self.current_user,
                'export_date': datetime.now().isoformat(),
                'total_conversations': len(self.chat_history),
                'conversation_summary': self.conversation_summary,
                'chat_history': self.chat_history,
                'user_profile': asdict(self.user_profile) if self.user_profile else None,
                'recommendation_history': [asdict(rec) for rec in self.recommendation_history],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"대화 기록과 프로필이 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")


def main():
    """고도화된 챗봇 사용 예시"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PRICING_DATA_FILE = os.getenv("PRICING_DATA_FILE")
    
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your-api-key 를 추가해주세요.")
        return
    
    print(f"✅ OpenAI API Key 로드됨: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
    
    try:
        agent = PersonalizedPricingAgent(OPENAI_API_KEY, PRICING_DATA_FILE)
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")
        return
    
    print("=== 🤖 개인 맞춤형 요금제 추천 에이전트 ===")
    print("✨ 새로운 기능:")
    print("  - 개인 프로필 학습 및 저장")
    print("  - 예산 범위 맞춤 추천")
    print("  - 사용 패턴 분석")
    print("  - 지능형 추천 시스템")
    print("")
    print("📋 명령어:")
    print("  - 'users': 저장된 사용자 목록 보기")
    print("  - 'login [이름]': 사용자 로그인")
    print("  - 'logout': 현재 사용자 로그아웃")
    print("  - 'profile': 내 프로필 보기")
    print("  - 'recommend': 검증된 맞춤 요금제 추천")
    print("  - 'report': 상세 검증 보고서")
    print("  - 'verify': 요금제 데이터베이스 정보")
    print("  - 'check [요금제명]': 특정 요금제 존재 확인")
    print("  - 'questions': 프로필 완성 질문")
    print("  - 'stats': 현재 사용자 통계")
    print("  - 'feedback [요금제명] [피드백]': 추천 피드백")
    print("  - 'clear': 대화 기록 초기화")
    print("  - 'export': 모든 데이터 내보내기")
    print("  - 'delete [이름]': 사용자 데이터 삭제")
    print("  - 'quit' 또는 'exit': 종료")
    print("-" * 60)
    
    while True:
        try:
            if agent.current_user:
                # 프로필 완성도 표시
                completeness = agent.calculate_profile_completeness()
                user_input = input(f"\n[{agent.current_user} | 프로필 {completeness}%] 입력: ").strip()
            else:
                user_input = input("\n[로그인 필요] 입력: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                if agent.current_user:
                    agent.logout_user()
                print("에이전트를 종료합니다. 안녕히 가세요! 🙋‍♀️")
                break
            
            if not user_input:
                continue
            
            # 명령어 처리
            if user_input.lower() == 'users':
                users = agent.list_all_users()
                if users:
                    print("\n📋 저장된 사용자 목록:")
                    for i, user in enumerate(users, 1):
                        budget_info = f" | 예산: {user['max_budget']:,}원" if user['max_budget'] else ""
                        type_info = f" | {user['user_type']}" if user['user_type'] else ""
                        plan_info = f" | {user['current_plan']}" if user['current_plan'] else ""
                        rec_info = f" | 추천 {user['total_recommendations']}회" if user['total_recommendations'] > 0 else ""
                        print(f"  {i}. {user['username']} "
                              f"(대화 {user['total_conversations']}개{budget_info}{type_info}{plan_info}{rec_info})")
                else:
                    print("저장된 사용자가 없습니다.")
                continue
            
            elif user_input.lower().startswith('login '):
                username = user_input[6:].strip()
                if username:
                    result = agent.login_user(username)
                    print(f"\n{result['message']}")
                    if result['success'] and not result['is_new_user']:
                        print(f"저장된 대화 수: {result['conversation_count']}개")
                        print(f"프로필 상태: {'설정됨' if result['has_profile'] else '미설정'}")
                else:
                    print("사용자 이름을 입력해주세요. 예: login 홍길동")
                continue
            
            elif user_input.lower() == 'logout':
                agent.logout_user()
                continue
            
            elif user_input.lower() == 'profile':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    profile_summary = agent.get_user_profile_summary()
                    if 'error' in profile_summary:
                        print(f"⚠️ {profile_summary['error']}")
                    else:
                        print(f"\n👤 {profile_summary['username']}님의 프로필:")
                        print(f"📊 프로필 완성도: {profile_summary['profile_completeness']}%")
                        if profile_summary['max_budget']:
                            print(f"💰 최대 예산: {profile_summary['max_budget']:,}원")
                        if profile_summary['monthly_data_usage']:
                            print(f"📱 월 데이터 사용량: {profile_summary['monthly_data_usage']}GB")
                        if profile_summary['user_type']:
                            print(f"👥 사용자 유형: {profile_summary['user_type']}")
                        if profile_summary['current_plan']:
                            print(f"📋 현재 요금제: {profile_summary['current_plan']}")
                        if profile_summary['priority_features']:
                            print(f"⭐ 중요 기능: {', '.join(profile_summary['priority_features'])}")
                        print(f"🎯 총 추천 받음: {profile_summary['total_recommendations']}회")
                continue
            
            elif user_input.lower() == 'recommend':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    print("🤖 개인 맞춤 요금제를 분석하고 검증 중입니다...")
                    recommendation = agent.get_verified_recommendation()
                    
                    if 'error' in recommendation:
                        print(f"⚠️ {recommendation['error']}")
                    else:
                        print(f"\n🏆 추천 요금제: {recommendation.get('recommended_plan', 'N/A')}")
                        print(f"💵 예상 월 요금: {recommendation.get('monthly_cost', 'N/A')}원")
                        
                        # 검증 결과 표시
                        if 'verification' in recommendation:
                            verification = recommendation['verification']
                            print(f"\n🔍 신뢰성 검증: {verification['verification_status']}")
                            print(f"📈 신뢰도: {verification['confidence_score']:.1%}")
                            
                            if verification['discrepancies']:
                                print("⚠️ 주의사항:")
                                for disc in verification['discrepancies'][:2]:  # 최대 2개만
                                    print(f"  • {disc}")
                        
                        # 경고 메시지
                        if 'warning' in recommendation:
                            print(f"\n{recommendation['warning']}")
                            if 'suggested_alternative' in recommendation:
                                print(f"💡 대안: {recommendation['suggested_alternative']}")
                        
                        print(f"\n✅ 추천 이유:\n{recommendation.get('reason', 'N/A')}")
                        
                        if 'pros' in recommendation and recommendation['pros']:
                            print(f"\n👍 주요 장점:")
                            for pro in recommendation['pros']:
                                print(f"  • {pro}")
                        
                        if 'alternative_plans' in recommendation and recommendation['alternative_plans']:
                            print(f"\n🔄 대안 요금제: {', '.join(recommendation['alternative_plans'])}")
                                                    
                            for pro in recommendation['pros']:
                                print(f"  • {pro}")
                        
                        if 'alternative_plans' in recommendation and recommendation['alternative_plans']:
                            print(f"\n🔄 대안 요금제: {', '.join(recommendation['alternative_plans'])}")
                continue
            
            elif user_input.lower() == 'report':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    print("📊 검증된 상세 추천 보고서를 생성 중입니다...")
                    report = agent.generate_recommendation_report()
                    print(f"\n{report}")
                continue
            
            elif user_input.lower() == 'verify':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    db_info = agent.get_plan_database_info()
                    print(f"\n🔍 요금제 데이터베이스 정보:")
                    print(f"  - 총 요금제 수: {db_info['total_plans']}개")
                    print(f"  - 데이터베이스 항목: {db_info['total_entries']}개")
                    print(f"  - 샘플 요금제: {', '.join(db_info['sample_plans'])}")
                continue
            
            elif user_input.lower().startswith('check '):
                plan_name = user_input[6:].strip()
                if plan_name and agent.current_user:
                    print(f"🔍 '{plan_name}' 요금제 검증 중...")
                    check_result = agent.check_plan_hallucination(plan_name, f"요금제명: {plan_name}")
                    
                    print(f"\n📊 검증 결과:")
                    print(f"  - 요금제 존재: {'✅ 예' if check_result.plan_exists else '❌ 아니오'}")
                    print(f"  - 신뢰도: {check_result.confidence_score:.1%}")
                    
                    if check_result.matched_plan:
                        print(f"  - 가장 유사한 요금제: {check_result.matched_plan}")
                    
                    if check_result.evidence:
                        print("  - 검증 근거:")
                        for evidence in check_result.evidence:
                            print(f"    • {evidence}")
                    
                    if check_result.discrepancies:
                        print("  - 발견된 문제:")
                        for disc in check_result.discrepancies:
                            print(f"    • {disc}")
                else:
                    print("사용법: check [요금제명]")
                continue
            
            elif user_input.lower() == 'questions':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    questions = agent.get_smart_questions()
                    if questions:
                        print("\n🤔 프로필 완성을 위한 질문:")
                        for i, question in enumerate(questions, 1):
                            print(f"  {i}. {question}")
                    else:
                        print("✅ 프로필이 충분히 완성되었습니다!")
                continue
            
            elif user_input.lower() == 'stats':
                if not agent.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    profile_summary = agent.get_user_profile_summary()
                    if 'error' not in profile_summary:
                        print(f"\n📊 {agent.current_user}님의 통계:")
                        print(f"  - 총 대화 수: {len(agent.chat_history)}개")
                        print(f"  - 프로필 완성도: {profile_summary['profile_completeness']}%")
                        print(f"  - 받은 추천 수: {profile_summary['total_recommendations']}회")
                        print(f"  - 프로필 생성일: {profile_summary['created_at'][:19]}")
                        print(f"  - 마지막 업데이트: {profile_summary['updated_at'][:19]}")
                continue
            
            elif user_input.lower().startswith('feedback '):
                parts = user_input[9:].strip().split(' ', 1)
                if len(parts) >= 2:
                    plan_name, feedback = parts[0], parts[1]
                    accepted = '좋다' in feedback or '만족' in feedback
                    agent.update_recommendation_feedback(plan_name, feedback, accepted)
                    print(f"✅ '{plan_name}' 요금제에 대한 피드백이 저장되었습니다.")
                else:
                    print("사용법: feedback [요금제명] [피드백 내용]")
                continue
            
            elif user_input.lower() == 'clear':
                agent.clear_chat_history()
                continue
            
            elif user_input.lower() == 'export':
                agent.save_chat_history()
                continue
            
            elif user_input.lower().startswith('delete '):
                username_to_delete = user_input[7:].strip()
                if username_to_delete:
                    confirm = input(f"⚠️ {username_to_delete}님의 모든 데이터를 삭제하시겠습니까? (y/N): ")
                    if confirm.lower() == 'y':
                        agent.delete_user_data(username_to_delete)
                    else:
                        print("삭제가 취소되었습니다.")
                else:
                    print("삭제할 사용자 이름을 입력해주세요. 예: delete 홍길동")
                continue
            
            # 일반 채팅 처리
            if not agent.current_user:
                print("먼저 로그인을 진행하겠습니다.")
                result = agent.login_user(user_input)
                print(f"{result['message']}")
                if result['success'] and not result['is_new_user']:
                    print(f"저장된 대화 수: {result['conversation_count']}개")
                    print(f"프로필 상태: {'설정됨' if result['has_profile'] else '미설정'}")
                continue
            
            # 개인화된 챗봇 응답 생성
            response = agent.chat(user_input)
            print(f"\n🤖 에이전트: {response}")
            
            
            # 프로필 완성도가 낮으면 힌트 제공
            completeness = agent.calculate_profile_completeness()
            if completeness < 50 and len(agent.chat_history) % 3 == 0:
                questions = agent.get_smart_questions()
                if questions:
                    print(f"\n💡 프로필 힌트: 더 정확한 추천을 위해 이런 정보를 알려주세요:")
                    print(f"   {questions[0]}")
            
            # 할루시네이션 검증 힌트
            elif len(agent.chat_history) % 5 == 0 and agent.plan_database:
                print(f"\n🔍 검증 정보: 현재 {len(set(plan.get('name', '') for plan in agent.plan_database.values()))}개의 검증된 요금제를 기반으로 추천합니다.")
                print("   'verify' 명령어로 데이터베이스 정보를 확인하거나, 'check [요금제명]'으로 특정 요금제를 검증할 수 있습니다.")
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            if agent.current_user:
                agent.logout_user()
            break
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
    
    #+검증버젼