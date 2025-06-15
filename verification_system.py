import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fuzzywuzzy import fuzz
from data_models import HallucinationCheck

class VerificationSystem:
    def __init__(self, analysis_llm: ChatOpenAI):
        self.analysis_llm = analysis_llm
        self.plan_database = {}
        self.setup_verification_chains()
    
    def setup_verification_chains(self):
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
    
    def extract_plan_information(self, vectorstore):
        try:
            if not vectorstore:
                print("⚠️ 벡터스토어가 없어 요금제 정보를 추출할 수 없습니다.")
                return
            
            all_docs = vectorstore.similarity_search("요금제", k=20)
            
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
            all_text = "\n".join([doc.page_content for doc in all_docs])
            response = extraction_chain.invoke({"text": all_text})
            extracted_data = json.loads(response)
            
            for plan in extracted_data.get("plans", []):
                plan_name = plan.get("name", "").strip()
                if plan_name:
                    self.plan_database[plan_name.lower()] = plan
                    name_parts = plan_name.split()
                    for part in name_parts:
                        if len(part) > 2:
                            self.plan_database[part.lower()] = plan
            
            print(f"✅ 요금제 데이터베이스 구축 완료: {len(self.plan_database)}개 항목")
            
        except Exception as e:
            print(f"⚠️ 요금제 정보 추출 중 오류: {e}")
            self.plan_database = {}
    
    def check_plan_hallucination(self, recommended_plan: str, recommendation_content: str, retriever=None) -> HallucinationCheck:
        try:
            plan_name_lower = recommended_plan.lower().strip()
            exact_match = plan_name_lower in self.plan_database
            
            best_match = None
            best_score = 0
            
            for db_plan_name in self.plan_database.keys():
                ratio_score = fuzz.ratio(plan_name_lower, db_plan_name) / 100
                partial_score = fuzz.partial_ratio(plan_name_lower, db_plan_name) / 100
                token_score = fuzz.token_sort_ratio(plan_name_lower, db_plan_name) / 100
                combined_score = (ratio_score * 0.4 + partial_score * 0.3 + token_score * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = db_plan_name
            
            if retriever:
                verification_docs = retriever.get_relevant_documents(f"{recommended_plan} 요금제 정보")
                source_content = "\n".join([doc.page_content for doc in verification_docs])
            else:
                source_content = "RAG 시스템을 사용할 수 없습니다."
            
            actual_plans = "\n".join([
                f"- {name}: {info.get('monthly_fee', 'N/A')}원, {info.get('data', 'N/A')}, {info.get('features', [])}"
                for name, info in list(self.plan_database.items())[:10]
            ])
            
            verification_result = self.verification_chain.invoke({
                "recommended_plan": recommended_plan,
                "recommendation_content": recommendation_content,
                "actual_plans": actual_plans
            })
            
            verification_data = json.loads(verification_result)
            
            fact_check_result = self.fact_check_chain.invoke({
                "recommendation": recommendation_content,
                "source_documents": source_content
            })
            
            fact_data = json.loads(fact_check_result)
            
            plan_exists = exact_match or best_score > 0.8
            confidence_score = max(
                1.0 if exact_match else best_score,
                verification_data.get("confidence_score", 0),
                fact_data.get("accuracy_score", 0)
            )
            
            discrepancies = []
            discrepancies.extend(verification_data.get("discrepancies", []))
            discrepancies.extend(fact_data.get("false_claims", []))
            
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
    
    def get_verification_status_message(self, confidence_score: float) -> str:
        if confidence_score >= 0.9:
            return "✅ 높은 신뢰도 - 정확한 정보"
        elif confidence_score >= 0.7:
            return "🟡 보통 신뢰도 - 대체로 정확"
        elif confidence_score >= 0.5:
            return "🟠 낮은 신뢰도 - 일부 불일치 가능"
        else:
            return "❌ 매우 낮은 신뢰도 - 정보 확인 필요"
    
    def get_plan_database_info(self) -> Dict[str, Any]:
        unique_plans = set()
        for plan in self.plan_database.values():
            if "name" in plan:
                unique_plans.add(plan["name"])
        
        return {
            "total_plans": len(unique_plans),
            "total_entries": len(self.plan_database),
            "sample_plans": list(unique_plans)[:5]
        }
    
    def find_mentioned_plans(self, text: str) -> List[str]:
        mentioned_plans = []
        text_lower = text.lower()
        
        for plan_key in self.plan_database.keys():
            if plan_key in text_lower:
                plan_info = self.plan_database[plan_key]
                original_name = plan_info.get("name", plan_key)
                if original_name not in mentioned_plans:
                    mentioned_plans.append(original_name)
        
        return mentioned_plans