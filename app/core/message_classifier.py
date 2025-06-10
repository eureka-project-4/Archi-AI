import json
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MessageClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.setup_classification_chain()
    
    def setup_classification_chain(self):
        classification_prompt = ChatPromptTemplate.from_template("""
        AI 응답을 분석하여 일반 대화인지 요금제 추천인지 판단해주세요.

        AI 응답: {ai_response}

        분류 기준:
        - SUGGESTION: 구체적인 요금제명 언급, 월 요금 제시, "추천", "적합" 등 권유 표현
        - CHAT: 일반적인 대화, 인사, 질문 답변

        JSON만 응답:
        {{
            "message_type": "CHAT" 또는 "SUGGESTION",
            "mentioned_plans": ["요금제명들"],
            "has_pricing": true/false
        }}
        """)
        
        self.classification_chain = classification_prompt | self.llm | StrOutputParser()
    
    def classify_message(self, human_input: str, ai_response: str) -> Dict[str, Any]:
        try:
            # 1. 간단한 키워드 체크 먼저
            suggestion_indicators = [
                '요금제', '추천', '적합', '월.*원', '\d+GB', '혜택', '할인',
                '무제한', '데이터', '통화', '변경', '선택'
            ]
            
            has_suggestion_keywords = any(
                re.search(pattern, ai_response, re.IGNORECASE) 
                for pattern in suggestion_indicators
            )
            
            # 2. 요금제명 추출
            plan_patterns = [
                r'(\w+\s*요금제)',
                r'가족\s*쉐어\s*\d+인'
            ]
            
            mentioned_plans = []
            for pattern in plan_patterns:
                matches = re.findall(pattern, ai_response, re.IGNORECASE)
                mentioned_plans.extend(matches)
            
            # 3. LLM 분류 (키워드가 있을 때만)
            if has_suggestion_keywords:
                try:
                    result = self.classification_chain.invoke({
                        "ai_response": ai_response
                    })
                    
                    # JSON 추출
                    if '```json' in result:
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                        if json_match:
                            result = json_match.group(1)
                    
                    data = json.loads(result)
                    llm_type = data.get("message_type", "SUGGESTION")
                    llm_plans = data.get("mentioned_plans", [])
                    
                    # LLM 결과와 정규식 결과 병합
                    all_plans = list(set(mentioned_plans + llm_plans))
                    
                    return {
                        "message_type": llm_type,
                        "mentioned_plans": all_plans,
                        "reasoning": "LLM + 키워드 분석"
                    }
                    
                except Exception as e:
                    print(f"LLM 분류 실패, 키워드 기반으로 대체: {e}")
            
            # 4. 키워드 기반 최종 판단
            if has_suggestion_keywords or mentioned_plans:
                message_type = "suggestion"
            else:
                message_type = "chat"
            
            return {
                "message_type": message_type,
                "mentioned_plans": mentioned_plans,
                "reasoning": "키워드 기반 분류"
            }
            
        except Exception as e:
            print(f"분류 실패: {e}")
            return {
                "message_type": "chat",
                "mentioned_plans": [],
                "reasoning": f"오류: {e}"
            }