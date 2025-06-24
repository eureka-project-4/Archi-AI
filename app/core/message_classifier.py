import json
import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.models.message import MessageType

class MessageClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.setup_classification_chain()
    
    def setup_classification_chain(self):
        classification_prompt = ChatPromptTemplate.from_template("""
        사용자 입력과 AI 응답을 분석하여 메시지 타입을 정확히 분류해주세요.

        사용자 입력: {human_input}
        AI 응답: {ai_response}
        **critical point: don't include benefit (ex : name:5G 프리미어 에센셜, benefit:u+ tv 구독권 증정 -> result : 5G 프리미어 에센셜)**
        **중요: mentioned_items는 AI가 실제로 추천한 모든 상품/서비스를 포함하세요.**
        - 통신 요금제 (예: 5G 프리미어, LTE 베이직 등)
        - 부가서비스 (예: 로밍 서비스, 하버드 비즈니스 리뷰 구독 등)
        - 쿠폰/혜택 (예: 영화 할인 쿠폰, 쇼핑 쿠폰 등)
        - "추천드립니다", "권해드립니다"와 함께 언급된 항목들 포함

        분류 기준:
        - SUGGESTION: 사용자 성향/프로필 기반 맞춤형 추천...
        - PREFERENCE_UPDATE: 사용자 선호도 변경 관련...
        - GENERAL_RESPONSE: 일반 대화, 인사, 단순 정보 제공, 질문 답변
        - USER_MESSAGE: 사용자가 보낸 원본 메시지 (분류 불필요)
        - FILTERED_MESSAGE: 부적절한 내용 감지됨 (욕설, 비속어 등)

        JSON만 응답:
        {{
            "message_type": "위 6가지 타입 중 하나",
            "mentioned_items": {{
                "plans": ["추천된 요금제명들"],
                "vass": ["추천된 부가서비스명들"],
                "coupons": ["추천된 쿠폰명들"]
            }},
            "has_pricing": true/false,
            "confidence": 0.0-1.0
        }}
        """)
        
        self.classification_chain = classification_prompt | self.llm | StrOutputParser()
    
    def classify_message(self, human_input: str, ai_response: str) -> Dict[str, Any]:
        try:
            # 실제로 체인을 호출
            result = self.classification_chain.invoke({
                "human_input": human_input,
                "ai_response": ai_response
            })
            
            # JSON 추출
            if '```json' in result:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
            
            # JSON 파싱
            data = json.loads(result)
            llm_type_str = data.get("message_type", "GENERAL_RESPONSE")
            mentioned_items = data.get("mentioned_items", {})
            all_mentioned = []
            all_mentioned.extend(mentioned_items.get("plans", []))
            all_mentioned.extend(mentioned_items.get("vass", []))
            all_mentioned.extend(mentioned_items.get("coupons", []))
            # 문자열을 MessageType Enum으로 변환
            type_mapping = {
                "SUGGESTION": MessageType.SUGGESTION,
                "PREFERENCE_UPDATE": MessageType.PREFERENCE_UPDATE,
                "GENERAL_RESPONSE": MessageType.GENERAL_RESPONSE,
                "USER_MESSAGE": MessageType.USER_MESSAGE,
                "FILTERED_MESSAGE": MessageType.FILTERED_MESSAGE,
                # 하위 호환성
                "CHAT": MessageType.GENERAL_RESPONSE,
            }
            
            llm_type = type_mapping.get(llm_type_str, MessageType.GENERAL_RESPONSE)
            
            return {
                "message_type": llm_type,
                "mentioned_plans": all_mentioned,
                "confidence": data.get("confidence", 0.8),
                "has_pricing": data.get("has_pricing", False),
                "reasoning": "LLM 분석"
            }
            
        except Exception as e:
            print(f"분류 오류: {e}")
            return {
                "message_type": MessageType.GENERAL_RESPONSE,
                "mentioned_plans": [],
                "confidence": 0.0,
                "has_pricing": False,
                "reasoning": f"분류 실패: {e}"
            }