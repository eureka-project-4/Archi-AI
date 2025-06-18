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
        **중요: mentioned_plans는 AI가 실제로 추천한 요금제만 포함하세요.**
        - "추천드립니다", "권해드립니다"와 함께 언급된 요금제만 포함
        - 단순히 비교용으로 언급된 요금제는 제외
        - "대신에", "또는" 같은 대안으로 언급된 것도 제외
        분류 기준:
        - SUGGESTION: 사용자 성향/프로필 기반 맞춤형 추천. "조건에 맞는", "고객님께 추천", "프로필 기반" 등
        - PREFERENCE_UPDATE: 사용자 선호도 변경 관련. "이제는 ~를 선호", "~로 바꾸고 싶어", "요즘은" 등
        - PROACTIVE_SUGGESTION: 시스템 주도적 추천. "이번 달 추천", "새로운 요금제 출시", "정기 추천" 등
        - GENERAL_RESPONSE: 일반 대화, 인사, 단순 정보 제공, 질문 답변
        - USER_MESSAGE: 사용자가 보낸 원본 메시지 (분류 불필요)
        - FILTERED_MESSAGE: 부적절한 내용 감지됨 (욕설, 비속어 등)
        
        JSON만 응답:
        {{
            "message_type": "위 6가지 타입 중 하나",
            "mentioned_plans": ["언급된 요금제명들"],
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
            llm_plans = data.get("mentioned_plans", [])
            
            # 문자열을 MessageType Enum으로 변환
            type_mapping = {
                "SUGGESTION": MessageType.SUGGESTION,
                "PREFERENCE_UPDATE": MessageType.PREFERENCE_UPDATE,
                "PROACTIVE_SUGGESTION": MessageType.PROACTIVE_SUGGESTION,
                "GENERAL_RESPONSE": MessageType.GENERAL_RESPONSE,
                "USER_MESSAGE": MessageType.USER_MESSAGE,
                "FILTERED_MESSAGE": MessageType.FILTERED_MESSAGE,
                # 하위 호환성
                "CHAT": MessageType.GENERAL_RESPONSE,
            }
            
            llm_type = type_mapping.get(llm_type_str, MessageType.GENERAL_RESPONSE)
            
            return {
                "message_type": llm_type,
                "mentioned_plans": llm_plans,
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