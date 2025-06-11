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
            result = self.classification_chain.invoke({
                "ai_response": ai_response
            })
            if '```json' in result:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
            data = json.loads(result)
            llm_type = data.get("message_type", "SUGGESTION")
            llm_plans = data.get("mentioned_plans", [])

            # Fallback (더 엄격한 정규식 적용)
            if not llm_plans:
                pattern = r'([가-힣A-Za-z0-9\- ]{2,}(?:요금제|플랜))'
                for match in re.findall(pattern, ai_response):
                    cleaned = match.strip()
                    if any(x in cleaned for x in ['추천', '있', '알려', '문의', '주세요', '있는', '추천해', '추천해줘']):
                        continue
                    llm_plans.append(cleaned)
            return {
                "message_type": llm_type,
                "mentioned_plans": list(set(llm_plans)),
                "reasoning": "LLM+정규식"
            }
        except Exception as e:
            return {
                "message_type": "chat",
                "mentioned_plans": [],
                "reasoning": f"오류: {e}"
            }
