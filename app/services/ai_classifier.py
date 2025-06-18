# app/services/ai_classifier.py 수정
from app.core.message_classifier import MessageClassifier
from app.models.message import MessageType, AuthMetadata
from app.config import settings
from langchain_openai import ChatOpenAI

class AIClassifier:
    def __init__(self):
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            max_tokens=1000
        )
        self.message_classifier = MessageClassifier(llm)

    async def classify_with_ai_response(self, user_input: str, ai_response: str) -> dict:
        """
        AI 응답까지 생성된 후 정확한 분류를 수행하는 메서드
        processor.py에서 호출되어 정확한 분류 결과를 얻을 수 있음
        """
        try:
            classification_result = self.message_classifier.classify_message(user_input, ai_response)
            
            return {
                'message_type': classification_result.get('message_type', MessageType.GENERAL_RESPONSE),
                'mentioned_plans': classification_result.get('mentioned_plans', []),
                'reasoning': classification_result.get('reasoning', ''),
                'confidence': classification_result.get('confidence', 0.0),
                'has_pricing': classification_result.get('has_pricing', False)
            }
            
        except Exception as e:
            print(f"AI 응답 기반 분류 오류: {e}")
            return {
                'message_type': MessageType.GENERAL_RESPONSE,
                'mentioned_plans': [],
                'reasoning': f'분류 오류: {e}',
                'confidence': 0.0,
                'has_pricing': False
            }

# 인스턴스 생성
ai_classifier = AIClassifier()