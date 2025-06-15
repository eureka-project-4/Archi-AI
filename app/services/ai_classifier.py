from app.core.message_classifier import MessageClassifier
from app.models.message import MessageType, AuthMetadata
from app.config import settings
from langchain_openai import ChatOpenAI
import os

class AIClassifier:
    def __init__(self):
        # 현재는 실제 OpenAI 호출 사용 안 함
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        # MessageClassifier 인스턴스 생성
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            max_tokens=1000
        )
        self.message_classifier = MessageClassifier(llm)

    async def classify_message_type(self, content: str, metadata: AuthMetadata) -> MessageType:
        """
        consumer.py와의 인터페이스 맞춤
        """
        # 일단 USER_MESSAGE로 가정 (AI 응답이 아직 없으므로)
        # 실제로는 processor에서 AI 응답 후 재분류됨
        
        # 간단한 키워드 기반 사전 분류
        if any(word in content for word in ['추천', '알려', '찾아']):
            if '키워드' in content or '관련' in content:
                return MessageType.KEYWORD_RECOMMENDATION
            else:
                return MessageType.SUGGESTION
        elif any(word in content for word in ['바꾸', '변경', '이제는']):
            return MessageType.PREFERENCE_UPDATE
        else:
            return MessageType.USER_MESSAGE

# 인스턴스 생성
ai_classifier = AIClassifier()