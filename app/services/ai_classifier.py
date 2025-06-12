import openai
from app.models.message import MessageType, AuthMetadata
from app.config import settings

class AIClassifier:
    def __init__(self):
        # 현재는 실제 OpenAI 호출 사용 안 함
        openai.api_key = settings.OPENAI_API_KEY

    async def classify_message_type(self, content: str, metadata: AuthMetadata) -> MessageType:
        """
        사용자 메시지를 AI로 분류하는 함수입니다.

        [주의] 현재는 테스트를 위해 무조건 SUGGESTION으로 분류되도록 설정되어 있습니다.
        추후 실제 AI 모델 분류 로직으로 복구 필요합니다.
        """
        
        # TODO: 아래 테스트 코드 제거하고 실제 OpenAI 호출 로직으로 교체할 것
        return MessageType.SUGGESTION

        # 아래는 향후 사용할 실제 OpenAI 기반 분류 코드입니다.
        """
        prompt = f'''
        사용자 메시지: "{content}"
        사용자 정보 - 연령대 코드: {metadata.age_code}, 태그 코드: {metadata.tag_code}

        다음 중 어떤 타입에 해당하는지 정확히 하나만 선택하세요:

        1. SUGGESTION - "내 성향에 맞는 조합 추천해줘", "나한테 어울리는 멤버십 찾아줘"
        2. KEYWORD_RECOMMENDATION - "헬스와 카페를 할거야, 어울리는 멤버십 찾아줘", "영화관과 쇼핑 관련 조합"  
        3. PREFERENCE_UPDATE - "요즘 요가에 관심있어", "헬스 그만두고 수영하고싶어"
        4. GENERAL_RESPONSE - 일반 대화, 인사, 질문 등

        응답은 반드시 다음 중 하나만: SUGGESTION, KEYWORD_RECOMMENDATION, PREFERENCE_UPDATE, GENERAL_RESPONSE
        '''

        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 사용자 메시지를 정확히 분류하는 AI입니다. 반드시 지정된 4개 타입 중 하나만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip()
            
            if classification in [t.value for t in MessageType if t != MessageType.USER_MESSAGE]:
                return MessageType(classification)
            else:
                return MessageType.GENERAL_RESPONSE
                
        except Exception as e:
            print(f"AI 분류 실패: {e}")
            return MessageType.GENERAL_RESPONSE
        """

# 인스턴스 생성
ai_classifier = AIClassifier()