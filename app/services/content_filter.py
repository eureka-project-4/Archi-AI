import openai
from app.config import settings
#  금칙어 필터링
class ContentFilter:
    def __init__(self):
        # 현재는 OpenAI 연결만 설정, 실제 호출은 비활성화 상태입니다.
        openai.api_key = settings.OPENAI_API_KEY
    
    async def contains_forbidden_content(self, content: str) -> bool:
        """
        메시지가 욕설/비속어/부적절한 표현을 포함하는지 판단하는 함수입니다.

        [현재 상태]
        - 테스트 용도로 항상 정상적인 메시지(False)를 반환합니다.
        - 금칙어 필터링은 비활성화 되어 있습니다.

        [팀원 작업 메모]
        - 추후 OpenAI API를 사용해 금칙어 필터링을 활성화하려면
          아래의 `return False` 라인을 주석 처리하고,
          주석 블록을 해제하면 됩니다.
        - 응답은 "YES" 또는 "NO" 중 하나로 오며, "YES"일 경우 금칙어 포함으로 처리합니다.
        """
        return False  # TODO: 실제 필터링 로직 구현 전까지는 항상 통과 처리

        # 아래는 향후 사용할 실제 OpenAI 기반 금칙어 판별 로직입니다.
        """
        prompt = f'''
        다음 메시지가 욕설, 비속어, 혐오 표현, 부적절한 내용을 포함하는지 판단하세요:

        메시지: "{content}"

        판단 기준:
        - 직접적인 욕설이나 비속어
        - 우회적 표현 (ㅅㅂ, sㅣval 등)
        - 혐오 표현이나 차별적 언어
        - 성적이거나 폭력적인 내용
        - 기타 부적절한 표현

        응답은 "YES" 또는 "NO"만 해주세요.
        YES: 부적절한 내용 포함
        NO: 정상적인 내용
        '''

        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 텍스트의 부적절성을 판단하는 AI입니다. 매우 엄격하게 판단하며, 의심스러운 경우 YES로 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            print(f"AI 필터링 실패: {e}")
            return False  # 오류 시 필터링 우회
        """

# 인스턴스 생성
content_filter = ContentFilter()