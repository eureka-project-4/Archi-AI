import openai
from app.config import settings
import os
#  금칙어 필터링
class ContentFilter:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    
    async def contains_forbidden_content(self, content: str) -> bool:
        """
        OpenAI Moderation API를 사용하여 부적절한 콘텐츠 감지
        """
        try:
            # 최신 omni-moderation-latest 모델 사용
            response = self.client.moderations.create(
                model="omni-moderation-latest",
                input=content
            )
            
            # 결과 분석
            result = response.results[0]
            
            # 하나라도 플래그되면 True 반환
            is_flagged = result.flagged
            
            if is_flagged:
                # 플래그된 카테고리 확인
                flagged_categories = []
                categories = result.categories
                
                # 모든 카테고리 체크
                for category in dir(categories):
                    if not category.startswith('_') and getattr(categories, category):
                        flagged_categories.append(category)
                
                print(f"[MODERATION] 부적절한 콘텐츠 감지: {flagged_categories}")
                
                # 점수 확인 (디버깅용)
                scores = result.category_scores
                high_scores = {}
                for category in dir(scores):
                    if not category.startswith('_'):
                        score = getattr(scores, category)
                        if score > 0.5:
                            high_scores[category] = score
                
                if high_scores:
                    print(f"[MODERATION] 높은 점수 카테고리: {high_scores}")
            
            return is_flagged
            
        except Exception as e:
            print(f"[ERROR] Moderation API 오류: {e}")
            # 오류 시 안전을 위해 False 반환
            return False

# 인스턴스 생성
content_filter = ContentFilter()