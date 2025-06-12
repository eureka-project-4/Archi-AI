import openai
from app.config import settings
from typing import Dict, Any

class ContentFilter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
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
    
    async def get_moderation_details(self, content: str) -> Dict[str, Any]:
        """
        상세한 모더레이션 결과 반환 (디버깅용)
        """
        try:
            response = self.client.moderations.create(
                model="omni-moderation-latest",
                input=content
            )
            
            result = response.results[0]
            
            # Raw 응답 출력
            print("\n[DEBUG] Raw Moderation Response:")
            print(f"Model: {response.model}")
            print(f"ID: {response.id}")
            
            # 카테고리별 상세 정보
            print("\nCategories:")
            for category in result.categories.__dict__:
                if not category.startswith('_'):
                    value = getattr(result.categories, category)
                    score = getattr(result.category_scores, category)
                    print(f"  {category}: {value} (score: {score})")
            
            return {
                'flagged': result.flagged,
                'categories': result.categories,
                'category_scores': result.category_scores,
                'model': response.model,
                'id': response.id
            }
            
        except Exception as e:
            print(f"[ERROR] Moderation API 오류: {e}")
            return {
                'flagged': False,
                'error': str(e)
            }
# 인스턴스 생성
content_filter = ContentFilter()