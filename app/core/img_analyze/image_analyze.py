import openai
from app.config import settings
from app.models.preference_tags import PREFERENCE_TAGS, CONTENT_TAGS, LIFE_TAGS, SELF_TAGS, HOBBY_TAGS, CONSUME_TAGS, FEW_SHOT

class ImageAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def analyze_image_and_tags(self, base64_image: str) -> dict:
        """
        이미지로부터 성향 요약 & 적합 태그 추출
        """
        
        print("🚀 base64_image 길이:", len(base64_image))

        ALL_TAGS = CONTENT_TAGS + LIFE_TAGS + SELF_TAGS + HOBBY_TAGS + CONSUME_TAGS

  
        prompt = f"""
                Analyze the given image and provide a one-sentence summary in Korean describing the user's battery usage behavior.
                If the image is too small, blurry, or cropped and cannot be analyzed properly, the summary must be exactly "이미지가 불완전하여 다시 업로드해주세요" and the tags must be [].
                If the image is valid, select **1 to 5 tags ONLY** from the following categories:
                - CONTENT_TAGS: {', '.join(CONTENT_TAGS)}
                - LIFE_TAGS: {', '.join(LIFE_TAGS)}
                - SELF_TAGS: {', '.join(SELF_TAGS)}
                - HOBBY_TAGS: {', '.join(HOBBY_TAGS)}
                - CONSUME_TAGS: {', '.join(CONSUME_TAGS)}

                **From each category, you must choose at most ONE tag: if multiple candidates exist within the same category, select ONLY the tag with the highest battery usage percentage and exclude all others.**
                If your selected tags contain multiple tags from the same category, KEEP ONLY the tag with the highest battery usage and REMOVE all others before returning the final JSON.
                If there is no relevant tag in a category, skip that category.
                The total number of selected tags must be at least 1 and at most 5.
                Respond strictly in the following JSON format ONLY:
                {{
                "summary": "One-sentence summary in Korean or the exact error message",
                "tags": ["tag1", "tag2", ...] or []
                }}
            """


        


        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages= FEW_SHOT + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        print("GPT raw answer:", response.choices[0].message.content)

        import json
        import re

        raw_content = response.choices[0].message.content.strip()
        clean_content = re.sub(r"```json|```", "", raw_content).strip()
        result = json.loads(clean_content)

        return result
analyzer = ImageAnalyzer()