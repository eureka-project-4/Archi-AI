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

  
        prompt = (
            "1) Analyze this image and provide a one-sentence summary in Korean describing the user's battery usage behavior.\n"
            "2) If the image is too small, blurry, or cropped and cannot be analyzed properly, the summary must be exactly '이미지가 불완전하여 다시 업로드해주세요' and the tags must be [].\n"
            "3) If the image is valid, select **1 to 5 tags only from the following categories**:\n"
            f"- CONTENT_TAGS: Select at most 1 from {', '.join(CONTENT_TAGS)}\n"
            f"- LIFE_TAGS: Select at most 1 from {', '.join(LIFE_TAGS)}\n"
            f"- SELF_TAGS: Select at most 1 from {', '.join(SELF_TAGS)}\n"
            f"- HOBBY_TAGS: Select at most 1 from {', '.join(HOBBY_TAGS)}\n"
            f"- CONSUME_TAGS: Select at most 1 from {', '.join(CONSUME_TAGS)}\n"
            "From each category, choose only one tag. If multiple candidates exist within a category, select the tag with the highest battery usage percentage.\n"
            "If there is no relevant tag in a category, skip that category.\n"
            "The total number of selected tags must be at least 1 and at most 5.\n\n"
            f"Full preference tag list: {', '.join(ALL_TAGS)}\n\n"
            "You must respond strictly in the following JSON format only:\n"
            "{\n"
            "  \"summary\": \"One-sentence summary in Korean or the error message\",\n"
            "  \"tags\": [\"tag1\", \"tag2\", ...] or []\n"
            "}"
        )




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