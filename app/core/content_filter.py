import json
import openai
from typing import Dict, Any
from app.config import settings

class ContentFilter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def contains_forbidden_content(self, content: str) -> bool:
        """
        GPT-3.5-turbo를 사용하여 욕설 여부만 판단
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                         "content": (
                            "You are a content moderation system. Your task is to analyze a user's input "
                            "and determine whether it contains **any** of the following:\n"
                            "- profanity (swear words, vulgar language)\n"
                            "- hate speech\n"
                            "- discriminatory or offensive content based on gender, race, ethnicity, or religion\n"
                            "- sexist, racist, or other derogatory remarks\n\n"
                            "If the input contains any of the above, respond with:\n"
                            '{ "isProfane": true }\n\n'
                            "If the input is clean and does not contain any inappropriate language or offensive content, respond with:\n"
                            '{ "isProfane": false }\n\n'
                            "Respond strictly in JSON format. Do not include any explanation or additional text."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the following sentence: \"{content}\". Is it offensive? Respond true or false."
                    } 
                ]
            )

            result_json = response.choices[0].message.content.strip()
            result = json.loads(result_json)

            return result.get("isProfane", False)

        except Exception as e:
            print(f"[ERROR] GPT Moderation 판단 실패: {e}")
            return False
# 인스턴스 생성
content_filter = ContentFilter()
