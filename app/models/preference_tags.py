DATA_USAGE = ["data_heavy", "data_light"]
CALL_SMS = ["call_heavy", "call_light", "sms_heavy"]
NETWORK_PLAN = ["use_5g", "use_lte", "roaming", "plan_changer"]


PREFERENCE_TAGS = [
    "ott", "netflix", "disney", "tiving", "youtube",
    "music", "ebook", "audiobook", "webtoon", "webnovel",
    "book", "life", "living_alone", "family", "friend",
    "date", "kids", "pet", "english", "chinese",
    "edu", "it", "news", "stock", "self_Improvement",
    "hobby", "game", "baseball", "movie", "soccer",
    "car", "photo", "kakaotalk", "sns", "travel",
    "eating_out", "delivery", "shop", "food", "cafe",
    "coffee", "desert", "bread", "conveniencestore", "clothes"
]

FEW_SHOT = [
            {
                "role": "user",
                "content": "이미지: YouTube가 80% 이상 사용. 조건 동일."
            },
            {
                "role": "assistant",
                "content": """
                        {
                        "summary": "주로 YouTube를 많이 시청하는 데이터 집중형 사용자입니다.",
                        "tags": ["data_heavy", "call_light", "use_5g", "youtube" , "kakaotalk" , "self_Improvement" , "it" , "conveniencestore" ]
                        }
                        """
            },

            {
                "role": "user",
                "content": "이미지: YouTube와 Instagram 사용 비중이 각각 높음. 조건 동일."
            },
            {
                "role": "assistant",
                "content": """
                        {
                        "summary": "YouTube와 Instagram을 주로 사용하는 영상·소셜 중심형 사용자입니다.",
                        "tags": ["data_heavy", "use_5g", , "sms_heavy" ,"youtube", "sns", "kakaotalk", "photo", "cafe"]
                        }
                        """
            },
            {
                "role": "user",
                "content": "이미지: Instagram 사용 비중이 가장 높고 금융, 카카오톡, 네이버 사용도 있음. 조건 동일."
            },
            {
                "role": "assistant",
                "content": """
                        {
                        "summary": "Instagram과 금융·정보 앱을 균형 있게 사용하는 실속형 소셜 사용자입니다.",
                        "tags": ["data_light", "call_light", "use_lte", "sns", "kakaotalk", "self_Improvement", "news", "stock"]
                        }
                        """
            },
            {
                "role": "user",
                "content": "이미지: YouTube 사용 비중이 높고 짧은 게임과 간단한 생산성 앱 사용도 있음. 조건 동일."
            },
            {
                "role": "assistant",
                "content": """
                        {
                        "summary": "YouTube 시청과 짧은 모바일 게임으로 간단한 여가를 즐기는 실속형 엔터테인먼트 사용자입니다.",
                        "tags": ["data_light", "call_light", "use_5g", "youtube", "game", "hobby", "living_alone", "conveniencestore"]
                        }
                        """
            },
            {
                "role": "user",
                "content": "이미지: 흐릿하고 잘림. 조건 동일."
            },
            {
                "role": "assistant",
                "content": """
                {
                "summary": "이미지를 다시 업로드해주세요.",
                "tags": []
                }
                """
                },

                {
                "role": "user",
                "content": "이미지: 관련 없는 이미지 (예: 얼굴, 풍경 등) 업로드됨. 조건 동일."
                },
                {
                "role": "assistant",
                "content": """
                    {
                    "summary": "이미지를 다시 업로드해주세요.",
                    "tags": []
                    }
                    """
    }

]