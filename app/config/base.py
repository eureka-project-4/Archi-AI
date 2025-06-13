from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    ENVIRONMENT: str = "local"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 3
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7

    # 요청용 스트림 (Spring에서 aiMessage를 보내는 곳)
    REQUEST_STREAM: str = "ai-request-stream"

    # 응답용 스트림 (Spring이 소비하는 곳)
    RESPONSE_STREAM: str = "ai-response-stream"

    # Consumer Group: Spring과 매칭
    CONSUMER_GROUP: str = "request-processor"

    # 이 파이썬 소비자의 이름 (Consumer Group 내 개별 소비자 식별용)
    CONSUMER_NAME: str = "ai-consumer-1"
    
    class Config:
        env_file = ".env"