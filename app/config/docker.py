from .base import BaseSettings
class DockerSettings(BaseSettings):
    DATA_PATH: str = "/app/app/data/"
    PRICING_DATA_DIR: str = DATA_PATH+"pricing"
    VECTOR_STORE_DIR: str = DATA_PATH+"vectors"
    MEMORY_DIR: str = DATA_PATH+"user_memories"
    EXPORT_DIR: str = DATA_PATH+"chat_history"

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024

    RETRIEVAL_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    REQUEST_STREAM: str = "ai-request-stream"
    RESPONSE_STREAM: str = "ai-response-stream"
    CONSUMER_GROUP: str = "request-processor"
    CONSUMER_NAME: str = "ai-consumer-1"


    IMAGE_REQUEST_STREAM : str = "image-request-stream"
    IMAGE_CONSUMER_GROUP : str = "image-request-processor" 
    IMAGE_CONSUMER_NAME : str = "image-consumer-1"
    IMAGE_RESPONSE_STREAM : str = "image-response-stream"

    ENVIRONMENT: str = "docker"

    class Config:
        env_file = ".env.docker"