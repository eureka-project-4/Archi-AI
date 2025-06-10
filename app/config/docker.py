from .base import BaseSettings
class DockerSettings(BaseSettings):
    # 기존 설정
    DATA_PATH: str = "/app/data"
    PRICING_DATA_DIR: str = "/app/data/pricing"
    VECTOR_STORE_DIR: str = "/app/data/vectors"

    # ✅ DB 설정
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # ✅ OpenAI 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024

    # ✅ RAG 설정
    RETRIEVAL_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ✅ 환경 구분
    ENVIRONMENT: str = "docker"

    class Config:
        env_file = ".env.docker"