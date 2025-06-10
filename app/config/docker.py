from .base import BaseSettings
class DockerSettings(BaseSettings):
    DATA_PATH: str = "/app/data"
    PRICING_DATA_DIR: str = "/app/data/pricing"
    VECTOR_STORE_DIR: str = "/app/data/vectors"
    MEMORY_DIR: str = "/app/data/user_memories"
    EXPORT_DIR: str = "/app/data/chat_history"

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024

    RETRIEVAL_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    ENVIRONMENT: str = "docker"

    class Config:
        env_file = ".env.docker"