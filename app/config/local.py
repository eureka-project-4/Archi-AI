from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# .env 파일 명시적으로 로드
load_dotenv()

class LocalSettings(BaseSettings):
<<<<<<< HEAD
    # 데이터베이스 설정
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "archi_db")
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "8204")
    
    # Redis 설정
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # 기타 설정
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 추가 설정들 (있다면)
    OPENAI_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 5
    
    # 디렉토리 설정
    PRICING_DATA_DIR: str = "app/data/pricing"
    VECTOR_STORE_DIR: str = "app/data/vectors"
    MEMORY_DIR: str = "app/data/user_memories"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
=======
    DATA_PATH: str = "./app/data"
    PRICING_DATA_DIR: str = "./app/data/pricing"
    VECTOR_STORE_DIR: str = "./app/data/vectors"
    MEMORY_DIR: str = "./app/data/user_memories"
    EXPORT_DIR: str = "./app/data/chat_history"
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    PRICING_DATA_FILE: str = "./manuals/plans_manual.txt"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    env_file: ClassVar[str] = ".env"
>>>>>>> bed46aa58795337f2d2f6cb617ed016fdd58dd0c
