from typing import ClassVar
from .base import BaseSettings

from .base import BaseSettings
from typing import ClassVar

class LocalSettings(BaseSettings):
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
