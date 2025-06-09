from .base import BaseSettings

class DockerSettings(BaseSettings):
    DATA_PATH: str = "/app/data"
    PRICING_DATA_DIR: str = "/app/data/pricing"
    VECTOR_STORE_DIR: str = "/app/data/vectors"