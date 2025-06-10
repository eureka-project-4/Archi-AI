import os
from .local import LocalSettings
from .docker import DockerSettings

def get_settings():
    env = os.getenv("ENVIRONMENT", "local")
    
    if env == "docker":
        return DockerSettings()
    else:
        return LocalSettings()

settings = get_settings()