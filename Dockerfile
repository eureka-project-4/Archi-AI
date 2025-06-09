# archi-ai/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치 (ML 라이브러리들 때문에 필요)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 먼저 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# 사용자 생성 (보안)
RUN adduser --disabled-password --gecos '' fastapi
USER fastapi

EXPOSE 8080

# uvicorn으로 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]