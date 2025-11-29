# 1. 베이스 이미지: 가장 가벼운 파이썬 리눅스 버전 (Slim)
FROM python:3.10-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (gcc 등 빌드 도구) & 캐시 삭제로 용량 확보
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사
COPY requirements.txt .

# 5. [핵심] PyTorch CPU 버전 강제 설치 (RAM 최적화)
# requirements.txt 설치 전에 이걸 먼저 해서 가벼운 버전을 선점합니다.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. 나머지 라이브러리 설치 (FastAPI, Transformers 등)
RUN pip install --no-cache-dir -r requirements.txt

# 7. 소스 코드 및 모델 폴더 통째로 복사
# (emotion_model 폴더가 반드시 이 위치에 있어야 함)
COPY . .

# 8. Render 실행 명령어
# Render가 할당해주는 포트(PORT) 변수를 받아 실행합니다.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```