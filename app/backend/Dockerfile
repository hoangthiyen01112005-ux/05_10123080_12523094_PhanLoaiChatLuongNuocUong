FROM python:3.12-slim

WORKDIR /app

# Cài đặt trực tiếp các thư viện cho Backend Gateway
RUN pip install --no-cache-dir fastapi>=0.100.0 uvicorn>=0.22.0 httpx>=0.24.0 pydantic>=2.0.0

# Copy mã nguồn backend
COPY src ./src

EXPOSE 5000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]