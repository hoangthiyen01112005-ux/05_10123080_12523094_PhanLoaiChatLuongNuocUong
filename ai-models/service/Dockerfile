FROM python:3.9-slim

WORKDIR /app

# Copy file dependency và cài đặt
COPY ai-models/service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code AI và Models vào container
COPY ai-models/ /app/ai-models/

WORKDIR /app/ai-models/service

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]