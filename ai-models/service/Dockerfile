FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY service ./service
COPY models ./models

WORKDIR /app/service

CMD ["python", "main.py"]