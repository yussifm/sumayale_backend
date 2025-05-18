FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg \
 && pip install --no-cache-dir -r requirements.txt

COPY app app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
