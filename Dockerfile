FROM python:3.12-slim
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY . .
ENV DATA_DIR=/data
CMD ["sh", "-c", "uvicorn live.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
