FROM python:3.10-slim

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/api_server.py backend/train_yolo_abacaxi.py ./
COPY backend/modelo_yolo/ ./modelo_yolo/
COPY backend/images/ ./images/

# Copy frontend folder to parent directory so BASE_DIR.parent / "frontend" works
COPY frontend/ /frontend/

# Expose port and run
ENV PORT=8000
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT
