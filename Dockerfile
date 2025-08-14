FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_RUN_HOST=0.0.0.0

WORKDIR /app
COPY . /app

# Install system libs for OpenCV & YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Flask & Ultralytics (default PyPI index)
RUN pip install flask ultralytics opencv-python-headless

# Install PyTorch CPU wheels from official index
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0"]
