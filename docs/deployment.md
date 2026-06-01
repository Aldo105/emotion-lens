# EmotionLens — Deployment Guide

This guide covers deploying EmotionLens in various environments.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Production Server (Linux)](#production-server-linux)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment (AWS/GCP)](#cloud-deployment)
5. [Environment Variables](#environment-variables)
6. [HTTPS / TLS Setup](#https--tls-setup)
7. [Monitoring](#monitoring)

---

## Local Development

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://anaconda.com)
- NVIDIA GPU + CUDA 12.1 (optional, for GPU acceleration)
- Python 3.11+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/emotion-lens.git
cd emotion-lens

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate emotion-lens

# Start development server
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points
- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws/emotion

---

## Production Server (Linux)

### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.11 python3.11-venv git nginx

# (Optional) Install NVIDIA drivers + CUDA
# Follow: https://developer.nvidia.com/cuda-downloads
```

### 2. Application Setup

```bash
# Create app user
sudo useradd -m -s /bin/bash emotionlens
sudo su - emotionlens

# Clone and setup
git clone https://github.com/your-org/emotion-lens.git
cd emotion-lens
conda env create -f environment.yml
conda activate emotion-lens
```

### 3. Production Server with Gunicorn + Uvicorn Workers

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn backend.app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile /var/log/emotionlens/access.log \
  --error-logfile /var/log/emotionlens/error.log
```

> **Note**: For WebSocket support, use `--workers 1` or configure sticky sessions. Each worker needs its own GPU memory.

### 4. Systemd Service

Create `/etc/systemd/system/emotionlens.service`:

```ini
[Unit]
Description=EmotionLens - Emotion Analysis Platform
After=network.target

[Service]
Type=simple
User=emotionlens
WorkingDirectory=/home/emotionlens/emotion-lens
Environment="PATH=/home/emotionlens/miniconda3/envs/emotion-lens/bin"
ExecStart=/home/emotionlens/miniconda3/envs/emotion-lens/bin/gunicorn \
    backend.app.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable emotionlens
sudo systemctl start emotionlens
```

### 5. Nginx Reverse Proxy

Create `/etc/nginx/sites-available/emotionlens`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 500M;  # For video uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/emotionlens /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart websockets \
    sqlalchemy aiosqlite reportlab aiofiles \
    pydantic pydantic-settings python-dotenv \
    torch torchvision mediapipe opencv-python-headless \
    scipy scikit-learn pillow numpy gunicorn

EXPOSE 8000

CMD ["gunicorn", "backend.app.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  emotionlens:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - USE_GPU=true
      - DEBUG=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

```bash
# Build and run
docker compose up -d

# View logs
docker compose logs -f emotionlens
```

---

## Cloud Deployment

### AWS EC2 (GPU Instance)

1. **Instance type**: `g4dn.xlarge` (T4 GPU, 16GB VRAM) or `g5.xlarge` (A10G)
2. **AMI**: Ubuntu 22.04 with NVIDIA drivers (Deep Learning AMI)
3. **Security Group**: Allow inbound port 80, 443, 8000
4. Follow the [Production Server](#production-server-linux) steps

### Google Cloud (GCP)

1. **Machine type**: `n1-standard-4` with `nvidia-tesla-t4`
2. **Image**: Ubuntu 22.04 with GPU drivers
3. Install CUDA toolkit + follow production steps

### Cost Estimate (AWS)

| Instance | GPU | VRAM | $/hr (on-demand) | $/month |
|----------|-----|------|-------------------|---------|
| g4dn.xlarge | T4 | 16 GB | $0.526 | ~$380 |
| g5.xlarge | A10G | 24 GB | $1.006 | ~$725 |
| g4dn.xlarge (spot) | T4 | 16 GB | ~$0.16 | ~$115 |

---

## Environment Variables

All settings can be overridden via environment variables or a `.env` file:

```bash
# .env file
APP_NAME=EmotionLens
DEBUG=false

# GPU
USE_GPU=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.7
ENABLE_HALF_PRECISION=true
ENABLE_CUDNN_BENCHMARK=true

# Server
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["https://your-domain.com"]

# Database (PostgreSQL for production)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/emotionlens

# Video uploads
MAX_UPLOAD_SIZE_MB=500
UPLOADS_DIR=/data/uploads
REPORTS_DIR=/data/reports
```

### PostgreSQL for Production

```bash
# Install asyncpg
pip install asyncpg

# Set DATABASE_URL
export DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/emotionlens
```

---

## HTTPS / TLS Setup

### Using Certbot (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### WebSocket over TLS

Once HTTPS is configured, update your frontend to use `wss://`:

```javascript
// In config.js
const CONFIG = {
    WS_URL: `wss://${window.location.host}/ws/emotion`,
    API_URL: `https://${window.location.host}/api`,
};
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/health
# {"status": "healthy", "app": "EmotionLens", "version": "0.1.0", "device": "cuda"}
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Log Rotation

Add to `/etc/logrotate.d/emotionlens`:

```
/var/log/emotionlens/*.log {
    daily
    missingok
    rotate 14
    compress
    notifempty
}
```

---

## Security Considerations

1. **CORS**: Restrict `CORS_ORIGINS` to your domain in production
2. **Rate Limiting**: Add FastAPI rate limiting middleware for API endpoints
3. **Authentication**: Consider adding JWT or API key auth for sensitive endpoints
4. **File Uploads**: Validate file types server-side (already implemented)
5. **Database**: Use PostgreSQL with proper credentials for production
6. **HTTPS**: Always use TLS in production for WebSocket + API traffic
