# Hugging Face Space (Docker SDK) image for QualityGuard AI. Built by Hugging Face from the files uploaded by
# scripts/deploy_hf_space.py: source code + the trained checkpoints the product profiles use (no datasets).
FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch HF_HUB_DISABLE_TELEMETRY=1 GRADIO_ANALYTICS_ENABLED=False
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt
# ImageNet ResNet-18 weights used at inference by the anomaly feature extractor and the component recogniser
RUN python -c "import torchvision.models as m; m.resnet18(weights=m.ResNet18_Weights.DEFAULT)"
COPY --chown=user backend/ backend/
COPY --chown=user ml/ ml/
COPY --chown=user scripts/ scripts/
COPY --chown=user configs/ configs/
COPY --chown=user artifacts/ artifacts/
COPY --chown=user data/ data/
COPY --from=frontend --chown=user /fe/dist frontend/dist
RUN mkdir -p storage && chown -R user:user /app
USER user
# Production settings. QG_SECRET_KEY and QG_DEFAULT_ADMIN_PASSWORD come from Space secrets.
ENV QG_ENVIRONMENT=production QG_SEED_DEMO_USERS=false QG_ALLOW_REGISTRATION=true QG_WORKERS=1 \
    QG_CORS_ORIGINS="" QG_MAX_UPLOAD_MB=15
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:7860/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
