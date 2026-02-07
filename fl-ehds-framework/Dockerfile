# =============================================================================
# FL-EHDS Framework - Multi-stage Docker Build
# =============================================================================
# Supports three modes via FL_EHDS_MODE environment variable:
#   - terminal:  Interactive CLI (questionary + tqdm)
#   - dashboard: Streamlit web dashboard (port 8501)
#   - benchmark: Automated experiment runner
#
# Build:  docker compose build
# Run:    docker compose run --rm terminal
#         docker compose up dashboard
#         docker compose --profile benchmark run --rm benchmark
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller than CUDA variant)
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install terminal-specific dependencies
COPY terminal/requirements.txt terminal_requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r terminal_requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

LABEL maintainer="Fabio Liberti <fabio.liberti@unimercatorum.it>"
LABEL description="FL-EHDS: Privacy-Preserving Federated Learning for EHDS"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy framework source code
COPY . .

# Create mount points for volumes
RUN mkdir -p /app/data /app/results /app/logs

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FL_EHDS_CONFIG=/app/config/config.yaml
ENV FL_EHDS_MODE=terminal

# Expose Streamlit dashboard port
EXPOSE 8501

# Entrypoint handles mode switching
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
