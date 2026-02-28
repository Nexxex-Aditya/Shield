FROM python:3.11-slim

LABEL maintainer="Shield Command Team"
LABEL description="Shield — Intelligent Middleware Platform for Autonomous Systems"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY agentvault/ agentvault/
COPY server/ server/
COPY dashboard/ dashboard/
COPY pipelines/ pipelines/
COPY policies/ policies/
COPY skills/ skills/
COPY shield_cli.py .
COPY setup.py .

# Create data dirs
RUN mkdir -p /app/data /app/pipelines /app/skills

# Env defaults
ENV SHIELD_HOST=0.0.0.0
ENV SHIELD_PORT=8000
ENV DB_URI=sqlite+aiosqlite:///data/shield.db
ENV POLICY_PATH=policies/default_policy.yaml
ENV PIPELINES_DIR=pipelines
ENV SKILLS_DIR=skills

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "server.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
