# FastAPI prediction service. Built from the project root (see compose.yaml).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# 1. Install dependencies first (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# 2. Copy the code and install the project itself
COPY src ./src
COPY app ./app
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
