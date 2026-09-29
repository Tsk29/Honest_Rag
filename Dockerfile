# syntax=docker/dockerfile:1.7
#
# Two-stage build:
#   builder - installs the exact dependency set pinned in uv.lock into a venv
#   runtime - copies only that venv, the embedding model and the app source
#             into a clean slim image,
#             runs as a non-root user, and keeps all mutable state in /data
#
# Build:  docker build -t honestrag .
# Run:    docker compose up   (see docker-compose.yml)

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dependencies first, in their own layer, so editing app code doesn't
# invalidate the (slow) dependency install. --frozen fails the build if
# uv.lock is out of date with pyproject.toml instead of silently re-resolving.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Bake the embedding model (~1.3 GB) into the image. Hosts without a
# persistent disk (e.g. a free Hugging Face Space) would otherwise re-download
# it on the first upload after every restart. Must match the model in app.py.
ARG EMBED_MODEL=BAAI/bge-large-en-v1.5
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBED_MODEL}', cache_dir='/opt/models')"

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=dev
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="HonestRAG" \
      org.opencontainers.image.description="Corrective RAG that grades its own retrieval and fact-checks its answers" \
      org.opencontainers.image.source="https://github.com/Tsk29/Honest_Rag" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

# libgomp: OpenMP runtime needed by onnxruntime (FastEmbed's inference backend)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown app:app /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models

WORKDIR /app
COPY --chown=app:app app.py workflow.py llm_provider.py knowledge_base.py rag_service.py ./
COPY --chown=app:app eval/ ./eval/
COPY --chown=app:app assets/ ./assets/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HONESTRAG_DATA_DIR=/data \
    HONESTRAG_EMBED_CACHE_DIR=/opt/models \
    HONESTRAG_EMBED_OFFLINE=1 \
    HONESTRAG_VERSION=${VERSION} \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
VOLUME ["/data"]
EXPOSE 8501

# Streamlit's built-in liveness endpoint. python instead of curl so the slim
# image doesn't need an extra package just for the healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
