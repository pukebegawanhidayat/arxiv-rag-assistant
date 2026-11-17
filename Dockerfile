FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARXIV_RAG_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU PyTorch first so pip does not pull a CUDA wheel.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts

RUN pip install -e .

EXPOSE 8000 8501

CMD ["uvicorn", "arxiv_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
