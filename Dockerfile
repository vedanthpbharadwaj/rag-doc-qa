FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image small; sentence-transformers runs fine on CPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ragqa ./ragqa
COPY eval ./eval

RUN useradd --create-home appuser && mkdir -p /app/data /app/.cache && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
