FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/retriever/.cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# CPU-only torch first, at the versions requirements.txt pins, so the
# requirements install below sees them satisfied and never pulls the ~2.5 GB
# CUDA wheels from PyPI -- there is no GPU on Railway.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1 torchvision==0.27.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake BGE-M3 into the image: app.py loads it with local_files_only=True, and a
# 2.3 GB download on every cold start would otherwise gate the first answer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

# Railway injects $PORT; Streamlit must bind it on all interfaces.
CMD streamlit run app.py \
      --server.port=${PORT:-8501} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --client.toolbarMode=minimal \
      --browser.gatherUsageStats=false
