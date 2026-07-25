"""Encode gold questions with one embedding model, in a process of its own.

Exists purely for process isolation. faiss and torch each link an OpenMP runtime,
and on macOS arm64 having both live in one interpreter makes faiss's search
segfault intermittently -- fixing the import order made the crash rarer but not
gone. So the parent (fig_embedding_score_spread.py) imports faiss and never
torch, and every model's encode happens here, in a child process that imports
torch and never faiss.

Usage: _encode_queries.py <model_id> <prefix> <questions.json> <out.npy>
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = str(Path("models/retriever/.cache").resolve())
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

model_id, prefix, questions_path, out_path = sys.argv[1:5]
questions = json.loads(Path(questions_path).read_text())

model = SentenceTransformer(model_id, device="cpu")
embeddings = model.encode(
    [prefix + question for question in questions],
    normalize_embeddings=True,
    batch_size=16,
    show_progress_bar=False,
)
np.save(out_path, np.asarray(embeddings, dtype="float32"))
print(f"encoded {len(questions)} questions with {model_id} -> {np.shape(embeddings)}")
