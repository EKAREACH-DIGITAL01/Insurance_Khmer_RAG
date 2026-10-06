"""Streamlit caches for expensive model and index resources."""

import json
import os

import streamlit as st
from rank_bm25 import BM25Okapi

from khmer_rag.config import (
    ACTIVE_EMBEDDING_MODEL_DIR,
    EMBEDDING_LOCAL_FILES_ONLY,
    VECTOR_DB_ROOT,
)
from khmer_rag.language import khmer_aware_tokenize


@st.cache_resource(show_spinner=False)
def resolve_vector_db():
    """Return the active index, metadata path, and embedding model information.

    All three artifacts are required; guessing another model can silently
    produce incorrect retrieval."""
    db_dir = f"{VECTOR_DB_ROOT}/{ACTIVE_EMBEDDING_MODEL_DIR}"
    faiss_path = f"{db_dir}/khmer_insurance.index"
    metadata_path = f"{db_dir}/khmer_insurance_metadata.json"
    info_path = f"{db_dir}/embedding_model_info.json"

    missing = [
        os.path.basename(p) for p in (faiss_path, metadata_path, info_path) if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"{db_dir}/ is missing {', '.join(missing)}. "
            f"Run: python scripts/08_generate_embeddings.py"
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return faiss_path, metadata_path, info


@st.cache_resource(show_spinner=False)
def load_embedding_model_info():
    _, _, info = resolve_vector_db()
    return info


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    info = load_embedding_model_info()
    kwargs = {
        "trust_remote_code": bool(info.get("trust_remote_code")),
        "local_files_only": EMBEDDING_LOCAL_FILES_ONLY,
    }
    return SentenceTransformer(info["hf_id"], device="cpu", **kwargs)


@st.cache_resource
def load_index():
    faiss_path, _, _ = resolve_vector_db()
    import faiss

    faiss.omp_set_num_threads(1)
    return faiss.read_index(faiss_path)


@st.cache_resource
def load_metadata():
    _, metadata_path, _ = resolve_vector_db()
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_bm25_index(_chunks):
    """Cache the lexical ranker for the loaded corpus.

    Streamlit excludes _chunks from hashing. Restart the app after an index
    rebuild so FAISS, metadata, and BM25 are loaded together."""
    corpus_tokens = [
        khmer_aware_tokenize(chunk.get("embedding_text") or chunk["text"]) for chunk in _chunks
    ]
    return BM25Okapi(corpus_tokens)
