#!/usr/bin/env python3
"""Incrementally rebuild the active BGE-M3 vector DB.

The student legal corpus is stable and expensive to re-embed on CPU. This
script reuses existing vectors for unchanged legal/guidance chunks and embeds
only the refreshed website-derived chunks, then writes a fresh FAISS index and
metadata file in the same format as scripts/08_generate_embeddings.py.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "models" / "retriever" / ".cache"
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

import faiss
from sentence_transformers import SentenceTransformer

from irc_website_categories import WEBSITE_CATEGORY_LABEL_KH

INPUT_ROOT = PROJECT_ROOT / "data" / "08_merged_chunked_documents"
OUTPUT_ROOT = PROJECT_ROOT / "vector_db" / "BAAI-BGE-M3"
OLD_VECTOR_DB_ROOT = Path(os.environ.get("OLD_VECTOR_DB_DIR", OUTPUT_ROOT))
INDEX_PATH = OUTPUT_ROOT / "khmer_insurance.index"
METADATA_PATH = OUTPUT_ROOT / "khmer_insurance_metadata.json"
MODEL_INFO_PATH = OUTPUT_ROOT / "embedding_model_info.json"
OLD_INDEX_PATH = OLD_VECTOR_DB_ROOT / "khmer_insurance.index"
OLD_METADATA_PATH = OLD_VECTOR_DB_ROOT / "khmer_insurance_metadata.json"

EMBEDDING_HF_ID = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
MAX_EMBEDDING_CHARS = int(os.environ.get("MAX_EMBEDDING_CHARS", "2200"))
REUSE_DOCUMENT_TYPES = {"law", "prakas", "sub_decree", "guidance"}

DOCUMENT_TYPE_KH = {
    "law": "ច្បាប់",
    "prakas": "ប្រកាស",
    "guidance": "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
    "website": "ទំព័រគេហទំព័រ",
}
DOCUMENT_TYPE_KH.update(WEBSITE_CATEGORY_LABEL_KH)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def prepare_chunk(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {}).copy()
    document_type = metadata.get("document_type")
    if document_type:
        metadata["document_type_kh"] = DOCUMENT_TYPE_KH.get(document_type)

    text = (chunk.get("text") or chunk.get("content") or "").strip()
    embedding_parts = []
    for key in (
        "document_name",
        "document_type_kh",
        "parent_h1",
        "chapter",
        "chapter_title",
        "section",
        "section_title",
        "legal_unit",
        "heading_title",
        "source_url",
    ):
        if metadata.get(key):
            embedding_parts.append(metadata[key])
    embedding_parts.append(text)
    embedding_text = "\n".join(embedding_parts).strip()
    if MAX_EMBEDDING_CHARS > 0 and len(embedding_text) > MAX_EMBEDDING_CHARS:
        embedding_text = embedding_text[:MAX_EMBEDDING_CHARS]

    return {
        "chunk_id": chunk["chunk_id"],
        "text": text,
        "embedding_text": embedding_text,
        "metadata": metadata,
    }


def load_current_chunks() -> list[dict]:
    chunks = []
    for path in sorted(INPUT_ROOT.rglob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            chunks.extend(json.load(fh))
    return [prepare_chunk(chunk) for chunk in chunks if (chunk.get("text") or chunk.get("content"))]


def reusable_old_vectors():
    if not OLD_INDEX_PATH.exists() or not OLD_METADATA_PATH.exists():
        return {}
    old_index = faiss.read_index(str(OLD_INDEX_PATH))
    old_chunks = load_json(OLD_METADATA_PATH)
    reusable = {}
    for idx, chunk in enumerate(old_chunks):
        metadata = chunk.get("metadata", {})
        reusable[chunk["chunk_id"]] = {
            "chunk": chunk,
            "vector": old_index.reconstruct(idx),
        }
    return reusable


def main() -> int:
    start = time.time()
    current_chunks = load_current_chunks()
    reusable = reusable_old_vectors()

    final_chunks: list[dict | None] = [None] * len(current_chunks)
    vectors: list[np.ndarray | None] = [None] * len(current_chunks)
    to_embed = []
    to_embed_positions = []

    reused = 0
    reused_student = 0
    reused_unchanged = 0
    for pos, chunk in enumerate(current_chunks):
        metadata = chunk.get("metadata", {})
        old = reusable.get(chunk["chunk_id"])
        if old and metadata.get("document_type") in REUSE_DOCUMENT_TYPES:
            final_chunks[pos] = old["chunk"]
            vectors[pos] = old["vector"]
            reused += 1
            reused_student += 1
        elif old and (old["chunk"].get("embedding_text") or "") == (
            chunk.get("embedding_text") or ""
        ):
            final_chunks[pos] = chunk
            vectors[pos] = old["vector"]
            reused += 1
            reused_unchanged += 1
        else:
            final_chunks[pos] = chunk
            to_embed.append(chunk["embedding_text"])
            to_embed_positions.append(pos)

    print("========== INCREMENTAL BGE REBUILD ==========")
    print(f"Total chunks       : {len(current_chunks):,}")
    print(f"Reused vectors     : {reused:,}")
    print(f"  student legal    : {reused_student:,}")
    print(f"  unchanged text   : {reused_unchanged:,}")
    print(f"Chunks to embed    : {len(to_embed):,}")
    print(f"Max embedding chars: {MAX_EMBEDDING_CHARS:,}")

    if to_embed:
        model = SentenceTransformer(EMBEDDING_HF_ID, local_files_only=True)
        print(f"Device             : {model.device}")
        encode_start = time.time()
        embedded = model.encode(
            to_embed,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embedded = np.asarray(embedded, dtype="float32")
        print(f"Encode time        : {time.time() - encode_start:.2f} sec")
        for row, pos in enumerate(to_embed_positions):
            vectors[pos] = embedded[row]

    if any(vector is None for vector in vectors):
        raise RuntimeError("Internal error: missing vectors after rebuild.")
    if any(chunk is None for chunk in final_chunks):
        raise RuntimeError("Internal error: missing metadata chunks after rebuild.")

    embeddings = np.vstack([np.asarray(vector, dtype="float32") for vector in vectors])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(final_chunks, fh, ensure_ascii=False, indent=2)
    with MODEL_INFO_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "label": EMBEDDING_HF_ID,
                "hf_id": EMBEDDING_HF_ID,
                "hf_cache_dir": str(HF_CACHE_DIR),
                "trust_remote_code": False,
                "task_aware": False,
                "native_hf": False,
                "embedding_dimension": int(embeddings.shape[1]),
                "build_encode_time_sec": round(time.time() - start, 4),
                "chunk_count": len(final_chunks),
                "incremental_reused_vectors": reused,
                "incremental_reused_student_vectors": reused_student,
                "incremental_reused_unchanged_vectors": reused_unchanged,
                "max_embedding_chars": MAX_EMBEDDING_CHARS,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Vector count       : {index.ntotal:,}")
    print(f"Saved              : {OUTPUT_ROOT}")
    print(f"Total runtime      : {time.time() - start:.2f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
