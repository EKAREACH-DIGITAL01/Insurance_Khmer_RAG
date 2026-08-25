import json
import os
import time
import numpy as np
from pathlib import Path

# Set HF_HOME before any HF-related import (sentence_transformers,
# transformers, huggingface_hub all read it at module-init time).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

input_root = Path("data/08_merged_chunked_documents")
output_root = Path("vector_db")

output_root.mkdir(parents=True, exist_ok=True)

MAX_EMBEDDING_CHARS = int(os.environ.get("MAX_EMBEDDING_CHARS", "12000"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
SEQUENTIAL_EMBEDDING = os.environ.get("SEQUENTIAL_EMBEDDING", "0") == "1"

DOCUMENT_TYPE_KH = {
    "law": "ច្បាប់",
    "prakas": "ប្រកាស",
    "guidance": "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
    "website": "ទំព័រគេហទំព័រ",
}

# ----------------------------------------------------------------------------
# Embedding model options
# ----------------------------------------------------------------------------
# Duplicated from install_models.py -- this project's scripts/
# are each standalone, self-contained files with no shared utils module.
# If you change one, change the other.
#
# Each model gets its own subfolder under vector_db/, named after its
# local_dir's own folder name (e.g. vector_db/BAAI-BGE-M3/), so all three
# can exist side by side and scripts/09_compare_embedding_models.py never has
# to re-encode the whole corpus to compare them -- it just loads whichever
# precomputed indices are already here.
#
# This script no longer prompts for a single choice -- it builds (or
# skips, if already built) ALL installed models in one run.
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "hf_id": "BAAI/bge-m3",
        "marker_dir": Path("models/retriever/BAAI-BGE-M3"),
        "trust_remote_code": False,
        "task_aware": False,
    },
    "2": {
        "label": "intfloat/multilingual-e5-large",
        "hf_id": "intfloat/multilingual-e5-large",
        "marker_dir": Path("models/retriever/multilingual-e5-large"),
        "trust_remote_code": False,
        "task_aware": False,
        # e5 models require a short task prefix prepended to the raw text
        # (not a kwarg to .encode()): "passage: " for corpus documents and
        # "query: " for live questions. This is distinct from jina's LoRA
        # adapter switching -- it's a pure text transformation, handled
        # explicitly in encode_documents() below. This script only embeds the
        # corpus, so only doc_prefix is applied here; query_prefix is kept as
        # the record of the query-side convention, applied by whoever encodes
        # questions (scripts/09, app.py).
        "doc_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "marker_dir": Path("models/retriever/MiniLM-paraphrase-multilingual-L12-v2"),
        "trust_remote_code": False,
        "task_aware": False,
    },
}



def is_model_installed(model_config):
    """True if this embedding model has been installed by
    install_models.py. All models now use INSTALLED.json as
    their marker -- weights live in models/retriever/.cache/ (managed
    by huggingface_hub), not in the marker_dir itself."""
    return (model_config["marker_dir"] / "INSTALLED.json").exists()


def encode_documents(model, model_config, texts, **kwargs):
    """Encodes corpus chunks as PASSAGES. For e5-style models (multilingual-e5-large),
    this prepends a 'passage: ' prefix to every chunk -- e5 was trained with these
    literal prefixes and retrieval quality degrades noticeably without them."""
    if model_config.get("doc_prefix"):
        texts = [model_config["doc_prefix"] + t for t in texts]
    return model.encode(texts, **kwargs)


# ----------------------------------------------------------------------------
# Load and prepare chunks ONCE -- this part is identical regardless of
# which model(s) end up embedding the text, so it happens outside the
# per-model loop below rather than being redone three times.
# ----------------------------------------------------------------------------

chunk_files = sorted(input_root.rglob("*.json"))

all_chunks = []
load_start = time.time()

print("Loading chunk files...")

for chunk_path in chunk_files:
    print(f"Loading: {chunk_path.relative_to(input_root)}")

    try:
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    except Exception as e:
        print(f"Skipping broken chunk file: {chunk_path.name}")
        print(e)
        continue

print(f"\nTotal chunks loaded: {len(all_chunks)}")

embedding_chunks = []

for chunk in all_chunks:

    metadata = chunk.get("metadata", {}).copy()
    document_type = metadata.get("document_type")

    if document_type:
        metadata["document_type_kh"] = DOCUMENT_TYPE_KH.get(document_type)

    text = chunk.get("text") or chunk.get("content") or ""
    text = text.strip()

    if not text:
        continue

    embedding_parts = []

    if metadata.get("document_name"):
        embedding_parts.append(metadata["document_name"])

    if metadata.get("document_type_kh"):
        embedding_parts.append(metadata["document_type_kh"])

    if metadata.get("parent_h1"):
        embedding_parts.append(metadata["parent_h1"])

    if metadata.get("chapter"):
        embedding_parts.append(metadata["chapter"])

    if metadata.get("chapter_title"):
        embedding_parts.append(metadata["chapter_title"])

    if metadata.get("section"):
        embedding_parts.append(metadata["section"])

    if metadata.get("section_title"):
        embedding_parts.append(metadata["section_title"])

    if metadata.get("legal_unit"):
        embedding_parts.append(metadata["legal_unit"])

    if metadata.get("heading_title"):
        embedding_parts.append(metadata["heading_title"])

    if metadata.get("source_url"):
        embedding_parts.append(metadata["source_url"])

    embedding_parts.append(text)

    embedding_text = "\n".join(embedding_parts)
    if MAX_EMBEDDING_CHARS > 0 and len(embedding_text) > MAX_EMBEDDING_CHARS:
        embedding_text = embedding_text[:MAX_EMBEDDING_CHARS]

    embedding_chunks.append(
        {
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "embedding_text": embedding_text,
            "metadata": metadata,
        }
    )

all_chunks = embedding_chunks
texts = [chunk["embedding_text"] for chunk in all_chunks]

if not texts:
    raise ValueError("No chunks found.")

document_count = len(set(chunk["metadata"]["document_id"] for chunk in all_chunks))
chunk_lengths = [len(t) for t in texts]

print("\n========== DATASET INFO ==========")
print(f"Total Documents     : {document_count}")
print(f"Total Chunks        : {len(texts)}")
print(f"Largest Chunk Size  : {max(chunk_lengths)} chars")
print(f"Smallest Chunk Size : {min(chunk_lengths)} chars")
print(f"Average Chunk Size  : {sum(chunk_lengths) / len(texts):.0f} chars")

chunk_prep_time = time.time() - load_start

# ----------------------------------------------------------------------------
# Build (or skip) each model's index
# ----------------------------------------------------------------------------

overall_start = time.time()
built_count = 0
skipped_count = 0
unavailable_count = 0

for key, model_config in EMBEDDING_MODELS.items():
    label = model_config["label"]
    model_dir_name = model_config["marker_dir"].name
    subfolder = output_root / model_dir_name

    faiss_output = subfolder / "khmer_insurance.index"
    metadata_output = subfolder / "khmer_insurance_metadata.json"
    model_info_output = subfolder / "embedding_model_info.json"

    print(f"\n{'=' * 70}")
    print(f"MODEL: {label}")
    print(f"{'=' * 70}")

    existing_outputs = [faiss_output, metadata_output, model_info_output]
    newest_chunk_mtime = max(path.stat().st_mtime for path in chunk_files)
    outputs_current = (
        all(path.exists() for path in existing_outputs)
        and min(path.stat().st_mtime for path in existing_outputs) >= newest_chunk_mtime
    )

    if outputs_current:
        print(f"Already current -- skipping. ({subfolder}/)")
        skipped_count += 1
        continue

    if not is_model_installed(model_config):
        print(
            f"{label} not installed -- run: python install_models.py\n"
            "Run: python install_models.py"
        )
        unavailable_count += 1
        continue

    subfolder.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")

    # All three models use plain SentenceTransformer loaded by Hub ID.
    # Weights are served from models/retriever/.cache/ (already downloaded
    # during install_models.py) -- no network hit after first install.
    kwargs = {
        "trust_remote_code": bool(model_config["trust_remote_code"]),
        "local_files_only": True,
    }
    model = SentenceTransformer(model_config["hf_id"], **kwargs)

    print(f"Device              : {model.device}")

    print("Generating embeddings...")
    encode_start = time.time()

    if SEQUENTIAL_EMBEDDING:
        encoded = []
        for index, text in enumerate(texts, start=1):
            if index == 1 or index % 100 == 0 or index == len(texts):
                print(f"Encoding chunk {index}/{len(texts)}")
            vector = encode_documents(
                model,
                model_config,
                [text],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            encoded.append(np.asarray(vector, dtype="float32")[0])
        embeddings = np.vstack(encoded)
    else:
        embeddings = encode_documents(
            model,
            model_config,
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    encode_time = time.time() - encode_start
    print(f"Encode time         : {encode_time:.2f} sec")

    embeddings = np.asarray(embeddings, dtype="float32")
    embedding_dimension = embeddings.shape[1]

    import faiss

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    faiss.write_index(index, str(faiss_output))

    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(model_info_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": label,
                "hf_id": model_config["hf_id"],
                "hf_cache_dir": str(_HF_CACHE_DIR),
                "trust_remote_code": model_config["trust_remote_code"],
                "task_aware": model_config["task_aware"],
                "native_hf": model_config.get("native_hf", False),
                "embedding_dimension": int(embedding_dimension),
                "build_encode_time_sec": round(encode_time, 4),
                "chunk_count": len(all_chunks),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metadata_size_mb = metadata_output.stat().st_size / (1024 * 1024)
    faiss_size_mb = faiss_output.stat().st_size / (1024 * 1024)

    print(f"\nEmbedding Dimension : {embedding_dimension}")
    print(f"Vector Count        : {index.ntotal}")
    print(f"FAISS File Size     : {faiss_size_mb:.2f} MB")
    print(f"Metadata File Size  : {metadata_size_mb:.2f} MB")
    print(f"Saved to            : {subfolder}/")

    built_count += 1

    # Free before the next model -- multiple multilingual transformer
    # models in memory at once is unnecessary pressure on a machine that
    # may also be running llama-server.
    del model

    # Explicitly flush MPS (Apple Silicon GPU) memory. Without this,
    # Metal's lazy deallocation means the deleted model's GPU allocation
    # can linger until the next MPS sync point -- and if the next model
    # (e.g. jina-v3-hf loading on CPU) touches any MPS code path while
    # that allocation is still pending, it can trigger a segfault rather
    # than a clean OOM error. gc.collect() + mps.empty_cache() forces
    # immediate reclamation, making the GPU memory actually available
    # before the next model starts loading.
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

elapsed_time = time.time() - overall_start

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"Chunk preparation time : {chunk_prep_time:.2f} sec (done once, shared across all models)")
print(f"Models built           : {built_count}")
print(f"Models skipped (existing): {skipped_count}")
print(f"Models unavailable      : {unavailable_count}")
print(f"Total build runtime     : {elapsed_time:.2f} sec")

if built_count == 0 and skipped_count == 0:
    raise SystemExit(
        "\nNo embedding models are installed. Run: python install_models.py"
    )

print("\nDone.")
