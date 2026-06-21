import json
import faiss
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

input_root = Path("data/07_chunked_documents")
output_root = Path("vector_db")

output_root.mkdir(parents=True, exist_ok=True)

DOCUMENT_TYPE_KH = {
    "law": "ច្បាប់",
    "prakas": "ប្រកាស",
    "guidance": "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
}

# ----------------------------------------------------------------------------
# Embedding model options
# ----------------------------------------------------------------------------
# Duplicated from scripts/13_install_models.py -- this project's scripts/
# are each standalone, self-contained files with no shared utils module.
# If you change one, change the other.
#
# Each model gets its own subfolder under vector_db/, named after its
# local_dir's own folder name (e.g. vector_db/BAAI-BGE-M3/), so all three
# can exist side by side and scripts/09_compare_embeddings.py never has
# to re-encode the whole corpus to compare them -- it just loads whichever
# precomputed indices are already here.
#
# This script no longer prompts for a single choice -- it builds (or
# skips, if already built) ALL installed models in one run.
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "hf_id": "BAAI/bge-m3",
        "local_dir": Path("models/retriever/BAAI-BGE-M3"),
        "trust_remote_code": False,
        "task_aware": False,
    },
    "2": {
        "label": "jinaai/jina-embeddings-v3",
        "hf_id": "jinaai/jina-embeddings-v3",
        "local_dir": Path("models/retriever/jinaai-jina-embeddings-v3"),
        "trust_remote_code": True,
        "task_aware": True,
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "local_dir": Path("models/retriever/MiniLM-paraphrase-multilingual-L12-v2"),
        "trust_remote_code": False,
        "task_aware": False,
    },
}


def encode_documents(model, model_config, texts, **kwargs):
    """Encodes corpus chunks as PASSAGES, not queries -- matters only for
    task-aware models like jina-embeddings-v3, which use a different LoRA
    adapter depending on which side of retrieval the text is on."""
    if model_config.get("task_aware"):
        kwargs["task"] = "retrieval.passage"
        kwargs["prompt_name"] = "retrieval.passage"
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

    embedding_parts.append(text)

    embedding_text = "\n".join(embedding_parts)

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
    model_dir_name = model_config["local_dir"].name
    subfolder = output_root / model_dir_name

    faiss_output = subfolder / "khmer_insurance.index"
    metadata_output = subfolder / "khmer_insurance_metadata.json"
    model_info_output = subfolder / "embedding_model_info.json"

    print(f"\n{'=' * 70}")
    print(f"MODEL: {label}")
    print(f"{'=' * 70}")

    if faiss_output.exists() and metadata_output.exists():
        print(f"Already built -- skipping. ({subfolder}/)")
        skipped_count += 1
        continue

    if not (model_config["local_dir"] / "modules.json").exists():
        print(
            f"{label} is not installed at {model_config['local_dir']} -- skipping.\n"
            "Run: python scripts/13_install_models.py"
        )
        unavailable_count += 1
        continue

    subfolder.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    model_kwargs = (
        {"trust_remote_code": True} if model_config["trust_remote_code"] else {}
    )
    model = SentenceTransformer(str(model_config["local_dir"]), **model_kwargs)
    print(f"Device              : {model.device}")

    print("Generating embeddings...")
    encode_start = time.time()

    embeddings = encode_documents(
        model,
        model_config,
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    encode_time = time.time() - encode_start
    print(f"Encode time         : {encode_time:.2f} sec")

    embeddings = np.asarray(embeddings, dtype="float32")
    embedding_dimension = embeddings.shape[1]

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
                "local_dir": str(model_config["local_dir"]),
                "trust_remote_code": model_config["trust_remote_code"],
                "task_aware": model_config["task_aware"],
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
        "\nNo embedding models are installed. Run: python scripts/13_install_models.py"
    )

print("\nDone.")