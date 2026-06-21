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
# are each standalone, self-contained files with no shared utils module
# (consistent with the rest of this pipeline). If you change one, change
# the other.
#
# jina-embeddings-v3 needs `trust_remote_code=True` to load and a `task`
# argument at encode time (its task-specific behavior is implemented via
# LoRA adapters selected per-call, not a fixed architecture) -- BGE-M3
# needs neither. See encode_documents() below for how that's handled.
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

# FAISS vector index used for similarity search retrieval
faiss_output = output_root / "khmer_insurance.index"
# Metadata used to map retrieved vectors back to chunks and citations
metadata_output = output_root / "khmer_insurance_metadata.json"
# Records WHICH embedding model built this index -- the live app (and any
# evaluation script) must load the SAME model to embed queries, since
# embeddings from two different models are not compatible vector spaces
# even when the dimensions happen to match. Without this file, swapping
# which model scripts/07 used would silently break retrieval the next
# time the app starts with no error, just nonsense similarity scores.
model_info_output = output_root / "embedding_model_info.json"

# Skip if file already exists
if faiss_output.exists() and metadata_output.exists():

    print("\n========== EXISTING VECTOR DATABASE ==========")
    print(f"FAISS Index : {faiss_output}")
    print(f"Metadata    : {metadata_output}")

    raise SystemExit(
        "\nEmbeddings already exist.\n"
        "Delete the existing files and run the script again if you want to regenerate them."
    )

# MODEL SELECTION
print("""
        ========================================
        Embedding Model Selection
        ========================================

        Choose which model builds this vector database:

        1. BAAI/bge-m3                          (~568M params)
        2. jinaai/jina-embeddings-v3            (~572M params)
        3. paraphrase-multilingual-MiniLM-L12-v2 (~118M params, smaller/faster)

        Both are real, supported options -- not a "default vs.
        experimental" choice. See reports/embeddings/ (after running
        scripts/09_compare_embeddings.py) for a head-to-head comparison
        on this corpus before deciding.
        """)

embed_choice = input("Enter choice: ").strip()

if embed_choice not in EMBEDDING_MODELS:
    raise SystemExit("Invalid choice.")

selected = EMBEDDING_MODELS[embed_choice]

print(f"\nSelected embedding model: {selected['label']}")

print("Loading embedding model...")
MODEL_PATH = selected["local_dir"]

if not (MODEL_PATH / "modules.json").exists():
    raise FileNotFoundError(
        f"{selected['label']} not found at {MODEL_PATH}.\n"
        "Run: python scripts/13_install_models.py"
    )

model_kwargs = {"trust_remote_code": True} if selected["trust_remote_code"] else {}
model = SentenceTransformer(str(MODEL_PATH), **model_kwargs)


def encode_documents(model, model_config, texts, **kwargs):
    """Encodes corpus chunks as PASSAGES/DOCUMENTS, not queries -- matters
    only for task-aware models. jina-embeddings-v3 uses a different LoRA
    adapter for "this text will be searched against" (retrieval.passage)
    vs. "this text is a search query" (retrieval.query); using the wrong
    one doesn't error, it just silently produces worse retrieval. BGE-M3
    has no such distinction and ignores this entirely."""
    if model_config.get("task_aware"):
        kwargs["task"] = "retrieval.passage"
        kwargs["prompt_name"] = "retrieval.passage"
    return model.encode(texts, **kwargs)


# Find all JSONs recursively
chunk_files = sorted(input_root.rglob("*.json"))

all_chunks = []
start_time = time.time()

print("Loading chunk files...")

# Load all chunk files generated from the chunking pipeline
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

# Build embedding text by combining metadata context with chunk content
embedding_chunks = []

# Convert chunk structure into embedding-ready format
for chunk in all_chunks:

    # Get English document type translated into Khmer
    metadata = chunk.get("metadata", {}).copy()
    document_type = metadata.get("document_type")

    if document_type:
        metadata["document_type_kh"] = DOCUMENT_TYPE_KH.get(document_type)

    text = chunk.get("text") or chunk.get("content") or ""
    text = text.strip()

    if not text:
        continue

    # Include document, chapter, section, and heading context in the embedding text
    # so retrieval can match both content and document structure
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

    embedding_chunk = {
        "chunk_id": chunk["chunk_id"],
        "text": text,
        "embedding_text": embedding_text,
        "metadata": metadata,
    }

    embedding_chunks.append(embedding_chunk)

all_chunks = embedding_chunks

# Use enriched text for embedding generation
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
print(f"Device              : {model.device}")

print("\nGenerating embeddings...")
print("Starting encode...")
encode_start = time.time()

# Generate normalized embeddings for cosine similarity search
embeddings = encode_documents(
    model,
    selected,
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
)

encode_time = time.time() - encode_start

print(f"Encode time         : {encode_time:.2f} sec")
print("Encode finished.")

embeddings = np.asarray(embeddings, dtype="float32")

# Use cosine similarity via normalized vectors + inner product index
embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dimension)
index.add(embeddings)

# Save vector index to disk
faiss.write_index(index, str(faiss_output))

with open(metadata_output, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# Record which model built this index -- see model_info_output comment
# above for why this matters beyond just bookkeeping.
with open(model_info_output, "w", encoding="utf-8") as f:
    json.dump(
        {
            "label": selected["label"],
            "hf_id": selected["hf_id"],
            "local_dir": str(selected["local_dir"]),
            "trust_remote_code": selected["trust_remote_code"],
            "task_aware": selected["task_aware"],
            "embedding_dimension": int(embedding_dimension),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

elapsed_time = time.time() - start_time

metadata_size_mb = metadata_output.stat().st_size / (1024 * 1024)
faiss_size_mb = faiss_output.stat().st_size / (1024 * 1024)
embedding_size_mb = embeddings.nbytes / (1024 * 1024)

print("\n========== EMBEDDING INFO ==========")
print(f"Model               : {selected['label']} ({MODEL_PATH})")
print(f"Device              : {model.device}")
print(f"Embedding Shape     : {embeddings.shape}")
print(f"Embedding Dtype     : {embeddings.dtype}")
print(f"Embedding Dimension : {index.d}")
print(f"Min Value           : {embeddings.min():.6f}")
print(f"Max Value           : {embeddings.max():.6f}")
print(f"Mean Value          : {embeddings.mean():.6f}")
print(f"First Vector Norm   : {np.linalg.norm(embeddings[0]):.6f}")

print("\n=========== FAISS INFO =============")
print(f"Index Type          : {type(index).__name__}")
print(f"Vector Count        : {index.ntotal}")
print(f"Metadata File Size  : {metadata_size_mb:.2f} MB")
print(f"FAISS File Size     : {faiss_size_mb:.2f} MB")
print(f"Embedding RAM Size  : {embedding_size_mb:.2f} MB")

print("\n============= RUNTIME ==============")
print(f"Encode Time         : {encode_time:.2f} sec")
print(f"Total Runtime       : {elapsed_time:.2f} sec")

print(f"\nModel info saved    : {model_info_output}")
print("Embeddings and FAISS index created successfully.")