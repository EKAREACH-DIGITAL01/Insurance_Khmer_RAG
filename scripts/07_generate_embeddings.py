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
    "law"       : "ច្បាប់",
    "prakas"    : "ប្រកាស",
    "guidance"  : "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
}

# FAISS vector index used for similarity search retrieval
faiss_output = output_root / "khmer_insurance.index"
# Metadata used to map retrieved vectors back to chunks and citations
metadata_output = output_root / "khmer_insurance_metadata.json"

# Skip if file already exists
if faiss_output.exists() and metadata_output.exists():

    print("\n========== EXISTING VECTOR DATABASE ==========")
    print(f"FAISS Index : {faiss_output}")
    print(f"Metadata    : {metadata_output}")

    raise SystemExit(
        "\nEmbeddings already exist.\n"
        "Delete the existing files and run the script again if you want to regenerate them."
    )

# BGE-M3 provides strong multilingual retrieval performance, including Khmer
model = SentenceTransformer("BAAI/bge-m3")

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
embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

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

elapsed_time = time.time() - start_time

metadata_size_mb = metadata_output.stat().st_size / (1024 * 1024)
faiss_size_mb = faiss_output.stat().st_size / (1024 * 1024)
embedding_size_mb = embeddings.nbytes / (1024 * 1024)

print("\n========== EMBEDDING INFO ==========")
print(f"Model               : BAAI/bge-m3")
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

print("\nEmbeddings and FAISS index created successfully.")