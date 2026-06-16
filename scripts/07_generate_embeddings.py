import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# BGE-M3 provides strong multilingual retrieval performance, including Khmer
model = SentenceTransformer("BAAI/bge-m3")

input_root = Path("data/07_chunked_documents")
output_root = Path("vector_db")

output_root.mkdir(parents=True, exist_ok=True)

chunk_files = sorted(input_root.rglob("*_chunks.json"))

all_chunks = []
start_time = time.time()

print("Loading chunk files...")


for chunk_path in chunk_files:

    print(f"Loading: {chunk_path.relative_to(input_root)}")

    try:

        with open(chunk_path, "r", encoding="utf-8") as f:

            chunks = json.load(f)

            all_chunks.extend(chunks)

    except Exception as e:

        print(f"Skipping broken chunk file: " f"{chunk_path.name}")

        print(e)

        continue


print(f"\nTotal chunks loaded: " f"{len(all_chunks)}")

# Build embedding text by combining metadata context with chunk content
embedding_chunks = []

for chunk in all_chunks:

    metadata = chunk.get("metadata", {})

    text = chunk.get("text") or chunk.get("content") or ""

    text = text.strip()

    if not text:
        continue

    # Embed structural metadata together with content to improve retrieval recall
    embedding_parts = []

    if metadata.get("document_name"):
        embedding_parts.append(metadata["document_name"])

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

print("\nGenerating embeddings...")

if not texts:
    raise ValueError("No chunks found.")

embeddings = model.encode(
    texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

dimension = embeddings.shape[1]

# Use cosine similarity via normalized vectors + inner product index
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"\nFAISS index size: {index.ntotal}")


faiss_output = output_root / "khmer_insurance.index"

faiss.write_index(index, str(faiss_output))

# Preserve original chunk metadata for retrieval and citation generation
metadata_output = output_root / "khmer_insurance_metadata.json"

with open(metadata_output, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Chunk Files Loaded : {len(chunk_files)}")
print(f"Total Chunks       : {len(all_chunks)}")
print(f"Embedding Dimension: {dimension}")
print(f"Vector Count       : {index.ntotal}")
print(f"Total Runtime      : {elapsed_time:.2f} seconds")
print("\nEmbeddings and FAISS " "index created successfully.")
