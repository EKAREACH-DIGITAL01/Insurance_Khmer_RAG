import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# LOAD EMBEDDING MODEL
model = SentenceTransformer("BAAI/bge-m3")

# ROOT CHUNK FOLDER
input_root = Path("data/07_chunked_documents")

# OUTPUT VECTOR DB FOLDER
output_root = Path("vector_db")

output_root.mkdir(parents=True, exist_ok=True)

# FIND ALL CHUNK FILES RECURSIVELY
chunk_files = sorted(input_root.rglob("*_chunks.json"))

# STORE ALL CHUNKS
all_chunks = []

# TOTAL RUNTIME
start_time = time.time()

print("Loading chunk files...")


# LOAD ALL CHUNK FILES
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

# TOTAL CHUNKS
print(f"\nTotal chunks loaded: " f"{len(all_chunks)}")

# NORMALIZE CHUNK FORMAT
normalized_chunks = []

for chunk in all_chunks:

    metadata = chunk.get("metadata", {})

    text = chunk.get("text") or chunk.get("content") or ""

    text = text.strip()

    if not text:
        continue

    normalized_chunk = {
        "chunk_id": chunk["chunk_id"],
        "text": text,
        "metadata": metadata,
    }

    normalized_chunks.append(normalized_chunk)

all_chunks = normalized_chunks

# EXTRACT TEXT ONLY
texts = [chunk["text"] for chunk in all_chunks]

print("\nGenerating embeddings...")

# GENERATE EMBEDDINGS
embeddings = model.encode(
    texts, batch_size=8, show_progress_bar=True, normalize_embeddings=True
)

# CONVERT TO NUMPY FLOAT32
embeddings = np.array(embeddings).astype("float32")

# CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# ADD EMBEDDINGS TO INDEX
index.add(embeddings)

print(f"\nFAISS index size: {index.ntotal}")

# SAVE VECTOR DATABASE
faiss_output = output_root / "khmer_insurance.index"

faiss.write_index(index, str(faiss_output))

# SAVE METADATA
metadata_output = output_root / "khmer_insurance_metadata.json"

with open(metadata_output, "w", encoding="utf-8") as f:

    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# TOTAL RUNTIME
elapsed_time = time.time() - start_time


print("\n========== SUMMARY ==========")
print(f"Chunk Files Loaded : " f"{len(chunk_files)}")
print(f"Total Chunks       : " f"{len(all_chunks)}")
print(f"Embedding Dimension: " f"{dimension}")
print(f"Vector Count       : " f"{index.ntotal}")
print(f"Total Runtime      : " f"{elapsed_time:.2f} seconds")
print("\nEmbeddings and FAISS " "index created successfully.")
