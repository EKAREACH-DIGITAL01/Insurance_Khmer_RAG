import json
import time
import faiss
import numpy as np
from pathlib import Path
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/08_evaluation")

# Evaluate retrieval performance across multiple top-K cutoffs
EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

print("Loading embedding model...")
MODEL_PATH = Path("models/retriever/BAAI-BGE-M3")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "BAAI/BGE-M3 not found.\n" "Run: python scripts/09_install_models.py"
    )

embedding_model = SentenceTransformer(str(MODEL_PATH))

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Fast lookup table used to map chunk IDs back to metadata
chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
metadata_chunk_ids = set(chunk_lookup.keys())

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")

start_time = time.time()

# Store category-level results for final reporting
summary_table = []
hit_table = []
recall_table = []
mrr_table = []
file_hit_table = []


# Run the production retrieval pipeline:
# question -> BGE-M3 embedding -> FAISS similarity search
def retrieve(question):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        k=MAX_K,
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["metadata"].get("document_id"),
                "score": float(score),
            }
        )

    return results


for gold_file in gold_files:
    category = gold_file.parent.name

    with open(gold_file, "r", encoding="utf-8") as f:
        gold_questions = json.load(f)

    if not gold_questions:
        raise ValueError(f"{gold_file} is empty.")

    # Verify that all gold chunk IDs exist in the current metadata
    missing_chunk_ids = set()

    for item in gold_questions:
        for chunk_id in item["relevant_chunk_ids"]:
            if chunk_id not in metadata_chunk_ids:
                missing_chunk_ids.add(chunk_id)

    missing_questions = 0
    valid_questions = 0

    hit_sums = {k: 0 for k in EVAL_KS}
    recall_sums = {k: 0 for k in EVAL_KS}
    mrr_sums = {k: 0 for k in EVAL_KS}
    file_hit_sums = {k: 0 for k in EVAL_KS}

    for item in gold_questions:
        question = item["question"]
        relevant_chunk_ids = set(item["relevant_chunk_ids"])

        if not relevant_chunk_ids:
            print("\nSkipping question with no relevant_chunk_ids:")
            print(question)
            continue

        # Derive expected source documents from the gold chunk IDs
        relevant_document_ids = set()

        for chunk_id in relevant_chunk_ids:
            chunk = chunk_lookup.get(chunk_id)

            if chunk:
                document_id = chunk["metadata"].get("document_id")

                if document_id:
                    relevant_document_ids.add(document_id)

        if not relevant_document_ids:
            missing_questions += 1
            continue

        valid_questions += 1

        retrieved_results = retrieve(question)
        retrieved_chunk_ids = [result["chunk_id"] for result in retrieved_results]
        retrieved_document_ids = [result["document_id"] for result in retrieved_results]

        # Compute retrieval metrics for every evaluation cutoff
        for k in EVAL_KS:

            top_k_chunks = retrieved_chunk_ids[:k]
            top_k_documents = retrieved_document_ids[:k]

            found_chunks = set(top_k_chunks).intersection(relevant_chunk_ids)

            # Hit Rate@K: At least one relevant chunk was retrieved.
            if found_chunks:
                hit_sums[k] += 1

            # Recall@K: Proportion of relevant chunks retrieved.
            recall_sums[k] += len(found_chunks) / len(relevant_chunk_ids)

            # MRR@K: Ranking position of the first relevant chunk.
            reciprocal_rank = 0

            for rank, chunk_id in enumerate(top_k_chunks, start=1):

                if chunk_id in relevant_chunk_ids:
                    reciprocal_rank = 1 / rank
                    break

            mrr_sums[k] += reciprocal_rank

            # File Hit Rate@K: At least one chunk from the correct document was retrieved.
            found_documents = set(top_k_documents).intersection(relevant_document_ids)

            if found_documents:
                file_hit_sums[k] += 1

    if valid_questions == 0:
        raise ValueError(f"No valid questions found in {gold_file}.")

    summary_table.append(
        [
            category,
            len(gold_questions),
            valid_questions,
            missing_questions,
            len(missing_chunk_ids),
            ", ".join(sorted(missing_chunk_ids)) if missing_chunk_ids else "None",
        ]
    )

    # Convert aggregated metric sums into category-level results
    hit_row = [category]
    recall_row = [category]
    mrr_row = [category]
    file_hit_row = [category]

    for k in EVAL_KS:

        hit_row.append(hit_sums[k] / valid_questions)
        recall_row.append(recall_sums[k] / valid_questions)
        mrr_row.append(mrr_sums[k] / valid_questions)
        file_hit_row.append(file_hit_sums[k] / valid_questions)

    hit_table.append(hit_row)
    recall_table.append(recall_row)
    mrr_table.append(mrr_row)
    file_hit_table.append(file_hit_row)

# Final retrieval evaluation report
metric_headers = ["Category"] + [f"@{k}" for k in EVAL_KS]
summary_headers = [
    "Category",
    "Gold Questions",
    "Valid Questions",
    "Skipped Questions",
    "Missing Chunks",
    "Missing Chunk IDs",
]

metric_colalign = ("center",) * len(metric_headers)
summary_colalign = ("center",) * len(summary_headers)

BOLD = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH

print(f"\n{SEPARATOR}")
print(f"\t\t\t{BOLD}HIT RATE @ K{RESET} - At least one relevant chunk was retrieved.")
print(SEPARATOR)
print(
    tabulate(
        hit_table,
        headers=metric_headers,
        tablefmt="grid",
        floatfmt=".4f",
        colalign=metric_colalign,
    )
)

print(f"\n{SEPARATOR}")
print(f"\t\t\t{BOLD}RECALL @ K{RESET} - Proportion of relevant chunks retrieved.")
print(SEPARATOR)
print(
    tabulate(
        recall_table,
        headers=metric_headers,
        tablefmt="grid",
        floatfmt=".4f",
        colalign=metric_colalign,
    )
)

print(f"\n{SEPARATOR}")
print(f"\t\t\t{BOLD}MRR @ K{RESET} - Ranking position of the first relevant chunk.")
print(SEPARATOR)
print(
    tabulate(
        mrr_table,
        headers=metric_headers,
        tablefmt="grid",
        floatfmt=".4f",
        colalign=metric_colalign,
    )
)

print(f"\n{SEPARATOR}")
print(
    f"\t\t{BOLD}FILE HIT RATE @ K{RESET} - At least one chunk from the correct document was retrieved."
)
print(SEPARATOR)
print(
    tabulate(
        file_hit_table,
        headers=metric_headers,
        tablefmt="grid",
        floatfmt=".4f",
        colalign=metric_colalign,
    )
)

print(f"\n{SEPARATOR}============")
print(f"\t\t\t\t\t\t{BOLD}DATASET SUMMARY{RESET}")
print(f"{SEPARATOR}============")
print(
    tabulate(
        summary_table,
        headers=summary_headers,
        tablefmt="grid",
        colalign=summary_colalign,
    )
)

elapsed_time = time.time() - start_time

print(f"Total Runtime : {elapsed_time:.2f} seconds")
print("Retriever evaluation completed successfully.")
