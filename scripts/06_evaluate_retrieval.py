import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

TOP_K = 5

INDEX_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
GOLD_PATH = "data/evaluation/gold_questions.json"

# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

# -----------------------------
# LOAD FAISS
# -----------------------------

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

# -----------------------------
# LOAD METADATA
# -----------------------------

print("Loading metadata...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

# -----------------------------
# LOAD GOLD QUESTIONS
# -----------------------------

print("Loading evaluation questions...")

with open(
    GOLD_PATH,
    "r",
    encoding="utf-8"
) as f:

    gold_questions = json.load(f)

# -----------------------------
# METRICS
# -----------------------------

total_questions = len(
    gold_questions
)

hit_count = 0
recall_count = 0

precision_sum = 0
mrr_sum = 0

# -----------------------------
# EVALUATION LOOP
# -----------------------------

for item in gold_questions:

    question = item["question"]

    relevant_chunk_ids = set(
        item["relevant_chunk_ids"]
    )

    # -----------------------------
    # EMBED QUESTION
    # -----------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # -----------------------------
    # SEARCH FAISS
    # -----------------------------

    distances, indices = index.search(
        query_embedding,
        k=TOP_K
    )

    retrieved_chunk_ids = []

    for i in indices[0]:

        chunk = chunks[i]

        retrieved_chunk_ids.append(
            chunk["chunk_id"]
        )

    retrieved_set = set(
        retrieved_chunk_ids
    )

    # -----------------------------
    # HIT RATE / RECALL@K
    # -----------------------------

    found_relevant = (
        retrieved_set.intersection(
            relevant_chunk_ids
        )
    )

    if found_relevant:

        hit_count += 1
        recall_count += 1

    # -----------------------------
    # PRECISION@K
    # -----------------------------

    relevant_retrieved_count = len(
        found_relevant
    )

    precision = (
        relevant_retrieved_count
        / TOP_K
    )

    precision_sum += precision

    # -----------------------------
    # MRR
    # -----------------------------

    reciprocal_rank = 0

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1
    ):

        if chunk_id in relevant_chunk_ids:

            reciprocal_rank = (
                1 / rank
            )

            break

    mrr_sum += reciprocal_rank

    # -----------------------------
    # PRINT RESULT
    # -----------------------------

    print("\n--------------------------------")
    print("Question:")
    print(question)

    print("\nExpected Chunk:")
    print(list(relevant_chunk_ids))

    print("\nRetrieved Chunks:")
    print(retrieved_chunk_ids)

    print("\nFound:")
    print(bool(found_relevant))

# -----------------------------
# FINAL METRICS
# -----------------------------

hit_rate = (
    hit_count
    / total_questions
)

recall_at_k = (
    recall_count
    / total_questions
)

precision_at_k = (
    precision_sum
    / total_questions
)

mrr = (
    mrr_sum
    / total_questions
)

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(
    f"Total Questions: "
    f"{total_questions}"
)

print(
    f"Hit Rate: "
    f"{hit_rate:.4f}"
)

print(
    f"Recall@{TOP_K}: "
    f"{recall_at_k:.4f}"
)

print(
    f"Precision@{TOP_K}: "
    f"{precision_at_k:.4f}"
)

print(
    f"MRR: "
    f"{mrr:.4f}"
)