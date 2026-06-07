import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

TOP_K = 5

## Can be modified (300, 500, 700, etc.)
INDEX_PATH = "vector_db/khmer_insurance_300.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata_300.json"
GOLD_PATH = "data/evaluation/gold_questions_300.json"

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
# VALIDATION
# -----------------------------

total_questions = len(
    gold_questions
)

if total_questions == 0:

    raise ValueError(
        "gold_questions.json is empty."
    )

# -----------------------------
# METRICS
# -----------------------------

hit_count = 0

recall_sum = 0
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

    if not relevant_chunk_ids:

        print(
            "\nSkipping question with no "
            "relevant_chunk_ids:"
        )

        print(question)

        total_questions -= 1

        continue

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
    retrieved_results = []

    for idx, score in zip(
        indices[0],
        distances[0]
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        chunk_id = chunk["chunk_id"]

        retrieved_chunk_ids.append(
            chunk_id
        )

        retrieved_results.append(
            {
                "chunk_id": chunk_id,
                "score": float(score)
            }
        )

    retrieved_set = set(
        retrieved_chunk_ids
    )

    # -----------------------------
    # MATCH RELEVANT CHUNKS
    # -----------------------------

    found_relevant = (
        retrieved_set.intersection(
            relevant_chunk_ids
        )
    )

    # -----------------------------
    # HIT RATE
    # -----------------------------

    if found_relevant:

        hit_count += 1

    # -----------------------------
    # RECALL@K
    # -----------------------------
    # Recall@K = relevant retrieved / total relevant

    recall = (
        len(found_relevant)
        / len(relevant_chunk_ids)
    )

    recall_sum += recall

    # -----------------------------
    # PRECISION@K
    # -----------------------------
    # Precision@K = relevant retrieved / retrieved count

    retrieved_count = len(
        retrieved_chunk_ids
    )

    if retrieved_count == 0:

        precision = 0

    else:

        precision = (
            len(found_relevant)
            / retrieved_count
        )

    precision_sum += precision

    # -----------------------------
    # MRR
    # -----------------------------
    # MRR = 1 / rank of first relevant chunk

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

    print("\nExpected Chunks:")
    for chunk_id in sorted(
        relevant_chunk_ids
    ):

        print(f"- {chunk_id}")

    print("\nRetrieved Chunks:")
    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['chunk_id']} "
            f"(score={result['score']:.4f})"
        )

    print("\nFound:")
    print(bool(found_relevant))

    print(
        f"Recall@{TOP_K}: "
        f"{recall:.4f}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.4f}"
    )

    print(
        f"Reciprocal Rank: "
        f"{reciprocal_rank:.4f}"
    )

# -----------------------------
# FINAL METRICS
# -----------------------------

if total_questions == 0:

    raise ValueError(
        "No valid evaluation questions found."
    )

hit_rate = (
    hit_count
    / total_questions
)

recall_at_k = (
    recall_sum
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