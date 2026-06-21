import json
import re
import time
import faiss
import numpy as np
from pathlib import Path
from tabulate import tabulate
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/08_evaluation")

# Evaluate retrieval performance across multiple top-K cutoffs
EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

# ----------------------------------------------------------------------------
# Hybrid retrieval (BM25 + FAISS, Reciprocal Rank Fusion)
# ----------------------------------------------------------------------------
# NOTE: khmer_aware_tokenize, RRF_K, and NGRAM_SIZE are intentionally
# duplicated from app.py rather than imported. This project's scripts/ and
# app.py are each standalone, self-contained files with no shared utils
# module (consistent with the rest of this pipeline -- see e.g.
# DOCUMENT_TYPE_KH being redefined independently in multiple scripts), so
# this follows that existing convention. If this tokenizer changes in
# app.py, it must be changed here too -- there is no single source of
# truth enforcing that today.
RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al. 2009)
NGRAM_SIZE = 3  # character n-gram size for Khmer-script BM25 tokens

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")


def khmer_aware_tokenize(text):
    """BM25 tokenizer for mixed Khmer/Latin/numeric text -- see app.py's
    khmer_aware_tokenize for the full rationale. Character n-grams for
    Khmer-script tokens (no word segmenter available), whole tokens kept
    for Latin/numeric runs (company names, article numbers, dates)."""
    tokens = []
    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            chars = list(word)
            if len(chars) < NGRAM_SIZE:
                tokens.append(word)
            else:
                for i in range(len(chars) - NGRAM_SIZE + 1):
                    tokens.append("".join(chars[i : i + NGRAM_SIZE]))
        else:
            tokens.append(word.lower())
    return tokens


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Fuses multiple ranked lists of result dicts (each dict must have a
    'chunk_id' and 'document_id' key) into one ranking via Reciprocal Rank
    Fusion. See app.py's reciprocal_rank_fusion docstring for why RRF
    (rank-based) is used instead of blending raw scores (FAISS cosine and
    BM25 scores are on incomparable scales)."""
    fused_scores = {}
    lookup = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            lookup.setdefault(chunk_id, item)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [
        {
            "chunk_id": cid,
            "document_id": lookup[cid]["document_id"],
            "score": fused_scores[cid],
        }
        for cid in ranked_ids
    ]


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

print("Building BM25 index...")
bm25_corpus_tokens = [
    khmer_aware_tokenize(chunk.get("embedding_text") or chunk["text"])
    for chunk in chunks
]
bm25_index = BM25Okapi(bm25_corpus_tokens)

# Fast lookup table used to map chunk IDs back to metadata
chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
metadata_chunk_ids = set(chunk_lookup.keys())

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")


# Run the production DENSE retrieval pipeline exactly as the live app does
# for the safety gates: question -> BGE-M3 embedding -> FAISS similarity
# search. This is the pre-hybrid baseline.
def retrieve_dense(question):
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


# Lexical-only retrieval via BM25, used both standalone (for inspection)
# and as one half of the hybrid fusion below.
def retrieve_bm25(question):
    tokenized_query = khmer_aware_tokenize(question)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:MAX_K]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["metadata"].get("document_id"),
                "score": float(scores[idx]),
            }
        )

    return results


# The actual hybrid pipeline the live app uses: dense + lexical, fused via
# Reciprocal Rank Fusion, truncated back to MAX_K candidates.
def retrieve_hybrid(question):
    dense_results = retrieve_dense(question)
    bm25_results = retrieve_bm25(question)
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    return fused[:MAX_K]


def evaluate_retrieval(retrieve_fn):
    """Runs the full gold-question evaluation using the given retrieve_fn
    (retrieve_dense or retrieve_hybrid) and returns the same table data
    structures the original single-mode version of this script produced,
    so both modes can be run back-to-back and printed for a direct
    before/after comparison."""
    summary_table = []
    hit_table = []
    recall_table = []
    mrr_table = []
    file_hit_table = []

    for gold_file in gold_files:
        category = gold_file.parent.name

        with open(gold_file, "r", encoding="utf-8") as f:
            gold_questions = json.load(f)

        if not gold_questions:
            raise ValueError(f"{gold_file} is empty.")

        missing_chunk_ids = set()

        for item in gold_questions:
            for chunk_id in item["expected_chunks"]:
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
            expected_chunks = set(item["expected_chunks"])

            if not expected_chunks:
                print("\nSkipping question with no expected_chunks:")
                print(question)
                continue

            relevant_document_ids = set()

            for chunk_id in expected_chunks:
                chunk = chunk_lookup.get(chunk_id)

                if chunk:
                    document_id = chunk["metadata"].get("document_id")

                    if document_id:
                        relevant_document_ids.add(document_id)

            if not relevant_document_ids:
                missing_questions += 1
                continue

            valid_questions += 1

            retrieved_results = retrieve_fn(question)
            retrieved_chunk_ids = [result["chunk_id"] for result in retrieved_results]
            retrieved_document_ids = [
                result["document_id"] for result in retrieved_results
            ]

            for k in EVAL_KS:
                top_k_chunks = retrieved_chunk_ids[:k]
                top_k_documents = retrieved_document_ids[:k]

                found_chunks = set(top_k_chunks).intersection(expected_chunks)

                if found_chunks:
                    hit_sums[k] += 1

                recall_sums[k] += len(found_chunks) / len(expected_chunks)

                reciprocal_rank = 0

                for rank, chunk_id in enumerate(top_k_chunks, start=1):
                    if chunk_id in expected_chunks:
                        reciprocal_rank = 1 / rank
                        break

                mrr_sums[k] += reciprocal_rank

                found_documents = set(top_k_documents).intersection(
                    relevant_document_ids
                )

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

    return {
        "summary": summary_table,
        "hit": hit_table,
        "recall": recall_table,
        "mrr": mrr_table,
        "file_hit": file_hit_table,
    }


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


def print_mode_report(mode_label, tables):
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}RETRIEVAL MODE: {mode_label}{RESET}")
    print(banner)

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}HIT RATE @ K{RESET} - At least one relevant chunk was retrieved.")
    print(SEPARATOR)
    print(
        tabulate(
            tables["hit"],
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
            tables["recall"],
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
            tables["mrr"],
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
            tables["file_hit"],
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
            tables["summary"],
            headers=summary_headers,
            tablefmt="grid",
            colalign=summary_colalign,
        )
    )


def build_comparison_rows(dense_tables, hybrid_tables):
    """Computes the headline before/after numbers -- Hit@5 and Recall@5 per
    category for both modes plus the delta -- as plain row data, reused by
    both the stdout printer and the saved report writer. These two cutoffs
    are chosen because CONTEXT_K=5 in app.py is what actually determines
    how many chunks reach the LLM prompt -- @5 is the number that matters
    for real answer quality, not just for the report."""
    rows = []
    dense_hit_by_cat = {row[0]: row[5] for row in dense_tables["hit"]}  # @5 is index 5
    hybrid_hit_by_cat = {row[0]: row[5] for row in hybrid_tables["hit"]}
    dense_recall_by_cat = {row[0]: row[5] for row in dense_tables["recall"]}
    hybrid_recall_by_cat = {row[0]: row[5] for row in hybrid_tables["recall"]}

    for category in dense_hit_by_cat:
        dense_hit = dense_hit_by_cat[category]
        hybrid_hit = hybrid_hit_by_cat[category]
        dense_recall = dense_recall_by_cat[category]
        hybrid_recall = hybrid_recall_by_cat[category]
        rows.append(
            [
                category,
                f"{dense_hit:.4f}",
                f"{hybrid_hit:.4f}",
                f"{hybrid_hit - dense_hit:+.4f}",
                f"{dense_recall:.4f}",
                f"{hybrid_recall:.4f}",
                f"{hybrid_recall - dense_recall:+.4f}",
            ]
        )

    headers = [
        "Category",
        "Hit@5 (dense)",
        "Hit@5 (hybrid)",
        "Δ Hit@5",
        "Recall@5 (dense)",
        "Recall@5 (hybrid)",
        "Δ Recall@5",
    ]
    return rows, headers


def print_comparison_summary(dense_tables, hybrid_tables):
    rows, headers = build_comparison_rows(dense_tables, hybrid_tables)
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}BEFORE / AFTER -- DENSE-ONLY vs. HYBRID (FAISS + BM25, RRF){RESET}")
    print(banner)
    print(tabulate(rows, headers=headers, tablefmt="grid", colalign=("center",) * 7))


REPORTS_DIR = Path("reports/retrieval")


def write_reports(dense_tables, hybrid_tables, comparison_rows, comparison_headers):
    """Saves both a raw JSON (full numeric data, re-analyzable later --
    e.g. for the technical report's tables/charts) and a human-readable
    Markdown report (same content as the stdout output, real Markdown
    tables via tabulate's "github" format rather than the box-drawing
    "grid" format used for the terminal) into reports/retrieval/. Printing
    to stdout during a run is useful in the moment; these files are what
    actually persist as a deliverable afterward."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "dense": dense_tables,
        "hybrid": hybrid_tables,
        "comparison": {
            "headers": comparison_headers,
            "rows": comparison_rows,
        },
        "eval_ks": EVAL_KS,
        "metric_headers": metric_headers,
    }
    json_path = REPORTS_DIR / "retrieval_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    def md_table(title, rows, headers):
        return f"### {title}\n\n" + tabulate(
            rows, headers=headers, tablefmt="github", floatfmt=".4f"
        ) + "\n\n"

    md_parts = ["# Retrieval Evaluation Report\n\n"]
    for mode_label, tables in [
        ("Dense-Only (BGE-M3)", dense_tables),
        ("Hybrid (BGE-M3 + BM25, RRF)", hybrid_tables),
    ]:
        md_parts.append(f"## {mode_label}\n\n")
        md_parts.append(md_table("Hit Rate @ K", tables["hit"], metric_headers))
        md_parts.append(md_table("Recall @ K", tables["recall"], metric_headers))
        md_parts.append(md_table("MRR @ K", tables["mrr"], metric_headers))
        md_parts.append(
            md_table("File Hit Rate @ K", tables["file_hit"], metric_headers)
        )
        md_parts.append(
            md_table("Dataset Summary", tables["summary"], summary_headers)
        )

    md_parts.append("## Before / After -- Dense-Only vs. Hybrid\n\n")
    md_parts.append(
        tabulate(comparison_rows, headers=comparison_headers, tablefmt="github")
    )
    md_parts.append("\n")

    md_path = REPORTS_DIR / "retrieval_eval.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


start_time = time.time()

print("\nRunning DENSE-ONLY (BGE-M3) evaluation...")
dense_tables = evaluate_retrieval(retrieve_dense)

print("Running HYBRID (BGE-M3 + BM25, RRF) evaluation...")
hybrid_tables = evaluate_retrieval(retrieve_hybrid)

print_mode_report("DENSE-ONLY (BGE-M3)", dense_tables)
print_mode_report("HYBRID (BGE-M3 + BM25, RRF)", hybrid_tables)
print_comparison_summary(dense_tables, hybrid_tables)

comparison_rows, comparison_headers = build_comparison_rows(dense_tables, hybrid_tables)
write_reports(dense_tables, hybrid_tables, comparison_rows, comparison_headers)

elapsed_time = time.time() - start_time

print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print("Retriever evaluation completed successfully (dense + hybrid).")