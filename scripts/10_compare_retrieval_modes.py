import json
import os
import re
import time
import faiss
import numpy as np
from pathlib import Path
from tabulate import tabulate
from rank_bm25 import BM25Okapi

# Set HF_HOME before any HF import -- models live in retriever/.cache/,
# loaded by Hub ID. See install_models.py for the full explanation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

# Active embedding model -- must match ACTIVE_EMBEDDING_MODEL_DIR in app.py
EMBEDDING_HF_ID = "BAAI/bge-m3"
ACTIVE_MODEL_DIR = "BAAI-BGE-M3"

INDEX_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance.index"
METADATA_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/09_evaluation")
REPORTS_DIR = Path("reports/retrieval_mode_comparison")

EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

# ----------------------------------------------------------------------------
# Hybrid retrieval (BM25 + FAISS, Reciprocal Rank Fusion)
# ----------------------------------------------------------------------------
RRF_K = 60
NGRAM_SIZE = 3

KHMER_SCRIPT_PATTERN = re.compile(r"[ក-៿]")
# Mirrors app.py:69/83/87/92. These were missing here, so this script's BM25 was
# weaker than production's -- no article-reference boost, no digit-run
# extraction -- which understated the hybrid arm of the very comparison this
# script exists to measure. Kept byte-identical to app.py and scripts/13/16.
DIGIT_RUN_PATTERN = re.compile(r"[០១២៣៤៥៦៧៨៩0-9]+")
ARTICLE_REF_PATTERN = re.compile(r"(មាត្រា|ប្រការ)\s*([០១២៣៤៥៦៧៨៩0-9]+)")
ARTICLE_REF_BOOST = 20
_ARABIC_TO_KHMER = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


def to_khmer_digits(s):
    """Mirrors app.py:90 -- the corpus writes provision numbers in Khmer
    digits, so an Arabic-digit query has to be normalised to match."""
    return s.translate(_ARABIC_TO_KHMER)


def khmer_aware_tokenize(text):
    tokens = []
    for keyword, ref_num in ARTICLE_REF_PATTERN.findall(text):
        tokens.extend([keyword + to_khmer_digits(ref_num)] * ARTICLE_REF_BOOST)
    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            for digit_run in DIGIT_RUN_PATTERN.findall(word):
                tokens.append(digit_run)
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


# Check vector_db subfolder exists
if not Path(INDEX_PATH).exists():
    raise FileNotFoundError(
        f"FAISS index not found at {INDEX_PATH}.\n"
        "Run: python scripts/08_generate_embeddings.py"
    )

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_HF_ID, local_files_only=True)

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

chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
metadata_chunk_ids = set(chunk_lookup.keys())

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")


def retrieve_dense(question):
    query_embedding = embedding_model.encode([question], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, indices = index.search(query_embedding, k=MAX_K)
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


def retrieve_hybrid(question):
    dense_results = retrieve_dense(question)
    bm25_results = retrieve_bm25(question)
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    return fused[:MAX_K]


def evaluate_retrieval(retrieve_fn):
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
    """Prints the four metric tables for one retrieval mode.
    DATASET SUMMARY is intentionally excluded -- it is printed once after
    both modes via print_dataset_summary(), since the gold question counts
    are identical for dense and hybrid (they ran on the same dataset)."""
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}RETRIEVAL MODE: {mode_label}{RESET}")
    print(banner)

    print(f"\n{SEPARATOR}")
    print(
        f"\t\t\t{BOLD}HIT RATE @ K{RESET} - At least one relevant chunk was retrieved."
    )
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


def print_dataset_summary(tables):
    """Prints the DATASET SUMMARY once, after both modes' metric tables.
    Dense and hybrid ran on the same gold question set so the counts are
    identical -- printing it twice would just repeat the same rows."""
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}============")
    print(f"\t\t\t\t\t\t{BOLD}DATASET SUMMARY{RESET}")
    print(f"{banner}============")
    print(
        tabulate(
            tables["summary"],
            headers=summary_headers,
            tablefmt="grid",
            colalign=summary_colalign,
        )
    )


def macro_avg_at_5(tables, metric_key):
    """Macro-average of the @5 column across all categories.
    Row index 5 = @5 (index 0 is the category label, 1-10 are @1 through @10)."""
    vals = [row[5] for row in tables[metric_key]]
    return sum(vals) / len(vals) if vals else 0.0


def print_comparison_summary(dense_tables, hybrid_tables):
    """Single unified before/after comparison table.
    Rows = 4 metrics. Columns = Dense @5, Hybrid @5, Delta.
    Each value is the macro-average across all active gold-question categories."""
    metric_defs = [
        ("Hit Rate", "hit"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("File Hit Rate", "file_hit"),
    ]

    rows = []
    for display_name, metric_key in metric_defs:
        dense_val = macro_avg_at_5(dense_tables, metric_key)
        hybrid_val = macro_avg_at_5(hybrid_tables, metric_key)
        delta = hybrid_val - dense_val
        rows.append(
            [
                f"{display_name} @5",
                f"{dense_val:.4f}",
                f"{hybrid_val:.4f}",
                f"{delta:+.4f}",
            ]
        )

    headers = ["Metric", "Dense-Only @5", "Hybrid @5", "Δ (Hybrid − Dense)"]
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}BEFORE / AFTER -- DENSE-ONLY vs. HYBRID (FAISS + BM25, RRF){RESET}")
    print(
        f"{BOLD}(each value = macro-avg across active gold-question categories){RESET}"
    )
    print(banner)
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "center", "center", "center"),
        )
    )


def write_reports(dense_tables, hybrid_tables):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the unified comparison rows for JSON/markdown
    metric_defs = [
        ("Hit Rate", "hit"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("File Hit Rate", "file_hit"),
    ]
    comparison_rows = []
    for display_name, metric_key in metric_defs:
        dense_val = macro_avg_at_5(dense_tables, metric_key)
        hybrid_val = macro_avg_at_5(hybrid_tables, metric_key)
        delta = hybrid_val - dense_val
        comparison_rows.append(
            {
                "metric": f"{display_name} @5",
                "dense": round(dense_val, 4),
                "hybrid": round(hybrid_val, 4),
                "delta": round(delta, 4),
            }
        )

    json_payload = {
        "dense": dense_tables,
        "hybrid": hybrid_tables,
        "comparison": {
            "note": (
                "Each value is the macro-average of the @5 score across all "
                "active gold-question categories."
            ),
            "rows": comparison_rows,
        },
        "eval_ks": EVAL_KS,
        "metric_headers": metric_headers,
    }
    json_path = REPORTS_DIR / "retrieval_mode_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    def md_table(title, rows, headers):
        return (
            f"### {title}\n\n"
            + tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f")
            + "\n\n"
        )

    md_parts = ["# Retrieval Mode Comparison Report\n\n"]
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

    # Dataset summary once in the markdown report too
    md_parts.append("## Dataset Summary\n\n")
    md_parts.append(
        tabulate(dense_tables["summary"], headers=summary_headers, tablefmt="github")
    )
    md_parts.append("\n\n")

    # Unified before/after comparison table
    md_parts.append("## Before / After -- Dense-Only vs. Hybrid (Macro-Avg @5)\n\n")
    md_parts.append(
        "> Each value is the macro-average of the @5 score across "
        "active gold-question categories.\n\n"
    )
    comp_headers = ["Metric", "Dense-Only @5", "Hybrid @5", "Δ (Hybrid − Dense)"]
    comp_md_rows = [
        [r["metric"], f"{r['dense']:.4f}", f"{r['hybrid']:.4f}", f"{r['delta']:+.4f}"]
        for r in comparison_rows
    ]
    md_parts.append(tabulate(comp_md_rows, headers=comp_headers, tablefmt="github"))
    md_parts.append("\n")

    md_path = REPORTS_DIR / "retrieval_mode_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

start_time = time.time()

print("\nRunning DENSE-ONLY (BGE-M3) evaluation...")
dense_tables = evaluate_retrieval(retrieve_dense)

print("Running HYBRID (BGE-M3 + BM25, RRF) evaluation...")
hybrid_tables = evaluate_retrieval(retrieve_hybrid)

# Print per-mode metric tables (no DATASET SUMMARY inside each)
print_mode_report("DENSE-ONLY (BGE-M3)", dense_tables)
print_mode_report("HYBRID (BGE-M3 + BM25, RRF)", hybrid_tables)

# DATASET SUMMARY printed once, after both modes' metric tables
print_dataset_summary(dense_tables)

# Single unified before/after comparison table:
# rows = 4 metrics, columns = Dense @5 / Hybrid @5 / Delta,
# values = macro-average across all active gold-question categories
print_comparison_summary(dense_tables, hybrid_tables)

write_reports(dense_tables, hybrid_tables)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print("Retrieval mode comparison completed successfully (dense + hybrid).")
