import json
import os
import time
import faiss
import numpy as np
from pathlib import Path
from tabulate import tabulate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Embedding Model Comparison Report (Internship Proposal §6.3, §12)
# ----------------------------------------------------------------------------
# "Students should research and compare at least two multilingual text
# embedding approaches, evaluating their performance on Khmer insurance
# queries using the gold Q&A pairs" -- and the named deliverable is "a
# short written report comparing at least two embedding approaches...
# with results and a justified recommendation."
#
# This is deliberately a DENSE-ONLY comparison -- no BM25, no RRF fusion,
# no hybrid retrieval. Mixing hybrid in here would make it impossible to
# tell whether a score difference came from the embedding model itself or
# from the lexical layer riding alongside it.
#
# IMPORTANT: this script no longer encodes the corpus itself. It loads
# the PRECOMPUTED index + metadata that scripts/08_generate_embeddings.py
# already built into vector_db/<model_dir>/ for each installed model --
# by far the most expensive step (encoding the whole corpus) happens
# once, offline, in that script, not every time this comparison runs.
# Only the gold questions (a few hundred short texts) get encoded fresh
# here, since those need to be embedded as QUERIES at evaluation time
# regardless. Run scripts/08_generate_embeddings.py first if a model's
# index doesn't exist yet -- this script will tell you exactly which one
# is missing rather than silently rebuilding it.
#
# ----------------------------------------------------------------------------
# NOTE ON MiniLM ALL-ZERO SCORES
# ----------------------------------------------------------------------------
# If paraphrase-multilingual-MiniLM-L12-v2 returns 0.0000 across every
# metric and every category (hit, recall, MRR, file hit), the almost
# certain cause is a chunk ID mismatch between its precomputed index and
# the gold questions dataset:
#
#   - Its vector_db/MiniLM-.../khmer_insurance_metadata.json was built
#     at a different time than the other two models' indices, from a
#     different version of data/07_chunked_documents/ (e.g. after a
#     re-chunking run that renumbered the chunks).
#   - The gold questions in data/09_evaluation/ reference chunk IDs from
#     a DIFFERENT run -- so every expected_chunk lookup returns None,
#     making every question "missing" and producing 0/0 = 0.0 across all
#     metrics.
#   - The File Hit Rate being non-zero (e.g. 0.12 for guidance, rising
#     to ~0.36 for laws at @10) confirms the model CAN find the right
#     *documents* -- it just can't match the right *chunks* because the
#     IDs in its metadata don't align with the gold set.
#
# Fix: delete vector_db/MiniLM-paraphrase-multilingual-L12-v2/ and
# re-run scripts/08_generate_embeddings.py to rebuild it fresh alongside
# the other two models from the same chunk files. Post-rebuild the scores
# will be genuinely low (MiniLM is a small model and not well-suited to
# Khmer legal retrieval) but numerically real, which is what the
# comparison report needs.
#
# Quick diagnostic to confirm the mismatch before rebuilding:
#   python - <<'EOF'
#   import json
#   mini = json.load(open("vector_db/MiniLM-paraphrase-multilingual-L12-v2/khmer_insurance_metadata.json"))
#   bge  = json.load(open("vector_db/BAAI-BGE-M3/khmer_insurance_metadata.json"))
#   print("MiniLM chunks:", len(mini), "| first ID:", mini[0]["chunk_id"])
#   print("BGE-M3 chunks:", len(bge),  "| first ID:", bge[0]["chunk_id"])
#   EOF
# If the chunk counts differ or first IDs don't match, the rebuild is needed.
# ----------------------------------------------------------------------------

EVALUATION_ROOT = Path("data/09_evaluation")
VECTOR_DB_ROOT = Path("vector_db")
REPORTS_DIR = Path("reports/embedding_model_comparison")

EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

# Duplicated from install_models.py / scripts/08_generate_embeddings.py
# -- this project's scripts/ are each standalone, self-contained files
# with no shared utils module. If this changes in one place, change it
# in all three.
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "hf_id": "BAAI/bge-m3",
        "marker_dir": Path("models/retriever/BAAI-BGE-M3"),
        "trust_remote_code": False,
        "task_aware": False,
    },
    "2": {
        "label": "intfloat/multilingual-e5-large",
        "hf_id": "intfloat/multilingual-e5-large",
        "marker_dir": Path("models/retriever/multilingual-e5-large"),
        "trust_remote_code": False,
        "task_aware": False,
        # e5 models require a short task prefix prepended to the raw text
        # (not a kwarg to .encode()): "passage: " for corpus documents and
        # "query: " for live questions. This is distinct from jina's LoRA
        # adapter switching -- it's a pure text transformation, handled
        # explicitly in encode_documents() and encode_queries() below.
        "doc_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "marker_dir": Path("models/retriever/MiniLM-paraphrase-multilingual-L12-v2"),
        "trust_remote_code": False,
        "task_aware": False,
    },
}


def is_model_installed(model_config):
    """All models now use INSTALLED.json as their marker."""
    return (model_config["marker_dir"] / "INSTALLED.json").exists()


BOLD = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH

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


def vector_db_subfolder(model_config):
    return VECTOR_DB_ROOT / model_config["marker_dir"].name


for key, model_config in EMBEDDING_MODELS.items():
    if not is_model_installed(model_config):
        raise FileNotFoundError(
            f"{model_config['label']} not installed -- run: python install_models.py"
        )
    subfolder = vector_db_subfolder(model_config)
    if not (
        (subfolder / "khmer_insurance.index").exists()
        and (subfolder / "khmer_insurance_metadata.json").exists()
    ):
        raise FileNotFoundError(
            f"No precomputed index for {model_config['label']} at {subfolder}/.\n"
            "Run: python scripts/08_generate_embeddings.py"
        )

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")


def encode_queries(model, model_config, texts, **kwargs):
    """Encodes search QUERIES. For e5-style models, prepends 'query: ' prefix."""
    if model_config.get("query_prefix"):
        texts = [model_config["query_prefix"] + t for t in texts]
    return model.encode(texts, **kwargs)


def load_precomputed(model_config):
    """Loads the FAISS index, chunk metadata, and build-time info that
    scripts/08_generate_embeddings.py already produced for this model --
    no encoding happens here."""
    subfolder = vector_db_subfolder(model_config)

    index = faiss.read_index(str(subfolder / "khmer_insurance.index"))

    with open(subfolder / "khmer_insurance_metadata.json", "r", encoding="utf-8") as f:
        model_chunks = json.load(f)

    build_info = {}
    info_path = subfolder / "embedding_model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            build_info = json.load(f)

    return index, model_chunks, build_info


def make_retrieve_fn(model, model_config, index, model_chunks):
    """Returns a retrieve(question) function closed over this specific
    model/index/chunks trio."""

    def retrieve(question):
        query_embedding = encode_queries(
            model, model_config, [question], normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        scores, indices = index.search(query_embedding, k=MAX_K)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(model_chunks):
                continue
            chunk = model_chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["metadata"].get("document_id"),
                    "score": float(score),
                }
            )
        return results

    return retrieve


def evaluate_retrieval(retrieve_fn, chunk_lookup, metadata_chunk_ids):
    """Same gold-question evaluation harness as 10_compare_retrieval_modes.py
    (duplicated rather than imported -- see that script's own comment on
    this project's standalone-scripts convention). chunk_lookup and
    metadata_chunk_ids are passed in explicitly rather than read from a
    module-level global, since with three separate per-model chunk lists
    now in play (content-identical, but structurally separate files) an
    explicit parameter is less error-prone than an implicit shared global
    left over from when there was only ever one."""
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
            retrieved_chunk_ids = [r["chunk_id"] for r in retrieved_results]
            retrieved_document_ids = [r["document_id"] for r in retrieved_results]

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


def print_mode_report(label, tables):
    """Prints the four metric tables (Hit Rate, Recall, MRR, File Hit Rate)
    for one embedding model. DATASET SUMMARY is intentionally excluded here
    -- it is printed once after all models via print_dataset_summary(), to
    keep the terminal output concise (the summary rows are identical across
    all models since they all ran on the same gold question set)."""
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}EMBEDDING MODEL: {label}{RESET}")
    print(banner)

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}HIT RATE @ K{RESET}")
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
    print(f"\t\t\t{BOLD}RECALL @ K{RESET}")
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
    print(f"\t\t\t{BOLD}MRR @ K{RESET}")
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
    print(f"\t\t{BOLD}FILE HIT RATE @ K{RESET}")
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


def print_dataset_summary(model_results):
    """Prints the DATASET SUMMARY once, after all models' metric tables.
    All models ran on the same gold question set so the question counts,
    valid/skipped counts, and missing chunk IDs are identical across them
    -- printing it per-model would just repeat the same rows three times
    and make the terminal output unnecessarily long.

    Uses the first model's summary table as the canonical one; any model's
    would produce the same output for these fields."""
    first_tables = next(iter(model_results.values()))["tables"]
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}============")
    print(f"\t\t\t\t\t\t{BOLD}DATASET SUMMARY{RESET}")
    print(f"{banner}============")
    print(
        tabulate(
            first_tables["summary"],
            headers=summary_headers,
            tablefmt="grid",
            colalign=summary_colalign,
        )
    )


def macro_avg_at_5(tables, metric_key):
    """Computes the macro-average of the @5 column across all categories
    for a given metric. Row index 5 = @5 (index 0 is the category label,
    indices 1-10 are @1 through @10)."""
    vals = [row[5] for row in tables[metric_key]]
    return sum(vals) / len(vals) if vals else 0.0


def build_unified_comparison_table(model_results):
    """Builds a single comparison table with one row per metric and one
    column per model. Each cell is the macro-average of that metric's @5
    score across all four document categories (guidance, laws, prakas,
    sub_decrees). This gives a single at-a-glance number per model per
    metric rather than separate per-category breakdowns."""
    labels = list(model_results.keys())

    # Row order matches the per-model report section order above
    metric_defs = [
        ("Hit Rate", "hit"),
        ("Recall", "recall"),
        ("MRR", "mrr"),
        ("File Hit Rate", "file_hit"),
    ]

    rows = []
    for display_name, metric_key in metric_defs:
        row = [f"{display_name} @5"]
        for label in labels:
            avg = macro_avg_at_5(model_results[label]["tables"], metric_key)
            row.append(f"{avg:.4f}")
        rows.append(row)

    headers = ["Metric"] + labels
    return rows, headers


def rank_models(model_results):
    """Ranks models by the sum of all four macro-avg @5 metrics (equal
    weighting -- Hit Rate, Recall, MRR, File Hit Rate). Using all four
    rather than just Hit+Recall gives a more complete picture of ranking
    quality (MRR penalises models that find the right chunk but rank it
    low; File Hit Rate measures document-level coverage as a softer
    fallback signal)."""
    ranked = []
    for label, result in model_results.items():
        hit5 = macro_avg_at_5(result["tables"], "hit")
        recall5 = macro_avg_at_5(result["tables"], "recall")
        mrr5 = macro_avg_at_5(result["tables"], "mrr")
        filehit5 = macro_avg_at_5(result["tables"], "file_hit")
        combined = hit5 + recall5 + mrr5 + filehit5
        ranked.append((label, hit5, recall5, mrr5, filehit5, combined))

    ranked.sort(key=lambda x: x[5], reverse=True)
    return ranked


def build_recommendation(ranked):
    lines = ["RANKING (by macro-avg Hit@5 + Recall@5 + MRR@5 + File Hit Rate@5):"]
    for i, (label, hit5, recall5, mrr5, filehit5, combined) in enumerate(
        ranked, start=1
    ):
        lines.append(
            f"  {i}. {label}"
            f" -- Hit@5={hit5:.4f}, Recall@5={recall5:.4f},"
            f" MRR@5={mrr5:.4f}, FileHit@5={filehit5:.4f},"
            f" combined={combined:.4f}"
        )

    if len(ranked) > 1:
        gap = ranked[0][5] - ranked[1][5]
        if gap < 0.05:
            lines.append(
                f"\nNOTE: the top two models are close (combined-score gap "
                f"{gap:.4f}). Manual review of the per-category tables is "
                "recommended before trusting this ranking alone."
            )

    lines.append(f"\nRECOMMENDATION: {ranked[0][0]}")
    return "\n".join(lines)


def write_reports(
    model_results, comparison_rows, comparison_headers, recommendation, ranked
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "models": model_results,
        "comparison": {
            "headers": comparison_headers,
            "rows": comparison_rows,
            "note": (
                "Each cell is the macro-average of the @5 score for that metric "
                "across all four document categories (guidance, laws, prakas, sub_decrees)."
            ),
        },
        "ranking": [
            {
                "label": label,
                "macro_hit_at_5": hit5,
                "macro_recall_at_5": recall5,
                "macro_mrr_at_5": mrr5,
                "macro_file_hit_at_5": filehit5,
                "combined_score": combined,
            }
            for label, hit5, recall5, mrr5, filehit5, combined in ranked
        ],
        "recommendation": recommendation,
        "eval_ks": EVAL_KS,
    }
    json_path = REPORTS_DIR / "embedding_model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    def md_table(title, rows, headers):
        return (
            f"### {title}\n\n"
            + tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f")
            + "\n\n"
        )

    md_parts = ["# Embedding Model Comparison Report\n\n"]
    md_parts.append(
        f"Compares {len(model_results)} embedding models (dense-only, no "
        "hybrid/BM25) on the full gold question set, per Internship "
        "Proposal §6.3 / §12. Corpus encoding was precomputed by "
        "scripts/08_generate_embeddings.py; only the gold questions were "
        "encoded fresh here.\n\n"
    )

    for label, result in model_results.items():
        md_parts.append(f"## {label}\n\n")
        meta = result["meta"]
        md_parts.append(
            f"Model load time (this run): {meta['model_load_time']:.2f}s -- "
        )
        build_encode = meta.get("build_encode_time_sec")
        md_parts.append(
            f"Corpus encode time (at build, precomputed): " f"{build_encode:.2f}s\n\n"
            if build_encode is not None
            else "Corpus encode time (at build): unknown (no embedding_model_info.json)\n\n"
        )
        md_parts.append(
            f"Embedding dimension: {meta.get('embedding_dim', 'unknown')}\n\n"
        )
        md_parts.append(
            md_table("Hit Rate @ K", result["tables"]["hit"], metric_headers)
        )
        md_parts.append(
            md_table("Recall @ K", result["tables"]["recall"], metric_headers)
        )
        md_parts.append(md_table("MRR @ K", result["tables"]["mrr"], metric_headers))
        md_parts.append(
            md_table("File Hit Rate @ K", result["tables"]["file_hit"], metric_headers)
        )

    # Dataset summary once, before the cross-model comparison
    first_tables = next(iter(model_results.values()))["tables"]
    md_parts.append("## Dataset Summary\n\n")
    md_parts.append(
        tabulate(first_tables["summary"], headers=summary_headers, tablefmt="github")
    )
    md_parts.append("\n\n")

    # Single unified comparison table
    md_parts.append("## Comparison -- Macro-Avg @5 Across All Categories\n\n")
    md_parts.append(
        "> Each cell is the macro-average of the @5 score for that metric "
        "across guidance, laws, prakas, and sub_decrees.\n\n"
    )
    md_parts.append(
        tabulate(comparison_rows, headers=comparison_headers, tablefmt="github")
    )
    md_parts.append("\n\n## Recommendation\n\n```\n" + recommendation + "\n```\n")

    md_path = REPORTS_DIR / "embedding_model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

start_time = time.time()

model_results = {}
for key, model_config in EMBEDDING_MODELS.items():
    label = model_config["label"]
    print(
        f"\n{'=' * SEPARATOR_WIDTH}\nLoading precomputed index for: {label}\n{'=' * SEPARATOR_WIDTH}"
    )

    index, model_chunks, build_info = load_precomputed(model_config)
    chunk_lookup = {c["chunk_id"]: c for c in model_chunks}
    metadata_chunk_ids = set(chunk_lookup.keys())

    load_start = time.time()

    # All three models use plain SentenceTransformer loaded by Hub ID.
    kwargs = {
        "trust_remote_code": bool(model_config["trust_remote_code"]),
        "local_files_only": True,
    }
    model = SentenceTransformer(model_config["hf_id"], **kwargs)

    model_load_time = time.time() - load_start

    retrieve_fn = make_retrieve_fn(model, model_config, index, model_chunks)

    print(
        f"Evaluating gold questions for {label} (encoding queries only -- corpus already embedded)..."
    )
    tables = evaluate_retrieval(retrieve_fn, chunk_lookup, metadata_chunk_ids)

    model_results[label] = {
        "tables": tables,
        "meta": {
            "model_load_time": model_load_time,
            "build_encode_time_sec": build_info.get("build_encode_time_sec"),
            "embedding_dim": build_info.get("embedding_dimension", index.d),
        },
    }

    del model, index

# Print all per-model metric tables (no DATASET SUMMARY inside each)
for label, result in model_results.items():
    print_mode_report(label, result["tables"])

# DATASET SUMMARY printed once, after all three models' metric tables
print_dataset_summary(model_results)

# Single unified comparison table: rows = 4 metrics, columns = 3 models,
# values = macro-average of the @5 score across all four document categories
comparison_rows, comparison_headers = build_unified_comparison_table(model_results)
ranked = rank_models(model_results)
recommendation = build_recommendation(ranked)

banner = "=" * SEPARATOR_WIDTH
print(f"\n\n{banner}")
print(f"{BOLD}COMPARISON -- MACRO-AVG @5 ACROSS ALL CATEGORIES{RESET}")
print(
    f"{BOLD}(each value = average of guidance / laws / prakas / sub_decrees @5){RESET}"
)
print(banner)
print(
    tabulate(
        comparison_rows,
        headers=comparison_headers,
        tablefmt="grid",
        colalign=("left",) + ("center",) * (len(comparison_headers) - 1),
    )
)

print(f"\n\n{banner}")
print(f"{BOLD}{recommendation}{RESET}")
print(banner)

write_reports(
    model_results, comparison_rows, comparison_headers, recommendation, ranked
)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print(
    "Embedding comparison completed successfully (precomputed indices, no corpus re-encoding)."
)
