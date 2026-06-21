import json
import time
import faiss
import numpy as np
from pathlib import Path
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Embedding Comparison Report (Internship Proposal §6.3, §12)
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
# from the lexical layer riding alongside it. The hybrid question is
# already answered separately (see 08_evaluate_retrieval.py); this script
# isolates ONE variable: which embedding model produces better vectors
# for this corpus.
#
# MiniLM is the model referenced early in this project's development --
# tested informally, found weaker, never benchmarked with real saved
# numbers. This script produces those real numbers instead of relying on
# memory of "it wasn't as good."
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/08_evaluation")
REPORTS_DIR = Path("reports/embeddings")

EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

# (label, local model path)
# Both models are loaded from local installed copies -- run
# scripts/13_install_models.py first. MiniLM is installed there as a
# comparison-only model (not used by the live app), specifically so this
# script never has to silently download ~470MB from the Hugging Face Hub
# mid-evaluation.
MODEL_CONFIGS = [
    ("BGE-M3 (current)", "models/retriever/BAAI-BGE-M3"),
    (
        "MiniLM (paraphrase-multilingual-MiniLM-L12-v2)",
        "models/retriever/MiniLM-paraphrase-multilingual-L12-v2",
    ),
]

for label, model_path in MODEL_CONFIGS:
    if not (Path(model_path) / "modules.json").exists():
        raise FileNotFoundError(
            f"{label} not found at {model_path}.\n"
            "Run: python scripts/13_install_models.py"
        )

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


print("Loading chunk metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
metadata_chunk_ids = set(chunk_lookup.keys())

# Same text every model embeds, identical to what 07_generate_embeddings.py
# feeds BGE-M3 in production -- isolates the comparison to the model only,
# not to differences in what text each model happened to see.
embedding_texts = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")


def build_model_index(model_path_or_id):
    """Loads a SentenceTransformer model and builds a fresh in-memory FAISS
    index over the exact same corpus text the production index uses.
    Nothing here touches vector_db/ -- this is a throwaway index that
    exists only for the duration of this script, never written to disk
    and never used by the live app."""
    load_start = time.time()
    model = SentenceTransformer(model_path_or_id)
    load_time = time.time() - load_start

    print(f"  Encoding {len(embedding_texts)} chunks...")
    encode_start = time.time()
    embeddings = model.encode(
        embedding_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    encode_time = time.time() - encode_start

    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, {
        "load_time": load_time,
        "encode_time": encode_time,
        "embedding_dim": int(embeddings.shape[1]),
    }


def make_retrieve_fn(model, index):
    """Returns a retrieve(question) function closed over this specific
    model/index pair, matching the same result-dict shape
    08_evaluate_retrieval.py's retrieve_dense() uses, so the evaluation
    loop below is identical between the two."""

    def retrieve(question):
        query_embedding = model.encode([question], normalize_embeddings=True)
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

    return retrieve


def evaluate_retrieval(retrieve_fn):
    """Same gold-question evaluation harness as 08_evaluate_retrieval.py
    (duplicated rather than imported -- this project's scripts/ are each
    standalone, self-contained files with no shared utils module; see
    that script's own comment on this same convention)."""
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
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}EMBEDDING MODEL: {label}{RESET}")
    print(banner)

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}HIT RATE @ K{RESET}")
    print(SEPARATOR)
    print(tabulate(tables["hit"], headers=metric_headers, tablefmt="grid", floatfmt=".4f", colalign=metric_colalign))

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}RECALL @ K{RESET}")
    print(SEPARATOR)
    print(tabulate(tables["recall"], headers=metric_headers, tablefmt="grid", floatfmt=".4f", colalign=metric_colalign))

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}MRR @ K{RESET}")
    print(SEPARATOR)
    print(tabulate(tables["mrr"], headers=metric_headers, tablefmt="grid", floatfmt=".4f", colalign=metric_colalign))

    print(f"\n{SEPARATOR}")
    print(f"\t\t{BOLD}FILE HIT RATE @ K{RESET}")
    print(SEPARATOR)
    print(tabulate(tables["file_hit"], headers=metric_headers, tablefmt="grid", floatfmt=".4f", colalign=metric_colalign))

    print(f"\n{SEPARATOR}============")
    print(f"\t\t\t\t\t\t{BOLD}DATASET SUMMARY{RESET}")
    print(f"{SEPARATOR}============")
    print(tabulate(tables["summary"], headers=summary_headers, tablefmt="grid", colalign=summary_colalign))


def build_comparison(model_a_label, tables_a, model_b_label, tables_b):
    """Headline Hit@5 / Recall@5 per category for both models plus the
    delta, and an overall macro-average across categories used to drive
    the final recommendation (the proposal explicitly asks for "a
    justified recommendation," not just raw tables)."""
    rows = []
    a_hit = {row[0]: row[5] for row in tables_a["hit"]}
    b_hit = {row[0]: row[5] for row in tables_b["hit"]}
    a_recall = {row[0]: row[5] for row in tables_a["recall"]}
    b_recall = {row[0]: row[5] for row in tables_b["recall"]}

    for category in a_hit:
        rows.append(
            [
                category,
                f"{a_hit[category]:.4f}",
                f"{b_hit[category]:.4f}",
                f"{b_hit[category] - a_hit[category]:+.4f}",
                f"{a_recall[category]:.4f}",
                f"{b_recall[category]:.4f}",
                f"{b_recall[category] - a_recall[category]:+.4f}",
            ]
        )

    headers = [
        "Category",
        f"Hit@5 ({model_a_label})",
        f"Hit@5 ({model_b_label})",
        "Δ Hit@5",
        f"Recall@5 ({model_a_label})",
        f"Recall@5 ({model_b_label})",
        "Δ Recall@5",
    ]

    macro_a_hit = sum(a_hit.values()) / len(a_hit)
    macro_b_hit = sum(b_hit.values()) / len(b_hit)
    macro_a_recall = sum(a_recall.values()) / len(a_recall)
    macro_b_recall = sum(b_recall.values()) / len(b_recall)

    if macro_a_hit >= macro_b_hit and macro_a_recall >= macro_b_recall:
        winner = model_a_label
    elif macro_b_hit >= macro_a_hit and macro_b_recall >= macro_a_recall:
        winner = model_b_label
    else:
        # Mixed result -- one model wins Hit@5, the other wins Recall@5.
        # Don't force a winner; report both macro numbers and let a human
        # make the call, since the proposal asks for a JUSTIFIED
        # recommendation, not an automated one.
        winner = None

    recommendation = (
        f"RECOMMENDATION: {winner} -- macro-avg Hit@5 {max(macro_a_hit, macro_b_hit):.4f} "
        f"and Recall@5 {max(macro_a_recall, macro_b_recall):.4f} across all categories, "
        f"versus {min(macro_a_hit, macro_b_hit):.4f} / {min(macro_a_recall, macro_b_recall):.4f} "
        f"for the other model."
        if winner
        else (
            f"MIXED RESULT -- no single model wins on both metrics. "
            f"{model_a_label}: macro-avg Hit@5={macro_a_hit:.4f}, Recall@5={macro_a_recall:.4f}. "
            f"{model_b_label}: macro-avg Hit@5={macro_b_hit:.4f}, Recall@5={macro_b_recall:.4f}. "
            f"Manual review of per-category results recommended before choosing."
        )
    )

    return rows, headers, recommendation, {
        model_a_label: {"macro_hit_at_5": macro_a_hit, "macro_recall_at_5": macro_a_recall},
        model_b_label: {"macro_hit_at_5": macro_b_hit, "macro_recall_at_5": macro_b_recall},
    }


def write_reports(model_results, comparison_rows, comparison_headers, recommendation, macro_scores):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "models": model_results,
        "comparison": {"headers": comparison_headers, "rows": comparison_rows},
        "macro_scores": macro_scores,
        "recommendation": recommendation,
        "eval_ks": EVAL_KS,
    }
    json_path = REPORTS_DIR / "embedding_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    def md_table(title, rows, headers):
        return f"### {title}\n\n" + tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f") + "\n\n"

    md_parts = ["# Embedding Comparison Report\n\n"]
    md_parts.append(
        "Compares retrieval quality (dense-only, no hybrid/BM25) across "
        "embedding models on the full gold question set, per Internship "
        "Proposal §6.3 / §12.\n\n"
    )
    for label, result in model_results.items():
        md_parts.append(f"## {label}\n\n")
        md_parts.append(f"Model load time: {result['meta']['load_time']:.2f}s -- ")
        md_parts.append(f"Encode time: {result['meta']['encode_time']:.2f}s -- ")
        md_parts.append(f"Embedding dimension: {result['meta']['embedding_dim']}\n\n")
        md_parts.append(md_table("Hit Rate @ K", result["tables"]["hit"], metric_headers))
        md_parts.append(md_table("Recall @ K", result["tables"]["recall"], metric_headers))
        md_parts.append(md_table("MRR @ K", result["tables"]["mrr"], metric_headers))
        md_parts.append(md_table("File Hit Rate @ K", result["tables"]["file_hit"], metric_headers))
        md_parts.append(md_table("Dataset Summary", result["tables"]["summary"], summary_headers))

    md_parts.append("## Before / After Comparison (Hit@5 / Recall@5)\n\n")
    md_parts.append(tabulate(comparison_rows, headers=comparison_headers, tablefmt="github"))
    md_parts.append("\n\n## Recommendation\n\n")
    md_parts.append(recommendation + "\n")

    md_path = REPORTS_DIR / "embedding_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


start_time = time.time()

model_results = {}
for label, model_path in MODEL_CONFIGS:
    print(f"\n{'=' * SEPARATOR_WIDTH}\nLoading and encoding with: {label}\n{'=' * SEPARATOR_WIDTH}")
    model, index, meta = build_model_index(model_path)
    retrieve_fn = make_retrieve_fn(model, index)

    print(f"Running gold-question evaluation for {label}...")
    tables = evaluate_retrieval(retrieve_fn)
    model_results[label] = {"tables": tables, "meta": meta}

    # Free the model/index before loading the next one -- two multilingual
    # transformer models held in memory simultaneously is unnecessary
    # memory pressure on a machine that's also expected to run llama-server.
    del model, index

(label_a, _), (label_b, _) = MODEL_CONFIGS
comparison_rows, comparison_headers, recommendation, macro_scores = build_comparison(
    label_a, model_results[label_a]["tables"], label_b, model_results[label_b]["tables"]
)

for label, result in model_results.items():
    print_mode_report(label, result["tables"])

banner = "=" * SEPARATOR_WIDTH
print(f"\n\n{banner}")
print(f"{BOLD}BEFORE / AFTER -- {label_a} vs. {label_b}{RESET}")
print(banner)
print(tabulate(comparison_rows, headers=comparison_headers, tablefmt="grid", colalign=("center",) * 7))
print(f"\n{BOLD}{recommendation}{RESET}")

write_reports(model_results, comparison_rows, comparison_headers, recommendation, macro_scores)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print("Embedding comparison completed successfully.")