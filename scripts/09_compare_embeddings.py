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
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/08_evaluation")
REPORTS_DIR = Path("reports/embeddings")

EVAL_KS = list(range(1, 11))
MAX_K = max(EVAL_KS)

# Duplicated from scripts/13_install_models.py / scripts/07_generate_embeddings.py
# -- this project's scripts/ are each standalone, self-contained files
# with no shared utils module. If this changes in one place, change it in
# all three.
#
# jina-embeddings-v3 needs `trust_remote_code=True` to load and a `task`
# argument at encode time -- a DIFFERENT task for documents
# ("retrieval.passage") vs. queries ("retrieval.query"). Using the wrong
# one, or none at all, doesn't error -- it silently produces a worse
# embedding via the wrong (or no) LoRA adapter, which would make this
# comparison invalid for that model specifically. BGE-M3 and MiniLM
# ignore the task argument entirely (task_aware=False below).
EMBEDDING_MODELS = {
    "1": {
        "label": "BAAI/bge-m3",
        "local_dir": Path("models/retriever/BAAI-BGE-M3"),
        "trust_remote_code": False,
        "task_aware": False,
    },
    "2": {
        "label": "jinaai/jina-embeddings-v3",
        "local_dir": Path("models/retriever/jinaai-jina-embeddings-v3"),
        "trust_remote_code": True,
        "task_aware": True,
    },
    "3": {
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "local_dir": Path("models/retriever/MiniLM-paraphrase-multilingual-L12-v2"),
        "trust_remote_code": False,
        "task_aware": False,
    },
}

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


for key, info in EMBEDDING_MODELS.items():
    if not (info["local_dir"] / "modules.json").exists():
        raise FileNotFoundError(
            f"{info['label']} not found at {info['local_dir']}.\n"
            "Run: python scripts/13_install_models.py"
        )


print("Loading chunk metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
metadata_chunk_ids = set(chunk_lookup.keys())

# Same text every model embeds, identical to what 07_generate_embeddings.py
# feeds the chosen model in production -- isolates the comparison to the
# model only, not to differences in what text each model happened to see.
embedding_texts = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]

gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))

if not gold_files:
    raise ValueError("No gold_questions.json files found.")


def encode_documents(model, model_config, texts, **kwargs):
    """Encodes corpus chunks as PASSAGES, not queries -- matters only for
    task-aware models. See the EMBEDDING_MODELS comment block above."""
    if model_config.get("task_aware"):
        kwargs["task"] = "retrieval.passage"
        kwargs["prompt_name"] = "retrieval.passage"
    return model.encode(texts, **kwargs)


def encode_queries(model, model_config, texts, **kwargs):
    """Encodes search QUERIES, not documents -- the other half of the
    asymmetric task pair for task-aware models."""
    if model_config.get("task_aware"):
        kwargs["task"] = "retrieval.query"
        kwargs["prompt_name"] = "retrieval.query"
    return model.encode(texts, **kwargs)


def build_model_index(model_config):
    """Loads a SentenceTransformer model and builds a fresh in-memory FAISS
    index over the exact same corpus text the production index uses.
    Nothing here touches vector_db/ -- this is a throwaway index that
    exists only for the duration of this script."""
    load_start = time.time()
    kwargs = {"trust_remote_code": True} if model_config["trust_remote_code"] else {}
    model = SentenceTransformer(str(model_config["local_dir"]), **kwargs)
    load_time = time.time() - load_start

    print(f"  Encoding {len(embedding_texts)} chunks (as passages)...")
    encode_start = time.time()
    embeddings = encode_documents(
        model,
        model_config,
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


def make_retrieve_fn(model, model_config, index):
    """Returns a retrieve(question) function closed over this specific
    model/index pair. Encodes the question as a QUERY (encode_queries),
    not as a document -- this is the piece that was previously missing
    entirely for jina-embeddings-v3."""

    def retrieve(question):
        query_embedding = encode_queries(
            model, model_config, [question], normalize_embeddings=True
        )
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
    (duplicated rather than imported -- see that script's own comment on
    this project's standalone-scripts convention)."""
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


def build_metric_comparison_table(model_results, metric_key):
    """Category-by-model comparison at @5 for one metric ('hit' or
    'recall') -- one row per category, one column per model. Generalizes
    to any number of models, unlike a fixed before/after two-column
    layout."""
    labels = list(model_results.keys())
    categories = [row[0] for row in next(iter(model_results.values()))["tables"][metric_key]]

    per_model_lookup = {
        label: {row[0]: row[5] for row in result["tables"][metric_key]}  # @5 is index 5
        for label, result in model_results.items()
    }

    rows = []
    for category in categories:
        row = [category] + [f"{per_model_lookup[label][category]:.4f}" for label in labels]
        rows.append(row)

    headers = ["Category"] + labels
    return rows, headers


def rank_models(model_results):
    """Ranks all models by macro-averaged Hit@5 + Recall@5 across every
    category. Returns a list of (label, macro_hit5, macro_recall5,
    combined_score) sorted best-first. Generalizes the old binary
    A-vs-B winner logic to N models."""
    ranked = []
    for label, result in model_results.items():
        hit5_vals = [row[5] for row in result["tables"]["hit"]]
        recall5_vals = [row[5] for row in result["tables"]["recall"]]
        macro_hit5 = sum(hit5_vals) / len(hit5_vals)
        macro_recall5 = sum(recall5_vals) / len(recall5_vals)
        ranked.append((label, macro_hit5, macro_recall5, macro_hit5 + macro_recall5))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked


def build_recommendation(ranked):
    """Builds the "justified recommendation" text the proposal's
    deliverable asks for -- not just a bare winner name, but the full
    ranking plus an honest flag when the top two are close enough that a
    human should look at the per-category breakdown before trusting the
    automatic pick."""
    lines = ["RANKING (by macro-avg Hit@5 + Recall@5 across all categories):"]
    for i, (label, hit5, recall5, combined) in enumerate(ranked, start=1):
        lines.append(
            f"  {i}. {label} -- Hit@5={hit5:.4f}, Recall@5={recall5:.4f}, combined={combined:.4f}"
        )

    if len(ranked) > 1:
        gap = ranked[0][3] - ranked[1][3]
        if gap < 0.02:
            lines.append(
                f"\nNOTE: the top {2 if len(ranked) >= 2 else 1} models are close "
                f"(combined-score gap {gap:.4f}). Manual review of the "
                "per-category tables is recommended before trusting this "
                "ranking alone."
            )

    lines.append(f"\nRECOMMENDATION: {ranked[0][0]}")
    return "\n".join(lines)


def write_reports(model_results, hit_rows, hit_headers, recall_rows, recall_headers, recommendation, ranked):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "models": model_results,
        "comparison": {
            "hit_at_5": {"headers": hit_headers, "rows": hit_rows},
            "recall_at_5": {"headers": recall_headers, "rows": recall_rows},
        },
        "ranking": [
            {"label": label, "macro_hit_at_5": h, "macro_recall_at_5": r, "combined_score": c}
            for label, h, r, c in ranked
        ],
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
        f"Compares {len(model_results)} embedding models (dense-only, no "
        "hybrid/BM25) on the full gold question set, per Internship "
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

    md_parts.append("## Comparison -- Hit@5 by Category\n\n")
    md_parts.append(tabulate(hit_rows, headers=hit_headers, tablefmt="github"))
    md_parts.append("\n\n## Comparison -- Recall@5 by Category\n\n")
    md_parts.append(tabulate(recall_rows, headers=recall_headers, tablefmt="github"))
    md_parts.append("\n\n## Recommendation\n\n```\n" + recommendation + "\n```\n")

    md_path = REPORTS_DIR / "embedding_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


start_time = time.time()

model_results = {}
for key, model_config in EMBEDDING_MODELS.items():
    label = model_config["label"]
    print(f"\n{'=' * SEPARATOR_WIDTH}\nLoading and encoding with: {label}\n{'=' * SEPARATOR_WIDTH}")
    model, index, meta = build_model_index(model_config)

    retrieve_fn = make_retrieve_fn(model, model_config, index)

    print(f"Running gold-question evaluation for {label}...")
    tables = evaluate_retrieval(retrieve_fn)
    model_results[label] = {"tables": tables, "meta": meta}

    # Free each model/index before loading the next -- three multilingual
    # transformer models in memory simultaneously is unnecessary pressure
    # on a machine that may also be running llama-server.
    del model, index

for label, result in model_results.items():
    print_mode_report(label, result["tables"])

hit_rows, hit_headers = build_metric_comparison_table(model_results, "hit")
recall_rows, recall_headers = build_metric_comparison_table(model_results, "recall")
ranked = rank_models(model_results)
recommendation = build_recommendation(ranked)

banner = "=" * SEPARATOR_WIDTH
print(f"\n\n{banner}")
print(f"{BOLD}COMPARISON -- HIT@5 BY CATEGORY{RESET}")
print(banner)
print(tabulate(hit_rows, headers=hit_headers, tablefmt="grid", colalign=("center",) * len(hit_headers)))

print(f"\n\n{banner}")
print(f"{BOLD}COMPARISON -- RECALL@5 BY CATEGORY{RESET}")
print(banner)
print(tabulate(recall_rows, headers=recall_headers, tablefmt="grid", colalign=("center",) * len(recall_headers)))

print(f"\n\n{banner}")
print(f"{BOLD}{recommendation}{RESET}")
print(banner)

write_reports(model_results, hit_rows, hit_headers, recall_rows, recall_headers, recommendation, ranked)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print("Embedding comparison completed successfully.")