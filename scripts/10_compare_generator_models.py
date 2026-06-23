import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tabulate import tabulate

# Set HF_HOME before any HF import -- models live in retriever/.cache/,
# loaded by Hub ID. See scripts/13_install_models.py for the full explanation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Generator Comparison Report (Internship Proposal §6.4 / §12)
# ----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES AND DOES NOT MEASURE
#
# RAGAS defines four generator-quality metrics: Context Relevance,
# Faithfulness, Answer Relevance, Answer Correctness. Of those four, only
# Context Relevance is something a script can score without a human or an
# LLM judge in the loop -- it's a deterministic set-overlap check against
# expected_chunks, identical in spirit to scripts/08 and scripts/09's
# existing retrieval metrics, just computed against the FINAL context sent
# to the LLM (CONTEXT_K=5, post-RRF, post-select_context_chunks) rather
# than the broader raw top-10 candidate pool those two scripts evaluate.
#
# Faithfulness, Answer Relevance, and Answer Correctness all require
# actually reading the generated answer and judging it -- this script
# does NOT attempt that. scripts/11_evaluate_rag.py produces the
# human-scoring template for exactly those three (per Internship Proposal
# §9.3's human evaluation panel requirement). Conflating "this script ran
# without errors" with "the answer was good" would be worse than not
# measuring it at all, so this script's output should be read as: "how
# well does each generator's CONTEXT match the gold chunks, and how fast
# is it" -- not "which generator writes the best answers".
#
# This mirrors scripts/09's hybrid retrieval pipeline EXACTLY (same FAISS
# index, same BM25 tokenization, same RRF constant) rather than
# scripts/08's deliberately dense-only comparison, because the live
# chatbot (app.py) always runs hybrid retrieval -- evaluating generators
# against a retrieval mode the app never actually uses would make the
# comparison meaningless.

# Active embedding model -- must match ACTIVE_EMBEDDING_MODEL_DIR in app.py
EMBEDDING_HF_ID = "BAAI/bge-m3"
ACTIVE_MODEL_DIR = "BAAI-BGE-M3"

INDEX_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance.index"
METADATA_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance_metadata.json"
EVALUATION_ROOT = Path("data/08_evaluation")
REPORTS_DIR = Path("reports/generator")
LLM_DIR = Path("models/generator")

LLAMA_SERVER_URL = "http://localhost:8080/v1"
LLAMA_SERVER_ROOT = "http://localhost:8080"
LLAMA_SERVER_STARTUP_TIMEOUT = 120

# Mirrors app.py exactly -- context selection and prompt construction must
# be IDENTICAL to the live app, or this evaluation is testing a different
# pipeline than what users actually experience.
TOP_K = 10
CONTEXT_K = 5
MAX_CHUNKS_PER_DOC = 1
BM25_TOP_K = TOP_K
RRF_K = 60
NGRAM_SIZE = 3
MAX_TOKENS = 1024

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")

# ----------------------------------------------------------------------------
# Generator model registry
# ----------------------------------------------------------------------------
# Duplicated from scripts/13_install_models.py and start_chatbot.py --
# this project's scripts/ are each standalone, self-contained files with
# no shared utils module. If this changes in one place, change it in all
# three. extra_args mirrors start_chatbot.py's per-model llama-server
# flags exactly (Qwen3.5 needs --jinja --reasoning off or its <think>
# block can consume the entire MAX_TOKENS budget and the visible answer
# is never generated -- see start_chatbot.py's comment for the full
# explanation).
LLM_MODELS = {
    "1": {
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "label": "SeaLLM v3 7B Q4_K_M",
        "extra_args": ["--jinja"],
    },
    "2": {
        "local_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "label": "Qwen 3.5 9B Q4_K_M",
        "extra_args": ["--jinja", "--reasoning", "off"],
    },
    "3": {
        "local_filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "label": "Gemma 4 E4B Q4_K_M",
        "extra_args": ["--jinja"],
    },
    "4": {
        "local_filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.1 8B Instruct Q4_K_M",
        "extra_args": ["--jinja"],
    },
}

BOLD = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH


def khmer_aware_tokenize(text):
    """Identical to app.py / scripts/09 -- see either for the full
    docstring on why Khmer needs character n-grams under BM25."""
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


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Identical fusion logic to app.py's reciprocal_rank_fusion() and
    scripts/09's version -- see app.py for the full docstring on why RRF
    (rank-based) rather than raw score blending is used."""
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[cid], chunk_lookup[cid]) for cid in ranked_chunk_ids]


def select_context_chunks(results):
    """Identical to app.py's select_context_chunks() -- caps CONTEXT_K
    total, MAX_CHUNKS_PER_DOC per document. This is what actually reaches
    the LLM, and what Context Relevance is scored against below."""
    selected = []
    per_doc_count = {}
    for score, chunk in results:
        doc_name = chunk.get("metadata", {}).get("document_name", "unknown")
        if per_doc_count.get(doc_name, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        selected.append((score, chunk))
        per_doc_count[doc_name] = per_doc_count.get(doc_name, 0) + 1
        if len(selected) >= CONTEXT_K:
            break
    return selected


def build_context(results):
    """Identical to app.py's build_context()."""
    context_parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = metadata.get("document_name", "N/A")
        doc_type = metadata.get(
            "document_type_kh", metadata.get("document_type", "N/A")
        )
        chapter = metadata.get("chapter_title", metadata.get("chapter", "N/A"))
        section = metadata.get("section_title", metadata.get("section", "N/A"))
        legal_unit = metadata.get("legal_unit", "N/A")
        release_date = metadata.get("release_date", "N/A")
        header = (
            f"[Source {rank}] {doc_name} ({doc_type}, {release_date}) | "
            f"{chapter} > {section} | {legal_unit} | {chunk['chunk_id']}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(context_parts)


def build_prompt(question, context):
    """Identical core prompt to app.py's build_prompt() (low_confidence
    relevance-check branch omitted -- this script unconditionally
    generates for every valid gold question regardless of best_score,
    since the point here is comparing generators on identical inputs,
    not replicating the live out-of-domain gating)."""
    not_found_message = "រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។"
    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.

Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Provide a complete and detailed answer. If the context lists multiple conditions, sub-points, exceptions, or requirements, enumerate ALL of them.
- Only combine information from multiple sources if EACH of them directly addresses the question asked. Do not stitch together unrelated facts from different sources just because they were retrieved -- if most of the retrieved sources don't actually answer the question, say so instead of forcing a combined answer out of all of them.
- Do not reinterpret a number, rate, or statistic for a purpose other than what it was stated for in the source.
- Do not recommend, endorse, or compare any specific insurance product, company, or policy, even if the context mentions one and the question asks for a recommendation.
- Do not calculate, estimate, or infer a premium, payout, deductible, or any monetary amount that is not stated explicitly and directly in the context for that exact purpose.
- Do not give personalized financial or legal advice. If the question asks for a personal recommendation, a calculation, or advice tailored to the user's situation, decline and reply with the message below instead of attempting an answer.
- If the answer is not found in the context, OR the question asks for a product recommendation, a premium/payout calculation, or personalized advice, reply EXACTLY with this message and nothing else:

{not_found_message}

Context:
{context}

Question:
{question}

Answer:
"""


def is_installed(info):
    return (LLM_DIR / info["local_filename"]).exists()


def start_llama_server(info):
    """Launches llama-server for one generator model, mirroring
    start_chatbot.py's flags exactly (same --ctx-size/--flash-attn/
    --cache-type-k/v, same per-model extra_args). Returns the Popen
    handle so the caller can terminate() it once this model's evaluation
    is done and move on to the next one."""
    llama_server_bin = shutil.which("llama-server")
    if llama_server_bin is None:
        raise SystemExit(
            "llama-server not found. Please install llama.cpp and add it to PATH."
        )

    model_path = LLM_DIR / info["local_filename"]

    process = subprocess.Popen(
        [
            llama_server_bin,
            "-m",
            str(model_path),
            "-ngl",
            "999",
            "--ctx-size",
            "16384",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]
        + info["extra_args"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start_time = time.time()
    while True:
        if time.time() - start_time > LLAMA_SERVER_STARTUP_TIMEOUT:
            process.kill()
            raise SystemExit(
                f"llama-server failed to start within "
                f"{LLAMA_SERVER_STARTUP_TIMEOUT}s for {info['label']}."
            )
        try:
            response = requests.get(f"{LLAMA_SERVER_ROOT}/v1/models", timeout=5)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)

    return process


def stop_llama_server(process):
    """Mirrors start_chatbot.py's shutdown -- terminate() rather than
    kill() so llama-server can release the model's memory cleanly before
    the next model in the sweep loads."""
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    # Give the OS a moment to actually reclaim GPU/RAM before the next
    # model's llama-server tries to load -- without this, two server
    # processes can briefly overlap and double the memory footprint.
    time.sleep(2)


def count_tokens(text):
    """Identical to app.py's count_tokens() -- real token count via
    llama-server's /tokenize endpoint, not a character estimate."""
    try:
        resp = requests.post(
            f"{LLAMA_SERVER_ROOT}/tokenize", json={"content": text}, timeout=10
        )
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Load embedding model, FAISS index, BM25 index, gold questions -- ONCE.
# Retrieval is generator-independent, so none of this is redone per model.
# ----------------------------------------------------------------------------

if not Path(INDEX_PATH).exists():
    raise FileNotFoundError(
        f"FAISS index not found at {INDEX_PATH}.\n"
        "Run: python scripts/07_generate_embeddings.py"
    )

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_HF_ID)

print("Loading FAISS index...")
faiss_index = faiss.read_index(INDEX_PATH)

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
    scores, indices = faiss_index.search(query_embedding, k=TOP_K)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((float(score), chunks[idx]))
    return results


def retrieve_bm25(question):
    tokenized_query = khmer_aware_tokenize(question)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def retrieve_hybrid(question):
    dense_results = retrieve_dense(question)
    bm25_results = retrieve_bm25(question)
    return reciprocal_rank_fusion([dense_results, bm25_results])


# ----------------------------------------------------------------------------
# Per-model evaluation
# ----------------------------------------------------------------------------


def evaluate_model(client, model_name):
    """Runs every gold question through hybrid retrieval -> context
    selection -> generation, against the currently-running model. Scores
    only Context Relevance (deterministic, against the FINAL CONTEXT_K
    chunks) and timing -- never answer quality. See module docstring."""
    per_category_rows = []
    all_first_token_times = []
    all_generation_times = []
    all_tokens_per_sec = []
    skipped_total = 0
    valid_total = 0

    for gold_file in gold_files:
        category = gold_file.parent.name
        with open(gold_file, "r", encoding="utf-8") as f:
            gold_questions = json.load(f)

        if not gold_questions:
            raise ValueError(f"{gold_file} is empty.")

        hit_count = 0
        recall_sum = 0.0
        valid_questions = 0
        skipped_questions = 0

        for item in gold_questions:
            question = item["question"]
            expected_chunks = set(item.get("expected_chunks", []))

            if not expected_chunks:
                skipped_questions += 1
                continue

            valid_questions += 1

            fused_results = retrieve_hybrid(question)
            context_results = select_context_chunks(fused_results)
            context_chunk_ids = {chunk["chunk_id"] for _, chunk in context_results}

            found = context_chunk_ids.intersection(expected_chunks)
            if found:
                hit_count += 1
            recall_sum += len(found) / len(expected_chunks)

            context = build_context(context_results)
            prompt = build_prompt(question, context)

            gen_start = time.time()
            first_token_time = None
            total_tokens_out = 0

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                for chunk_resp in response:
                    delta = chunk_resp.choices[0].delta.content
                    if delta:
                        total_tokens_out += 1
                        if first_token_time is None:
                            first_token_time = time.time() - gen_start
                try:
                    response.close()
                except Exception:
                    pass
            except Exception as e:
                print(f"  Generation failed for question (skipping timing): {e}")
                continue

            generation_time = time.time() - gen_start
            decode_time = generation_time - (first_token_time or 0.0)

            all_first_token_times.append(first_token_time or 0.0)
            all_generation_times.append(generation_time)
            if decode_time > 0 and total_tokens_out > 0:
                all_tokens_per_sec.append(total_tokens_out / decode_time)

        if valid_questions == 0:
            raise ValueError(f"No valid questions found in {gold_file}.")

        per_category_rows.append(
            [
                category,
                valid_questions,
                skipped_questions,
                hit_count / valid_questions,
                recall_sum / valid_questions,
            ]
        )
        valid_total += valid_questions
        skipped_total += skipped_questions

    return {
        "per_category": per_category_rows,
        "avg_first_token_time": (
            sum(all_first_token_times) / len(all_first_token_times)
            if all_first_token_times
            else 0.0
        ),
        "avg_generation_time": (
            sum(all_generation_times) / len(all_generation_times)
            if all_generation_times
            else 0.0
        ),
        "avg_tokens_per_sec": (
            sum(all_tokens_per_sec) / len(all_tokens_per_sec)
            if all_tokens_per_sec
            else 0.0
        ),
        "valid_questions": valid_total,
        "skipped_questions": skipped_total,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

context_headers = [
    "Category",
    "Valid Questions",
    "Skipped Questions",
    f"Context Hit@{CONTEXT_K}",
    f"Context Recall@{CONTEXT_K}",
]
context_colalign = ("center",) * len(context_headers)


def print_model_report(label, result):
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}GENERATOR: {label}{RESET}")
    print(banner)

    print(f"\n{SEPARATOR}")
    print(
        f"\t\t{BOLD}CONTEXT RELEVANCE{RESET} - overlap between expected_chunks "
        f"and the actual CONTEXT_K={CONTEXT_K} chunks sent to this generator"
    )
    print(SEPARATOR)
    print(
        tabulate(
            result["per_category"],
            headers=context_headers,
            tablefmt="grid",
            floatfmt=".4f",
            colalign=context_colalign,
        )
    )

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}TIMING{RESET}")
    print(SEPARATOR)
    print(
        tabulate(
            [
                ["Avg first-token time (s)", f"{result['avg_first_token_time']:.4f}"],
                ["Avg total generation time (s)", f"{result['avg_generation_time']:.4f}"],
                ["Avg decode speed (tokens/sec)", f"{result['avg_tokens_per_sec']:.4f}"],
            ],
            tablefmt="grid",
        )
    )


def build_comparison_rows(model_results):
    rows = []
    for label, result in model_results.items():
        all_hits = [row[3] for row in result["per_category"]]
        all_recalls = [row[4] for row in result["per_category"]]
        macro_hit = sum(all_hits) / len(all_hits) if all_hits else 0.0
        macro_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
        rows.append(
            [
                label,
                f"{macro_hit:.4f}",
                f"{macro_recall:.4f}",
                f"{result['avg_first_token_time']:.4f}",
                f"{result['avg_generation_time']:.4f}",
                f"{result['avg_tokens_per_sec']:.4f}",
            ]
        )
    headers = [
        "Generator",
        f"Macro Context Hit@{CONTEXT_K}",
        f"Macro Context Recall@{CONTEXT_K}",
        "Avg First Token (s)",
        "Avg Generation Time (s)",
        "Avg Tokens/Sec",
    ]
    return rows, headers


def write_reports(model_results, comparison_rows, comparison_headers):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "models": model_results,
        "comparison": {"headers": comparison_headers, "rows": comparison_rows},
        "context_k": CONTEXT_K,
        "note": (
            "Context Relevance and timing are automated, deterministic "
            "metrics. Faithfulness, Answer Relevance, and Answer "
            "Correctness are NOT scored by this script -- see "
            "scripts/11_evaluate_rag.py for the human-scoring template "
            "covering those three."
        ),
    }
    json_path = REPORTS_DIR / "generator_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    md_parts = ["# Generator Comparison Report\n\n"]
    md_parts.append(
        "Compares generator models on **Context Relevance** (deterministic, "
        f"against the actual CONTEXT_K={CONTEXT_K} chunks sent to the LLM -- "
        "identical retrieval pipeline to app.py) and generation timing. "
        "**Faithfulness, Answer Relevance, and Answer Correctness are NOT "
        "scored here** -- those need human or LLM-judge review of the actual "
        "generated text. See `scripts/11_evaluate_rag.py` and "
        "`reports/generator/generator_eval_template_<model>.txt` for those "
        "three.\n\n"
    )
    for label, result in model_results.items():
        md_parts.append(f"## {label}\n\n")
        md_parts.append(
            tabulate(
                result["per_category"],
                headers=context_headers,
                tablefmt="github",
                floatfmt=".4f",
            )
        )
        md_parts.append(
            f"\n\nAvg first-token time: {result['avg_first_token_time']:.4f}s -- "
            f"Avg generation time: {result['avg_generation_time']:.4f}s -- "
            f"Avg decode speed: {result['avg_tokens_per_sec']:.4f} tok/s\n\n"
        )

    md_parts.append("## Comparison\n\n")
    md_parts.append(tabulate(comparison_rows, headers=comparison_headers, tablefmt="github"))
    md_parts.append("\n")

    md_path = REPORTS_DIR / "generator_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------

start_time = time.time()

installed_models = {
    key: info for key, info in LLM_MODELS.items() if is_installed(info)
}

if not installed_models:
    raise SystemExit(
        "No generator models installed -- run: python scripts/13_install_models.py"
    )

skipped_labels = [
    info["label"] for key, info in LLM_MODELS.items() if key not in installed_models
]
if skipped_labels:
    print("Not installed, skipping:")
    for label in skipped_labels:
        print(f"  - {label}")

model_results = {}

for key, info in installed_models.items():
    label = info["label"]
    print(f"\n{'=' * SEPARATOR_WIDTH}\nStarting llama-server for: {label}\n{'=' * SEPARATOR_WIDTH}")

    process = start_llama_server(info)
    try:
        client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
        model_response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        model_name = model_response.json()["data"][0]["id"]

        print(f"Evaluating {label} on gold questions...")
        result = evaluate_model(client, model_name)
        model_results[label] = result
    finally:
        print(f"Stopping llama-server for: {label}")
        stop_llama_server(process)

for label, result in model_results.items():
    print_model_report(label, result)

comparison_rows, comparison_headers = build_comparison_rows(model_results)

banner = "=" * SEPARATOR_WIDTH
print(f"\n\n{banner}")
print(f"{BOLD}COMPARISON -- ALL GENERATORS{RESET}")
print(banner)
print(
    tabulate(
        comparison_rows,
        headers=comparison_headers,
        tablefmt="grid",
        colalign=("center",) * len(comparison_headers),
    )
)

write_reports(model_results, comparison_rows, comparison_headers)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print(
    "Generator comparison completed successfully (Context Relevance + "
    "timing only -- run scripts/11_evaluate_rag.py for Faithfulness / "
    "Answer Relevance / Answer Correctness)."
)