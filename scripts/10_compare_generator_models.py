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
# Fully Automated Generator Comparison (RAGAS-style proxy metrics)
# ----------------------------------------------------------------------------
# PURPOSE: pick a winner among the 4 installed generators, with ZERO human
# scoring time, across the FULL gold question set (no sampling, no
# max_tokens cap -- every answer is generated exactly as the live app
# would generate it).
#
# This is NOT the rigorous, human-verified RAGAS evaluation -- it's a
# fast, fully programmatic PROXY for all four metrics, good enough to
# rank 4 models against each other. scripts/11_evaluate_rag.py is where
# the WINNING model (and only that one) gets the full human-scoring
# treatment, compared side by side against this same proxy to check
# whether the proxy can be trusted going forward.
#
# How each of the four RAGAS-style metrics is approximated WITHOUT a
# human or an LLM judge in the loop:
#
#   Context Relevance  -- deterministic. Exact chunk_id overlap between
#                          expected_chunks and the actual CONTEXT_K=5
#                          chunks sent to the LLM (identical to the
#                          earlier version of this script). No proxy
#                          needed -- this one was never approximate.
#
#   Answer Correctness -- embedding cosine similarity between the
#                          generated answer and expected_answer, using
#                          BGE-M3 (the same model already loaded for
#                          retrieval -- no second model to load). This
#                          mirrors RAGAS's own "Answer Semantic
#                          Similarity" metric exactly: cosine similarity
#                          between generated-answer and ground-truth
#                          embeddings, thresholded into a 1-5 bucket.
#
#   Answer Relevance   -- embedding cosine similarity between the
#                          generated answer and the ORIGINAL QUESTION
#                          (not the expected answer). Real RAGAS computes
#                          this via an LLM generating hypothetical
#                          questions from the answer and comparing THOSE
#                          to the real question -- that extra LLM step is
#                          skipped here to keep this fully programmatic;
#                          comparing the answer directly to the question
#                          is a cruder but zero-LLM proxy for "does this
#                          answer's content actually address what was
#                          asked".
#
#   Faithfulness        -- NOT approximated via embedding similarity.
#                          Two semantically-similar texts can still
#                          disagree on a specific fact (an invented date,
#                          a wrong fee figure, a fabricated article
#                          number) -- exactly the failure mode embedding
#                          similarity is least able to catch, since a
#                          hallucinated number sits right next to real,
#                          on-topic legal text. Instead: every number
#                          (Arabic or Khmer numeral) in the generated
#                          answer is checked against the retrieved
#                          CONTEXT_K chunks. Any number in the answer that
#                          does NOT appear anywhere in the context is a
#                          potential fabricated figure -- this is a
#                          deterministic, zero-LLM, zero-embedding check
#                          that directly targets the highest-risk
#                          hallucination type in a legal-citation domain
#                          (dates, fees, article/prakas numbers), even
#                          though it can't catch every kind of
#                          unfaithfulness (e.g. a fabricated *qualitative*
#                          claim with no numbers in it would slip through
#                          -- this is a proxy, not a substitute for the
#                          human review scripts/11 still performs on the
#                          winning model).
#
# Thresholds for converting cosine similarity into a 1-5 bucket are
# calibrated conservatively rather than assuming an even 0.0-1.0 spread:
# multilingual sentence embeddings (BGE-M3 included) tend to compress
# unrelated-text similarity into a higher-than-naively-expected range, so
# a flat linear mapping would systematically over-score weak answers.
# These thresholds are a reasonable starting point, NOT empirically
# tuned against this corpus -- if scripts/11's human scores on the
# winning model disagree systematically with this script's proxy scores
# (e.g. humans consistently rate something a 2 that this script scores a
# 4), that's the signal to revisit these specific cutoffs, the same way
# OUT_OF_DOMAIN_THRESHOLD in app.py was calibrated empirically rather
# than guessed once and trusted forever.
SIMILARITY_THRESHOLDS = [
    (0.85, 5),
    (0.75, 4),
    (0.60, 3),
    (0.45, 2),
    (0.00, 1),
]

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
# pipeline than what users actually experience. No sampling, no token cap:
# every gold question, full max_tokens, exactly as the live app behaves.
TOP_K = 10
CONTEXT_K = 5
MAX_CHUNKS_PER_DOC = 1
BM25_TOP_K = TOP_K
RRF_K = 60
NGRAM_SIZE = 3
MAX_TOKENS = 1024

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")

# Arabic digit runs (1, 2, 3, ...) and Khmer numeral runs (១, ២, ៣, ...)
# -- used by the Faithfulness number-overlap proxy below. Runs rather
# than single digits, so "16384" is checked as one token, not five.
NUMBER_PATTERN = re.compile(r"[0-9]+|[០១២៣៤៥៦៧៨៩]+")

# ----------------------------------------------------------------------------
# Generator model registry
# ----------------------------------------------------------------------------
# Duplicated from scripts/13_install_models.py and start_chatbot.py -- see
# either for the full explanation of why this project's scripts/ each
# duplicate shared registries rather than importing a common module.
# extra_args mirrors start_chatbot.py's per-model llama-server flags
# exactly (Qwen3.5 needs --jinja --reasoning off or its <think> block can
# consume the entire MAX_TOKENS budget and the visible answer is never
# generated -- see start_chatbot.py's comment for the full explanation).
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
    """Identical fusion logic to app.py's reciprocal_rank_fusion() --
    see app.py for the full docstring on why RRF (rank-based) rather
    than raw score blending is used."""
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
    the LLM, and what Context Relevance / Faithfulness are scored
    against below."""
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
    start_chatbot.py's flags exactly. Returns the Popen handle so the
    caller can terminate() it once this model's evaluation is done and
    move on to the next one."""
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


# ----------------------------------------------------------------------------
# Proxy metric scoring
# ----------------------------------------------------------------------------


def cosine_similarity(vec_a, vec_b):
    """Both vectors are already L2-normalized by encode_for_scoring()
    (normalize_embeddings=True), so the dot product alone IS the cosine
    similarity -- no need to divide by norms again."""
    return float(np.dot(vec_a, vec_b))


def similarity_to_score(similarity):
    """Buckets a cosine similarity into a 1-5 score using
    SIMILARITY_THRESHOLDS (see module docstring for why these specific
    cutoffs, and why they're a starting point, not a tuned constant)."""
    for threshold, score in SIMILARITY_THRESHOLDS:
        if similarity >= threshold:
            return score
    return 1


def extract_numbers(text):
    """Returns the set of distinct number-strings (Arabic or Khmer
    numeral runs) appearing in text. Used by the Faithfulness proxy --
    see module docstring for why number-overlap specifically, rather
    than embedding similarity, is used for this one metric."""
    return set(NUMBER_PATTERN.findall(text))


def faithfulness_proxy_score(generated_answer, context):
    """Deterministic Faithfulness proxy: what fraction of the numbers
    appearing in the generated answer also appear somewhere in the
    retrieved context? 1.0 = every number in the answer is grounded in
    context (or the answer has no numbers at all, which trivially passes
    this specific check). Returns (score_1_to_5, grounded_count,
    total_count, ungrounded_numbers) so the raw evidence is always
    available in the report, not just the bucketed score."""
    answer_numbers = extract_numbers(generated_answer)
    if not answer_numbers:
        # No numbers to check -- this proxy has nothing to flag. Scored
        # as a neutral-to-good 4 rather than an automatic 5, since
        # "no numbers present" is weaker evidence of faithfulness than
        # "numbers present and all verified against context".
        return 4, 0, 0, []

    context_numbers = extract_numbers(context)
    grounded = answer_numbers & context_numbers
    ungrounded = sorted(answer_numbers - context_numbers)

    fraction_grounded = len(grounded) / len(answer_numbers)

    if fraction_grounded == 1.0:
        score = 5
    elif fraction_grounded >= 0.75:
        score = 4
    elif fraction_grounded >= 0.5:
        score = 3
    elif fraction_grounded > 0:
        score = 2
    else:
        score = 1

    return score, len(grounded), len(answer_numbers), ungrounded


# ----------------------------------------------------------------------------
# Load embedding model, FAISS index, BM25 index, gold questions -- ONCE.
# Retrieval AND scoring both reuse this single BGE-M3 instance, since it's
# already loaded for retrieval -- no second embedding model needed for
# the Answer Correctness / Answer Relevance proxies.
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


def encode_for_scoring(texts):
    """Encodes arbitrary text (generated answers, expected answers,
    questions) with the SAME BGE-M3 instance used for retrieval.
    normalize_embeddings=True so cosine_similarity() can use a plain dot
    product. Not prefixed with any e5-style query/passage prefix --
    BGE-M3 doesn't require one (only multilingual-e5-large does, and
    that's not the model in use here)."""
    return np.asarray(
        embedding_model.encode(texts, normalize_embeddings=True), dtype="float32"
    )


# ----------------------------------------------------------------------------
# Per-model evaluation
# ----------------------------------------------------------------------------


def collect_gold_entries():
    """Loads every gold question across all categories. Questions
    without expected_chunks are skipped for Context Relevance scoring;
    questions without expected_answer are skipped for Answer Correctness
    scoring -- each tracked and reported separately rather than silently
    dropped, since this script (unlike scripts/11) does NOT discard a
    question just because one of the two reference fields is missing."""
    entries = []
    for gold_file in gold_files:
        category = gold_file.parent.name
        with open(gold_file, "r", encoding="utf-8") as f:
            gold_questions = json.load(f)

        if not gold_questions:
            raise ValueError(f"{gold_file} is empty.")

        for item in gold_questions:
            entries.append(
                {
                    "category": category,
                    "question": item["question"],
                    "expected_chunks": item.get("expected_chunks", []),
                    "expected_answer": item.get("expected_answer"),
                }
            )

    return entries


def evaluate_model(client, model_name, entries):
    """Runs every gold question through hybrid retrieval -> context
    selection -> generation -> all four proxy metrics, against the
    currently-running model."""
    per_entry_results = []

    for i, entry in enumerate(entries, start=1):
        print(f"  [{i}/{len(entries)}] {entry['question'][:60]}...")

        fused_results = retrieve_hybrid(entry["question"])
        context_results = select_context_chunks(fused_results)
        context_chunk_ids = {chunk["chunk_id"] for _, chunk in context_results}
        context = build_context(context_results)
        prompt = build_prompt(entry["question"], context)

        gen_start = time.time()
        first_token_time = None
        total_tokens_out = 0
        answer_text = ""

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
                    answer_text += delta
                    total_tokens_out += 1
                    if first_token_time is None:
                        first_token_time = time.time() - gen_start
            try:
                response.close()
            except Exception:
                pass
        except Exception as e:
            print(f"    Generation failed: {e}")
            per_entry_results.append(
                {
                    "category": entry["category"],
                    "question": entry["question"],
                    "error": str(e),
                }
            )
            continue

        generation_time = time.time() - gen_start
        decode_time = generation_time - (first_token_time or 0.0)
        tokens_per_sec = (
            total_tokens_out / decode_time if decode_time > 0 and total_tokens_out > 0 else 0.0
        )

        # --- Context Relevance: deterministic chunk_id overlap ---
        context_relevance_score = None
        expected_chunks = set(entry["expected_chunks"])
        if expected_chunks:
            found = context_chunk_ids.intersection(expected_chunks)
            recall = len(found) / len(expected_chunks)
            context_relevance_score = round(1 + recall * 4)  # maps 0.0-1.0 -> 1-5

        # --- Answer Correctness: cosine sim(generated, expected_answer) ---
        answer_correctness_score = None
        answer_correctness_similarity = None
        if entry["expected_answer"]:
            vecs = encode_for_scoring([answer_text, entry["expected_answer"]])
            answer_correctness_similarity = cosine_similarity(vecs[0], vecs[1])
            answer_correctness_score = similarity_to_score(answer_correctness_similarity)

        # --- Answer Relevance: cosine sim(generated, question) ---
        vecs = encode_for_scoring([answer_text, entry["question"]])
        answer_relevance_similarity = cosine_similarity(vecs[0], vecs[1])
        answer_relevance_score = similarity_to_score(answer_relevance_similarity)

        # --- Faithfulness: number-overlap proxy vs context ---
        (
            faithfulness_score,
            grounded_count,
            total_numbers,
            ungrounded_numbers,
        ) = faithfulness_proxy_score(answer_text, context)

        per_entry_results.append(
            {
                "category": entry["category"],
                "question": entry["question"],
                "generated_answer": answer_text,
                "context_relevance_score": context_relevance_score,
                "answer_correctness_score": answer_correctness_score,
                "answer_correctness_similarity": answer_correctness_similarity,
                "answer_relevance_score": answer_relevance_score,
                "answer_relevance_similarity": answer_relevance_similarity,
                "faithfulness_score": faithfulness_score,
                "faithfulness_grounded": grounded_count,
                "faithfulness_total_numbers": total_numbers,
                "faithfulness_ungrounded_numbers": ungrounded_numbers,
                "first_token_time": first_token_time or 0.0,
                "generation_time": generation_time,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    return per_entry_results


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

PROXY_METRIC_KEYS = [
    ("context_relevance_score", "Context Relevance"),
    ("answer_correctness_score", "Answer Correctness"),
    ("answer_relevance_score", "Answer Relevance"),
    ("faithfulness_score", "Faithfulness"),
]


def summarize_model(per_entry_results):
    valid_results = [r for r in per_entry_results if "error" not in r]
    failed_count = len(per_entry_results) - len(valid_results)

    averages = {}
    for key, _ in PROXY_METRIC_KEYS:
        scored = [r[key] for r in valid_results if r.get(key) is not None]
        averages[key] = sum(scored) / len(scored) if scored else None

    timing = {
        "avg_first_token_time": sum(r["first_token_time"] for r in valid_results) / len(valid_results) if valid_results else 0.0,
        "avg_generation_time": sum(r["generation_time"] for r in valid_results) / len(valid_results) if valid_results else 0.0,
        "avg_tokens_per_sec": (
            sum(r["tokens_per_sec"] for r in valid_results if r["tokens_per_sec"] > 0)
            / max(len([r for r in valid_results if r["tokens_per_sec"] > 0]), 1)
        ),
    }

    return {
        "averages": averages,
        "timing": timing,
        "valid_count": len(valid_results),
        "failed_count": failed_count,
        "total_count": len(per_entry_results),
    }


def print_model_report(label, summary):
    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}GENERATOR: {label}{RESET}")
    print(banner)

    rows = []
    for key, metric_label in PROXY_METRIC_KEYS:
        avg = summary["averages"][key]
        rows.append([metric_label, f"{avg:.2f}" if avg is not None else "N/A"])

    print(f"\n{SEPARATOR}")
    print(f"\t\t{BOLD}PROXY METRICS (avg 1-5, fully automated){RESET}")
    print(SEPARATOR)
    print(tabulate(rows, headers=["Metric", "Avg Score"], tablefmt="grid"))

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}TIMING{RESET}")
    print(SEPARATOR)
    print(
        tabulate(
            [
                ["Avg first-token time (s)", f"{summary['timing']['avg_first_token_time']:.4f}"],
                ["Avg total generation time (s)", f"{summary['timing']['avg_generation_time']:.4f}"],
                ["Avg decode speed (tokens/sec)", f"{summary['timing']['avg_tokens_per_sec']:.4f}"],
                ["Valid / Failed / Total", f"{summary['valid_count']} / {summary['failed_count']} / {summary['total_count']}"],
            ],
            tablefmt="grid",
        )
    )


def build_comparison_rows(model_summaries):
    rows = []
    for label, summary in model_summaries.items():
        avg = summary["averages"]
        rows.append(
            [
                label,
                f"{avg['context_relevance_score']:.2f}" if avg["context_relevance_score"] is not None else "N/A",
                f"{avg['answer_correctness_score']:.2f}" if avg["answer_correctness_score"] is not None else "N/A",
                f"{avg['answer_relevance_score']:.2f}" if avg["answer_relevance_score"] is not None else "N/A",
                f"{avg['faithfulness_score']:.2f}" if avg["faithfulness_score"] is not None else "N/A",
                f"{summary['timing']['avg_generation_time']:.2f}",
            ]
        )
    headers = [
        "Generator",
        "Context Relevance",
        "Answer Correctness",
        "Answer Relevance",
        "Faithfulness (proxy)",
        "Avg Generation Time (s)",
    ]
    return rows, headers


def rank_models(model_summaries):
    """Ranks by the sum of the 4 average proxy scores (equal weighting --
    deliberately simple and transparent rather than a tuned weighted
    formula, since these are proxy metrics to begin with; a fancier
    weighting on top of an already-approximate signal would be false
    precision)."""
    ranked = []
    for label, summary in model_summaries.items():
        avg = summary["averages"]
        combined = sum(v for v in avg.values() if v is not None)
        ranked.append((label, avg, combined))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def write_reports(model_summaries, all_results, comparison_rows, comparison_headers, ranked):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "models": model_summaries,
        "per_entry_results": all_results,
        "comparison": {"headers": comparison_headers, "rows": comparison_rows},
        "ranking": [
            {"label": label, "averages": avg, "combined_score": combined}
            for label, avg, combined in ranked
        ],
        "note": (
            "These are AUTOMATED PROXY metrics, not human-verified RAGAS "
            "scores. Context Relevance is exact (deterministic chunk "
            "overlap). Answer Correctness and Answer Relevance use BGE-M3 "
            "cosine similarity thresholds, which is the same approach "
            "RAGAS's own Answer Semantic Similarity metric uses, but "
            "thresholds here are a starting calibration, not tuned "
            "against this corpus. Faithfulness uses a number-overlap "
            "check against retrieved context, which catches fabricated "
            "figures but not all forms of unfaithfulness. Run "
            "scripts/11_evaluate_rag.py on the winning model for "
            "human-verified scores to validate (or recalibrate) this "
            "proxy."
        ),
    }
    json_path = REPORTS_DIR / "generator_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    md_parts = ["# Generator Comparison Report (Automated Proxy Metrics)\n\n"]
    md_parts.append(
        "Compares all installed generator models on four RAGAS-style "
        "proxy metrics, computed with **zero human review**, across the "
        "full gold question set with no sampling and the same "
        f"`MAX_TOKENS={MAX_TOKENS}` budget the live app uses.\n\n"
        "**Context Relevance** is exact (deterministic chunk-ID overlap "
        "against the actual chunks sent to the LLM). **Answer "
        "Correctness** and **Answer Relevance** use BGE-M3 cosine "
        "similarity, the same underlying approach as RAGAS's own Answer "
        "Semantic Similarity metric. **Faithfulness** uses a "
        "number-overlap check against retrieved context, catching "
        "fabricated dates/fees/figures specifically -- it does not catch "
        "every kind of unfaithfulness.\n\n"
        "Run `scripts/11_evaluate_rag.py` on the winning model for "
        "human-verified scores, to check whether this proxy can be "
        "trusted going forward.\n\n"
    )

    for label, summary in model_summaries.items():
        md_parts.append(f"## {label}\n\n")
        rows = []
        for key, metric_label in PROXY_METRIC_KEYS:
            avg = summary["averages"][key]
            rows.append([metric_label, f"{avg:.2f}" if avg is not None else "N/A"])
        md_parts.append(
            tabulate(rows, headers=["Metric", "Avg Score"], tablefmt="github") + "\n\n"
        )
        md_parts.append(
            f"Avg first-token time: {summary['timing']['avg_first_token_time']:.4f}s -- "
            f"Avg generation time: {summary['timing']['avg_generation_time']:.4f}s -- "
            f"Avg decode speed: {summary['timing']['avg_tokens_per_sec']:.4f} tok/s -- "
            f"Valid/Failed/Total: {summary['valid_count']}/{summary['failed_count']}/{summary['total_count']}\n\n"
        )

    md_parts.append("## Comparison\n\n")
    md_parts.append(tabulate(comparison_rows, headers=comparison_headers, tablefmt="github"))
    md_parts.append("\n\n## Ranking\n\n```\n")
    for i, (label, avg, combined) in enumerate(ranked, start=1):
        md_parts.append(f"{i}. {label} -- combined score {combined:.2f}\n")
    md_parts.append(f"\nRECOMMENDATION: {ranked[0][0]}\n```\n")

    md_path = REPORTS_DIR / "generator_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved Markdown report : {md_path}")


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------

start_time = time.time()

entries = collect_gold_entries()
print(f"\nLoaded {len(entries)} gold questions across {len(gold_files)} categories.")

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

model_summaries = {}
all_results = {}

for key, info in installed_models.items():
    label = info["label"]
    print(f"\n{'=' * SEPARATOR_WIDTH}\nStarting llama-server for: {label}\n{'=' * SEPARATOR_WIDTH}")

    process = start_llama_server(info)
    try:
        client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
        model_response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        model_name = model_response.json()["data"][0]["id"]

        print(f"Evaluating {label} on {len(entries)} gold questions...")
        per_entry_results = evaluate_model(client, model_name, entries)
        model_summaries[label] = summarize_model(per_entry_results)
        all_results[label] = per_entry_results
    finally:
        print(f"Stopping llama-server for: {label}")
        stop_llama_server(process)

for label, summary in model_summaries.items():
    print_model_report(label, summary)

comparison_rows, comparison_headers = build_comparison_rows(model_summaries)
ranked = rank_models(model_summaries)

banner = "=" * SEPARATOR_WIDTH
print(f"\n\n{banner}")
print(f"{BOLD}COMPARISON -- ALL GENERATORS (proxy metrics){RESET}")
print(banner)
print(
    tabulate(
        comparison_rows,
        headers=comparison_headers,
        tablefmt="grid",
        colalign=("center",) * len(comparison_headers),
    )
)

print(f"\n{banner}")
print(f"{BOLD}RANKING (combined score = sum of 4 avg proxy metrics){RESET}")
print(banner)
for i, (label, avg, combined) in enumerate(ranked, start=1):
    print(f"  {i}. {label} -- combined score {combined:.2f}")
print(f"\n{BOLD}RECOMMENDATION: {ranked[0][0]}{RESET}")

write_reports(model_summaries, all_results, comparison_rows, comparison_headers, ranked)

elapsed_time = time.time() - start_time
print(f"\nTotal Runtime : {elapsed_time:.2f} seconds")
print(
    "Generator comparison completed successfully (fully automated proxy "
    "metrics -- run scripts/11_evaluate_rag.py on the winning model for "
    "human-verified scores)."
)