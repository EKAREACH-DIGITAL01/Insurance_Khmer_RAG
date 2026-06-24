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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Generator Model Comparison (fully automated, no human scoring)
# ----------------------------------------------------------------------------
# PURPOSE: rank all installed generator models against each other on the full
# gold question set (25 questions × 4 categories = 100 questions per model)
# using five automated proxy metrics. The winning model is then used for the
# human-scored RAGAS evaluation in scripts/11_evaluate_rag.py.
#
# METRICS (all automated, no LLM judge):
#
#   Context Relevance  -- deterministic chunk-ID overlap between
#                         expected_chunks and the CONTEXT_K chunks actually
#                         sent to the LLM. Exact, not a proxy.
#
#   Faithfulness       -- Arabic + Khmer numeral overlap between the generated
#                         answer and the retrieved context. Catches fabricated
#                         dates, fees, article numbers -- the highest-risk
#                         hallucination type in a legal domain. Returns a
#                         1-5 score based on fraction of answer numbers that
#                         are grounded in context.
#
#   Answer Correctness -- BGE-M3 cosine similarity between generated answer
#                         and expected_answer. Mirrors RAGAS's own Answer
#                         Semantic Similarity metric exactly. Uses its OWN
#                         threshold set (ANSWER_CORRECTNESS_THRESHOLDS),
#                         calibrated against this corpus's real distribution
#                         -- see docs/SCORING_CALIBRATION.md.
#
#   Answer Relevance   -- BGE-M3 cosine similarity between generated answer
#                         and the original question. Crude but zero-LLM proxy
#                         for "does this answer address what was asked". Uses
#                         its OWN threshold set (ANSWER_RELEVANCE_THRESHOLDS)
#                         -- this metric's similarity ceiling sits structurally
#                         lower than Answer Correctness's (a question and its
#                         answer are different grammatical forms), so sharing
#                         one threshold set with Answer Correctness would make
#                         a 5 nearly unreachable here. See
#                         docs/SCORING_CALIBRATION.md.
#
#   Completeness       -- Checks (a) whether the answer ends with a Khmer
#                         sentence-final punctuation mark (។ or ៕), and (b)
#                         the length ratio of generated vs expected answer,
#                         with BOTH a floor (catches truncation) AND a ceiling
#                         (catches routine over-length/padding). The ceiling
#                         was added after real data showed one model's
#                         length_ratio routinely running 2-4x+ the expected
#                         answer's length while still scoring a free 5 under
#                         the original floor-only logic -- see
#                         docs/SCORING_CALIBRATION.md for the full evidence.
#
# WEIGHTS (legal domain priorities):
#   Context Relevance  0.20  -- retrieval correctness
#   Faithfulness       0.25  -- fabricated numbers are actively harmful
#   Answer Correctness 0.25  -- semantic match to ground truth
#   Answer Relevance   0.15  -- addresses the question
#   Completeness       0.15  -- full answer, not truncated mid-sentence,
#                               not padded well past the expected length
#
# REAL-TIME WRITING: every generated answer is written to disk immediately
# after generation. If the run is interrupted (laptop dies, crash), every
# completed answer is already saved and the run can be restarted from where
# it left off (the script skips questions already answered for a given model).
#
# DRY RUN: pass --dry-run to run exactly 1 question per model and print the
# raw answer to the terminal. Use this to verify all 4 models behave
# technically correctly before committing to the full ~6-hour run.
#
# USAGE:
#   python scripts/10_compare_generator_models.py            # full run
#   python scripts/10_compare_generator_models.py --dry-run  # 1 question/model
# ----------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dry-run",
    action="store_true",
    help=(
        "Run exactly 1 question per model and print the raw answer. "
        "Use this to verify all models behave correctly before the full run."
    ),
)
args = parser.parse_args()
DRY_RUN = args.dry_run

# ----------------------------------------------------------------------------
# Config -- mirrors app.py exactly so evaluation matches production behaviour
# ----------------------------------------------------------------------------
EMBEDDING_HF_ID  = "BAAI/bge-m3"
ACTIVE_MODEL_DIR = "BAAI-BGE-M3"
INDEX_PATH       = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance.index"
METADATA_PATH    = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance_metadata.json"
EVALUATION_ROOT  = Path("data/08_evaluation")
REPORTS_DIR      = Path("reports/generator_model_comparison")
LLM_DIR          = Path("models/generator")

LLAMA_SERVER_URL             = "http://localhost:8080/v1"
LLAMA_SERVER_ROOT            = "http://localhost:8080"
LLAMA_SERVER_STARTUP_TIMEOUT = 120

# Mirrors app.py constants exactly
TOP_K             = 10
CONTEXT_K         = 5
MAX_CHUNKS_PER_DOC = 1
BM25_TOP_K        = TOP_K
RRF_K             = 60
NGRAM_SIZE        = 3
MAX_TOKENS        = 1024  # same as live app -- evaluate under production conditions

# Metric weights -- must sum to 1.0
WEIGHTS = {
    "context_relevance":  0.20,
    "faithfulness":       0.25,
    "answer_correctness": 0.25,
    "answer_relevance":   0.15,
    "completeness":       0.15,
}

# Khmer sentence-final punctuation -- used by completeness check
KHMER_END_PUNCT = {"។", "៕"}

# Arabic + Khmer numeral runs -- used by faithfulness proxy
NUMBER_PATTERN = re.compile(r"[0-9]+|[០១២៣៤៥៦៧៨៩]+")

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")

# Cosine similarity -> 1-5 bucket thresholds. Two SEPARATE sets because
# Answer Correctness (answer vs. expected_answer) and Answer Relevance
# (answer vs. question) compare structurally different text pairs with
# different natural similarity ceilings -- see docs/SCORING_CALIBRATION.md
# for the full percentile evidence behind both.
ANSWER_CORRECTNESS_THRESHOLDS = [(0.90,5),(0.80,4),(0.70,3),(0.60,2),(0.50,1),(0.00,1)]
ANSWER_RELEVANCE_THRESHOLDS = [(0.82,5),(0.75,4),(0.65,3),(0.55,2),(0.00,1)]

BOLD  = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH

# ----------------------------------------------------------------------------
# Generator model registry
# Duplicated from start_chatbot.py -- standalone scripts convention.
# extra_args mirrors start_chatbot.py exactly (Qwen needs --reasoning off).
# ----------------------------------------------------------------------------
LLM_MODELS = {
    "1": {
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "label": "SeaLLM v3 7B",
        "extra_args": ["--jinja"],
    },
    "2": {
        "local_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "label": "Qwen 3.5 9B",
        "extra_args": ["--jinja", "--reasoning", "off"],
    },
    "3": {
        "local_filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "label": "Gemma 4 E4B",
        "extra_args": ["--jinja"],
    },
    "4": {
        "local_filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.1 8B",
        "extra_args": ["--jinja"],
    },
}


# ----------------------------------------------------------------------------
# Retrieval -- identical to app.py
# ----------------------------------------------------------------------------

def khmer_aware_tokenize(text):
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
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[cid], chunk_lookup[cid]) for cid in ranked_ids]


def select_context_chunks(results):
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
    context_parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata     = chunk.get("metadata", {})
        doc_name     = metadata.get("document_name", "N/A")
        doc_type     = metadata.get("document_type_kh", metadata.get("document_type", "N/A"))
        chapter      = metadata.get("chapter_title", metadata.get("chapter", "N/A"))
        section      = metadata.get("section_title", metadata.get("section", "N/A"))
        legal_unit   = metadata.get("legal_unit", "N/A")
        release_date = metadata.get("release_date", "N/A")
        header = (
            f"[Source {rank}] {doc_name} ({doc_type}, {release_date}) | "
            f"{chapter} > {section} | {legal_unit} | {chunk['chunk_id']}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(context_parts)


# ----------------------------------------------------------------------------
# Prompt -- duplicated from app.py's build_prompt().
# If you update the prompt in app.py, update it here too.
# The low_confidence branch is omitted -- script 10 generates for every
# gold question unconditionally (no score gating during evaluation).
# ----------------------------------------------------------------------------

def build_prompt(question, context):
    not_found_message = (
        "រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។ "
        "សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ "
        "ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC) សម្រាប់ជំនួយបន្ថែម។"
    )
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


# ----------------------------------------------------------------------------
# Automated proxy metrics
# ----------------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    """Both vectors are L2-normalised so dot product == cosine similarity."""
    return float(np.dot(vec_a, vec_b))


def similarity_to_score(sim, thresholds):
    for threshold, score in thresholds:
        if sim >= threshold:
            return score
    return 1


def score_context_relevance(context_chunk_ids, expected_chunks):
    """Deterministic: fraction of expected chunks present in context, → 1-5."""
    if not expected_chunks:
        return None
    found  = set(context_chunk_ids) & set(expected_chunks)
    recall = len(found) / len(expected_chunks)
    return round(1 + recall * 4)


def score_faithfulness(generated_answer, context):
    """Number-overlap proxy: what fraction of numbers in the generated answer
    also appear in the retrieved context? Returns (score_1_to_5, grounded,
    total, ungrounded_list).

    CONFIRMED by real evidence (see docs/SCORING_CALIBRATION.md): this
    metric correctly separated Qwen (writes 2.9-4.3x more numbers per
    answer than the other 3 models, with only 72% grounded vs 94-100%
    for the others) from the rest -- a real, substantial difference in
    hallucination behaviour, not a length artifact (this is fraction-
    based per answer, already normalised for length). Left unchanged."""
    answer_numbers = set(NUMBER_PATTERN.findall(generated_answer))
    if not answer_numbers:
        # No numbers to check -- neutral score, not automatic 5
        return 4, 0, 0, []
    context_numbers = set(NUMBER_PATTERN.findall(context))
    grounded   = answer_numbers & context_numbers
    ungrounded = sorted(answer_numbers - context_numbers)
    fraction   = len(grounded) / len(answer_numbers)
    if fraction == 1.0:
        score = 5
    elif fraction >= 0.75:
        score = 4
    elif fraction >= 0.50:
        score = 3
    elif fraction > 0:
        score = 2
    else:
        score = 1
    return score, len(grounded), len(answer_numbers), ungrounded


def score_completeness(generated_answer, expected_answer):
    """Checks two things:
    1. Does the answer end with Khmer sentence-final punctuation (។ or ៕)?
       Answers that end mid-word/mid-sentence hit max_tokens and are incomplete.
    2. Length ratio: len(generated) / len(expected). Both TOO SHORT and
       TOO LONG relative to the expected answer are penalised.

    FIXED by real evidence (see docs/SCORING_CALIBRATION.md): the
    original version of this function only checked a FLOOR
    (length_ratio >= 0.80 -> automatic 5 if ends cleanly), with NO
    CEILING -- a length_ratio of 0.85 and a length_ratio of 10.0 scored
    identically. Real data showed this isn't a hypothetical edge case:
    Qwen's length_ratio median across 100 real answers was already 1.98x
    (49/100 answers over 2x, 30/100 over 3x, max 10.0x), while SeaLLM and
    Gemma rarely exceeded 2x (9/100 and 2/100 respectively). Combined
    with Faithfulness's finding that Qwen also grounds a smaller
    fraction of the numbers it states, the evidence points to a real
    pattern -- Qwen routinely writes long, elaborated answers containing
    more unverified detail -- that the floor-only version was silently
    rewarding with a free perfect score.

    Score 1-5 (ends_cleanly branch):
       5 = ends cleanly AND 0.80 <= length ratio <= 2.00
       4 = ends cleanly AND (0.60 <= ratio < 0.80, OR 2.00 < ratio <= 3.00)
       3 = ends cleanly AND (ratio < 0.60, OR ratio > 3.00)
    Score 1-5 (truncated branch -- UNCHANGED from original design, since
    the evidence above was specifically about the ceiling, not this
    branch):
       3 = truncated AND ratio >= 0.80
       2 = truncated AND ratio >= 0.40
       1 = truncated AND ratio < 0.40
    """
    stripped     = generated_answer.strip()
    ends_cleanly = bool(stripped) and stripped[-1] in KHMER_END_PUNCT

    if not expected_answer or len(expected_answer) == 0:
        # No expected answer to compare against -- score only on punctuation
        return (5 if ends_cleanly else 2), ends_cleanly, None

    length_ratio = len(stripped) / len(expected_answer)

    if ends_cleanly:
        if 0.80 <= length_ratio <= 2.00:
            score = 5
        elif (0.60 <= length_ratio < 0.80) or (2.00 < length_ratio <= 3.00):
            score = 4
        else:
            # either genuinely short (< 0.60) or well past 3x expected
            # length (> 3.00) -- not penalised as harshly as a real
            # truncation, since the model DID finish its thought, but no
            # longer a free top score either way.
            score = 3
    else:
        if length_ratio >= 0.80:
            score = 3
        elif length_ratio >= 0.40:
            score = 2
        else:
            score = 1

    return score, ends_cleanly, round(length_ratio, 4)


def weighted_combined(scores):
    """Weighted sum of the five proxy metric scores (each 1-5).
    Returns a value in [1, 5].

    BUG FIX: when expected_answer is missing, answer_correctness is None and
    skipped. If we used the full weight denominator (1.0) the combined score
    would be artificially low because we'd be dividing a partial sum by 1.0
    instead of the sum of weights that actually had values. We normalise by
    the sum of the weights that contributed, so the result stays on the
    correct [1, 5] scale regardless of which metrics are available."""
    total        = 0.0
    weight_sum   = 0.0
    for key, weight in WEIGHTS.items():
        val = scores.get(key)
        if val is not None:
            total      += val * weight
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
    # Normalise so the result is always on [1, 5] regardless of missing metrics
    return round(total / weight_sum, 4)


# ----------------------------------------------------------------------------
# llama-server lifecycle
# ----------------------------------------------------------------------------

def is_installed(info):
    return (LLM_DIR / info["local_filename"]).exists()


def start_llama_server(info):
    llama_bin = shutil.which("llama-server")
    if llama_bin is None:
        raise SystemExit("llama-server not found. Install llama.cpp and add it to PATH.")

    model_path = LLM_DIR / info["local_filename"]
    process = subprocess.Popen(
        [
            llama_bin, "-m", str(model_path),
            "-ngl", "999",
            "--ctx-size", "16384",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--host", "0.0.0.0",
            "--port", "8080",
        ] + info["extra_args"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = time.time()
    while True:
        if time.time() - start > LLAMA_SERVER_STARTUP_TIMEOUT:
            process.kill()
            raise SystemExit(
                f"llama-server failed to start within "
                f"{LLAMA_SERVER_STARTUP_TIMEOUT}s for {info['label']}."
            )
        try:
            r = requests.get(f"{LLAMA_SERVER_ROOT}/v1/models", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    return process


def stop_llama_server(process):
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    # Give the OS a moment to reclaim GPU/RAM before the next model loads
    time.sleep(2)


# ----------------------------------------------------------------------------
# Per-question generation + scoring
# ----------------------------------------------------------------------------

def generate_answer(client, model_name, prompt):
    """Streams the answer from llama-server. Returns (answer_text,
    first_token_time, generation_time, tokens_out)."""
    gen_start        = time.time()
    first_token_time = None
    answer_text      = ""
    tokens_out       = 0

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            answer_text += delta
            tokens_out  += 1
            if first_token_time is None:
                first_token_time = time.time() - gen_start
    try:
        response.close()
    except Exception:
        pass

    generation_time = time.time() - gen_start
    return answer_text, first_token_time or 0.0, generation_time, tokens_out


def score_answer(
    answer_text,
    question,
    expected_answer,
    expected_chunks,
    context_chunk_ids,
    context,
    embedding_model,
):
    """Runs all five proxy metrics and returns a dict of scores + detail.

    BUG FIX: answer_text may be an empty string if the model produced no
    output (silent failure -- no exception raised, just zero tokens). Cosine
    similarity between an empty-string embedding and any other vector is
    undefined / produces near-zero results that corrupt the ranking. Guard
    on non-empty answer_text before encoding for metrics 3 and 4."""

    # 1. Context Relevance (deterministic)
    ctx_rel = score_context_relevance(context_chunk_ids, expected_chunks)

    # 2. Faithfulness (number overlap)
    faith_score, faith_grounded, faith_total, faith_ungrounded = score_faithfulness(
        answer_text, context
    )

    # 3. Answer Correctness (BGE-M3 cosine vs expected_answer)
    ans_correct_score = None
    ans_correct_sim   = None
    if answer_text.strip() and expected_answer and expected_answer.strip():
        vecs = np.asarray(
            embedding_model.encode(
                [answer_text, expected_answer], normalize_embeddings=True
            ),
            dtype="float32",
        )
        ans_correct_sim   = cosine_similarity(vecs[0], vecs[1])
        ans_correct_score = similarity_to_score(ans_correct_sim, ANSWER_CORRECTNESS_THRESHOLDS)

    # 4. Answer Relevance (BGE-M3 cosine vs question)
    ans_rel_score = None
    ans_rel_sim   = None
    if answer_text.strip():
        vecs = np.asarray(
            embedding_model.encode(
                [answer_text, question], normalize_embeddings=True
            ),
            dtype="float32",
        )
        ans_rel_sim   = cosine_similarity(vecs[0], vecs[1])
        ans_rel_score = similarity_to_score(ans_rel_sim, ANSWER_RELEVANCE_THRESHOLDS)

    # 5. Completeness (punctuation + length ratio, floor AND ceiling)
    comp_score, ends_cleanly, length_ratio = score_completeness(
        answer_text, expected_answer
    )

    scores = {
        "context_relevance":  ctx_rel,
        "faithfulness":       faith_score,
        "answer_correctness": ans_correct_score,
        "answer_relevance":   ans_rel_score,
        "completeness":       comp_score,
    }

    return {
        "scores":   scores,
        "combined": weighted_combined(scores),
        "detail": {
            "faithfulness_grounded":         faith_grounded,
            "faithfulness_total_numbers":    faith_total,
            "faithfulness_ungrounded":       faith_ungrounded,
            "answer_correctness_similarity": ans_correct_sim,
            "answer_relevance_similarity":   ans_rel_sim,
            "completeness_ends_cleanly":     ends_cleanly,
            "completeness_length_ratio":     length_ratio,
        },
    }


# ----------------------------------------------------------------------------
# Real-time result file helpers
# ----------------------------------------------------------------------------
def model_dir(label):
    for info in LLM_MODELS.values():
        if info["label"] == label:
            return REPORTS_DIR / Path(info["local_filename"]).stem

    raise ValueError(f"Unknown model label: {label}")

def result_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / "generated_answer_results.jsonl"


def summary_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / "generated_answer_results_summary.json"


def load_existing_results(label):
    """Loads already-completed answers for this model so the run can
    resume from where it left off if interrupted."""
    path = result_path(label)
    if not path.exists():
        return {}
    done = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                done[entry["question"]] = entry
            except json.JSONDecodeError:
                pass
    return done


def append_result(label, entry):
    """Appends one completed answer to the per-model JSONL file immediately."""
    path = result_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Per-model evaluation loop
# ----------------------------------------------------------------------------

def evaluate_model(
    client,
    model_name,
    label,
    gold_entries,
    faiss_index,
    chunks,
    bm25_index,
    embedding_model,
    dry_run=False,
):
    # BUG FIX: dry-run must ALWAYS run the first question fresh regardless of
    # what's already saved -- its purpose is to verify current model behaviour,
    # not to replay a cached result. The resume logic only applies to full runs.
    if dry_run:
        existing = {}
        entries  = gold_entries[:1]
        total    = 1
    else:
        existing = load_existing_results(label)
        resumed  = len(existing)
        if resumed:
            print(f"  Resuming: {resumed} answers already saved, skipping those questions.")
        entries = gold_entries
        total   = len(gold_entries)

    results = list(existing.values())

    for i, entry in enumerate(entries, start=1):
        question        = entry["question"]
        expected_chunks = entry.get("expected_chunks", [])
        expected_answer = entry.get("expected_answer", "")
        category        = entry["category"]

        # Skip already-done questions (full run resume only -- dry_run always runs)
        if not dry_run and question in existing:
            print(f"  [{i}/{total}] Skipping (already done): {question[:60]}...")
            continue

        print(f"  [{i}/{total}] {category} | {question[:70]}...")

        # --- Retrieval (hybrid, mirrors app.py) ---
        query_emb = np.asarray(
            embedding_model.encode([question], normalize_embeddings=True),
            dtype="float32",
        )
        faiss_scores, faiss_indices = faiss_index.search(query_emb, k=TOP_K)
        dense_results = [
            (float(faiss_scores[0][j]), chunks[faiss_indices[0][j]])
            for j in range(len(faiss_indices[0]))
            if faiss_indices[0][j] >= 0
        ]

        bm25_tokens  = khmer_aware_tokenize(question)
        bm25_scores  = bm25_index.get_scores(bm25_tokens)
        bm25_top     = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
        bm25_results = [(float(bm25_scores[j]), chunks[j]) for j in bm25_top]

        fused             = reciprocal_rank_fusion([dense_results, bm25_results])
        context_chunks    = select_context_chunks(fused)
        context           = build_context(context_chunks)
        context_chunk_ids = [c["chunk_id"] for _, c in context_chunks]

        # --- Generation ---
        prompt = build_prompt(question, context)
        try:
            answer_text, first_token_time, generation_time, tokens_out = generate_answer(
                client, model_name, prompt
            )
        except Exception as e:
            print(f"    Generation failed: {e}")
            entry_out = {
                "question":         question,
                "category":         category,
                "expected_chunks":  expected_chunks,
                "expected_answer":  expected_answer,
                "generated_answer": "",
                "error":            str(e),
                "scores":           {},
                "combined":         0.0,
                "detail":           {},
                "timing":           {},
            }
            append_result(label, entry_out)
            results.append(entry_out)
            continue

        # Dry-run: print raw answer and skip scoring/saving
        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"MODEL    : {label}")
            print(f"QUESTION : {question}")
            print(f"ANSWER   :\n{answer_text}")
            print(
                f"FIRST TOKEN : {first_token_time:.2f}s | "
                f"TOTAL : {generation_time:.2f}s | "
                f"TOKENS : {tokens_out}"
            )
            print(f"{'=' * 60}\n")
            continue

        # --- Scoring ---
        scored = score_answer(
            answer_text,
            question,
            expected_answer,
            expected_chunks,
            context_chunk_ids,
            context,
            embedding_model,
        )

        decode_time  = max(generation_time - first_token_time, 0.001)
        tokens_per_s = tokens_out / decode_time if tokens_out > 0 else 0.0

        entry_out = {
            "question":          question,
            "category":          category,
            "expected_chunks":   expected_chunks,
            "expected_answer":   expected_answer,
            "generated_answer":  answer_text,
            "context_chunk_ids": context_chunk_ids,
            "scores":            scored["scores"],
            "combined":          scored["combined"],
            "detail":            scored["detail"],
            "timing": {
                "first_token_time": round(first_token_time, 4),
                "generation_time":  round(generation_time,  4),
                "tokens_out":       tokens_out,
                "tokens_per_sec":   round(tokens_per_s,     4),
            },
        }

        # Write to disk immediately -- real-time persistence
        append_result(label, entry_out)
        results.append(entry_out)

        print(
            f"    combined={scored['combined']:.3f} | "
            f"ctx={scored['scores'].get('context_relevance')} "
            f"faith={scored['scores'].get('faithfulness')} "
            f"corr={scored['scores'].get('answer_correctness')} "
            f"rel={scored['scores'].get('answer_relevance')} "
            f"comp={scored['scores'].get('completeness')} | "
            f"ttft={first_token_time:.1f}s tok/s={tokens_per_s:.1f}"
        )

    return results


# ----------------------------------------------------------------------------
# Summarise results for one model
# ----------------------------------------------------------------------------

def summarise(results, label):
    valid = [r for r in results if "error" not in r and r.get("scores")]

    def avg_metric(key):
        # Only include entries where the key is both present AND not None --
        # answer_correctness is None when expected_answer is missing.
        vals = [
            r["scores"][key]
            for r in valid
            if r["scores"].get(key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    def timing_vals(key):
        # Error entries have timing={} -- guard with .get() on both levels.
        return [
            r["timing"].get(key)
            for r in valid
            if r.get("timing") and r["timing"].get(key) is not None
        ]

    def avg_timing(key):
        vals = timing_vals(key)
        return round(sum(vals) / len(vals), 4) if vals else None

    def percentile(vals, p):
        """Returns the p-th percentile of a sorted list of floats."""
        if not vals:
            return None
        s = sorted(vals)
        idx = (p / 100) * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 4)

    scores_avg = {k: avg_metric(k) for k in WEIGHTS}

    # Combined average -- average the per-question combined scores directly
    # so it stays consistent with the normalised weighted_combined per entry.
    combined_vals = [r["combined"] for r in valid if r.get("combined") is not None]
    combined_avg  = round(sum(combined_vals) / len(combined_vals), 4) if combined_vals else 0.0

    # Timing detail
    ttft_vals     = timing_vals("first_token_time")
    gentime_vals  = timing_vals("generation_time")
    tokps_vals    = timing_vals("tokens_per_sec")
    tokout_vals   = [
        r["timing"].get("tokens_out")
        for r in valid
        if r.get("timing") and r["timing"].get("tokens_out") is not None
    ]

    timing_detail = {
        "first_token_time_avg": avg_timing("first_token_time"),
        "first_token_time_p50": percentile(ttft_vals, 50),
        "first_token_time_p95": percentile(ttft_vals, 95),
        "generation_time_avg":  avg_timing("generation_time"),
        "tokens_per_sec_avg":   avg_timing("tokens_per_sec"),
        "tokens_out_avg":       round(sum(tokout_vals) / len(tokout_vals), 1) if tokout_vals else None,
        "tokens_out_min":       min(tokout_vals) if tokout_vals else None,
        "tokens_out_max":       max(tokout_vals) if tokout_vals else None,
    }

    # Per-category breakdown -- scores AND timing so you can see if
    # certain document categories are consistently slower (e.g. laws
    # questions tend to have longer legal-unit chunks → bigger context
    # → slower prefill → higher TTFT).
    categories   = sorted({r["category"] for r in valid})
    per_category = {}
    for cat in categories:
        cat_valid = [r for r in valid if r["category"] == cat]

        cat_scores = {}
        for k in WEIGHTS:
            cat_vals = [
                r["scores"][k]
                for r in cat_valid
                if r["scores"].get(k) is not None
            ]
            cat_scores[k] = (
                round(sum(cat_vals) / len(cat_vals), 4) if cat_vals else None
            )

        cat_ttft = [
            r["timing"].get("first_token_time")
            for r in cat_valid
            if r.get("timing") and r["timing"].get("first_token_time") is not None
        ]
        cat_gentime = [
            r["timing"].get("generation_time")
            for r in cat_valid
            if r.get("timing") and r["timing"].get("generation_time") is not None
        ]
        cat_tokout = [
            r["timing"].get("tokens_out")
            for r in cat_valid
            if r.get("timing") and r["timing"].get("tokens_out") is not None
        ]

        per_category[cat] = {
            "scores": cat_scores,
            "timing": {
                "first_token_time_avg": round(sum(cat_ttft) / len(cat_ttft), 4) if cat_ttft else None,
                "generation_time_avg":  round(sum(cat_gentime) / len(cat_gentime), 4) if cat_gentime else None,
                "tokens_out_avg":       round(sum(cat_tokout) / len(cat_tokout), 1) if cat_tokout else None,
            },
        }

    truncated_count = sum(
        1 for r in valid
        if r.get("detail", {}).get("completeness_ends_cleanly") is False
    )

    summary = {
        "label":             label,
        "total_questions":   len(results),
        "valid_questions":   len(valid),
        "failed_questions":  len(results) - len(valid),
        "truncated_answers": truncated_count,
        "scores_avg":        scores_avg,
        "combined_avg":      combined_avg,
        "per_category":      per_category,
        # Keep timing_avg as a flat dict for backward compat with anything
        # that reads the summary JSON -- timing_detail has the full picture.
        "timing_avg": {
            "first_token_time": timing_detail["first_token_time_avg"],
            "generation_time":  timing_detail["generation_time_avg"],
            "tokens_per_sec":   timing_detail["tokens_per_sec_avg"],
        },
        "timing_detail": timing_detail,
    }

    # Save summary JSON for this model immediately
    path = summary_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ----------------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------------

def print_model_summary(summary):
    from tabulate import tabulate

    banner = "=" * SEPARATOR_WIDTH
    print(f"\n{banner}")
    print(f"{BOLD}GENERATOR: {summary['label']}{RESET}")
    print(banner)

    # --- Quality scores ---
    score_rows = [
        [k.replace("_", " ").title(), f"{v:.4f}" if v is not None else "N/A"]
        for k, v in summary["scores_avg"].items()
    ]
    score_rows.append(["── Combined (weighted) ──", f"{summary['combined_avg']:.4f}"])

    print(f"\n{SEPARATOR}")
    print(f"\t\t{BOLD}PROXY METRIC AVERAGES (1-5 scale){RESET}")
    print(SEPARATOR)
    print(tabulate(score_rows, headers=["Metric", "Avg Score"], tablefmt="grid"))

    # --- Latency detail ---
    td = summary.get("timing_detail", {})

    def fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    print(f"\n{SEPARATOR}")
    print(f"\t\t\t{BOLD}LATENCY & THROUGHPUT{RESET}")
    print(SEPARATOR)
    print(tabulate(
        [
            ["Avg first-token time (s)",    fmt(td.get("first_token_time_avg"), 2)],
            ["P50 first-token time (s)",    fmt(td.get("first_token_time_p50"), 2)],
            ["P95 first-token time (s)",    fmt(td.get("first_token_time_p95"), 2)],
            ["Avg total generation time (s)", fmt(td.get("generation_time_avg"), 2)],
            ["Avg decode speed (tok/s)",    fmt(td.get("tokens_per_sec_avg"), 2)],
            ["Avg tokens out",              fmt(td.get("tokens_out_avg"), 1)],
            ["Min tokens out",              str(td.get("tokens_out_min", "N/A"))],
            ["Max tokens out",              str(td.get("tokens_out_max", "N/A"))],
            ["Truncated answers",           summary["truncated_answers"]],
            ["Valid / Failed / Total",
             f"{summary['valid_questions']} / {summary['failed_questions']} / {summary['total_questions']}"],
        ],
        tablefmt="grid",
    ))

    # --- Per-category timing breakdown ---
    print(f"\n{SEPARATOR}")
    print(f"\t\t{BOLD}PER-CATEGORY TIMING (avg, seconds){RESET}")
    print(SEPARATOR)
    cat_timing_rows = []
    for cat, data in sorted(summary["per_category"].items()):
        t = data.get("timing", {})
        cat_timing_rows.append([
            cat,
            fmt(t.get("first_token_time_avg"), 2),
            fmt(t.get("generation_time_avg"),  2),
            fmt(t.get("tokens_out_avg"),        1),
        ])
    print(tabulate(
        cat_timing_rows,
        headers=["Category", "Avg TTFT (s)", "Avg Gen Time (s)", "Avg Tokens Out"],
        tablefmt="grid",
    ))


def print_final_comparison(summaries):
    from tabulate import tabulate

    banner = "=" * SEPARATOR_WIDTH
    print(f"\n\n{banner}")
    print(f"{BOLD}FINAL COMPARISON -- ALL GENERATORS{RESET}")
    print(f"{BOLD}(macro-avg across all 100 questions){RESET}")
    print(banner)

    labels = [s["label"] for s in summaries]

    def fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    # --- Quality metrics ---
    metric_keys = list(WEIGHTS.keys()) + ["combined_avg"]
    metric_display = {
        "context_relevance":  "Context Relevance",
        "faithfulness":       "Faithfulness",
        "answer_correctness": "Answer Correctness",
        "answer_relevance":   "Answer Relevance",
        "completeness":       "Completeness",
        "combined_avg":       "── Combined (weighted) ──",
    }

    rows = []
    for key in metric_keys:
        row = [metric_display[key]]
        for s in summaries:
            if key == "combined_avg":
                row.append(fmt(s["combined_avg"], 4))
            else:
                v = s["scores_avg"].get(key)
                row.append(fmt(v, 4) if v is not None else "N/A")
        rows.append(row)

    # --- Latency metrics ---
    rows.append(
        ["── Avg TTFT (s) ──"] + [
            fmt(s.get("timing_detail", {}).get("first_token_time_avg"))
            for s in summaries
        ]
    )
    rows.append(
        ["── P50 TTFT (s) ──"] + [
            fmt(s.get("timing_detail", {}).get("first_token_time_p50"))
            for s in summaries
        ]
    )
    rows.append(
        ["── P95 TTFT (s) ──"] + [
            fmt(s.get("timing_detail", {}).get("first_token_time_p95"))
            for s in summaries
        ]
    )
    rows.append(
        ["── Avg gen time (s) ──"] + [
            fmt(s.get("timing_detail", {}).get("generation_time_avg"))
            for s in summaries
        ]
    )
    rows.append(
        ["── Avg tok/s ──"] + [
            fmt(s.get("timing_detail", {}).get("tokens_per_sec_avg"))
            for s in summaries
        ]
    )
    rows.append(
        ["── Avg tokens out ──"] + [
            fmt(s.get("timing_detail", {}).get("tokens_out_avg"), 1)
            for s in summaries
        ]
    )
    rows.append(
        ["── Truncated answers ──"] + [str(s["truncated_answers"]) for s in summaries]
    )

    headers = ["Metric"] + labels
    print(tabulate(
        rows,
        headers=headers,
        tablefmt="grid",
        colalign=("left",) + ("center",) * len(labels),
    ))

    # --- Ranking ---
    ranked = sorted(summaries, key=lambda s: s["combined_avg"], reverse=True)
    print(f"\n{banner}")
    print(f"{BOLD}RANKING (by weighted combined score){RESET}")
    print(banner)
    for i, s in enumerate(ranked, start=1):
        td   = s.get("timing_detail", {})
        tok  = td.get("tokens_per_sec_avg") or 0.0
        ttft = td.get("first_token_time_avg") or 0.0
        print(
            f"  {i}. {s['label']} -- "
            f"combined={s['combined_avg']:.4f} | "
            f"avg TTFT={ttft:.1f}s | "
            f"tok/s={tok:.2f} | "
            f"truncated={s['truncated_answers']}"
        )

    if len(ranked) > 1:
        gap = ranked[0]["combined_avg"] - ranked[1]["combined_avg"]
        if gap < 0.05:
            print(
                f"\n  NOTE: top two models are close (gap={gap:.4f}). "
                "Check TTFT, tok/s, and truncated counts as tiebreakers."
            )

    print(f"\n  {BOLD}RECOMMENDATION: {ranked[0]['label']}{RESET}")
    print(f"  → Use this model for human-scored evaluation in scripts/11_evaluate_rag.py")


def write_final_report(summaries):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ranked = sorted(summaries, key=lambda s: s["combined_avg"], reverse=True)

    payload = {
        "summaries":      summaries,
        "ranking":        ranked,
        "recommendation": ranked[0]["label"],
        "weights":        WEIGHTS,
        "note": (
            "Fully automated proxy metrics -- no human scoring. "
            "Context Relevance is exact (deterministic chunk-ID overlap). "
            "Faithfulness uses number-overlap against retrieved context. "
            "Answer Correctness and Answer Relevance use BGE-M3 cosine "
            "similarity, each with its OWN threshold set calibrated against "
            "this corpus's real distribution (see docs/SCORING_CALIBRATION.md). "
            "Completeness checks sentence-final punctuation and BOTH a floor "
            "and a ceiling on length ratio vs expected answer. "
            "weighted_combined is normalised by the sum of weights that had values, "
            "so missing metrics (e.g. answer_correctness when expected_answer is absent) "
            "do not artificially deflate the combined score. "
            "Run scripts/11_evaluate_rag.py on the winning model for human-verified RAGAS scores."
        ),
    }

    json_path = REPORTS_DIR / "generator_model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final JSON report : {json_path}")
    print(
        f"Per-model results       : "
        f"{REPORTS_DIR}/<model>/generated_answer_results.jsonl"
    )
    print(
        f"Per-model summaries     : "
        f"{REPORTS_DIR}/<model>/generated_answer_results_summary.json"
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    wall_clock_start = time.time()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run: python scripts/07_generate_embeddings.py"
        )

    print("Loading embedding model (BGE-M3)...")
    embedding_model = SentenceTransformer(EMBEDDING_HF_ID)

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(INDEX_PATH)

    print("Loading chunk metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Building BM25 index...")
    bm25_corpus = [
        khmer_aware_tokenize(c.get("embedding_text") or c["text"])
        for c in chunks
    ]
    bm25_index = BM25Okapi(bm25_corpus)

    # Load all gold questions across all categories
    gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))
    if not gold_files:
        raise ValueError("No gold_questions.json files found under data/08_evaluation/.")

    gold_entries = []
    for gf in gold_files:
        category = gf.parent.name
        with open(gf, "r", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            gold_entries.append({**q, "category": category})

    mode_label = (
        "DRY RUN (1 question per model)"
        if DRY_RUN
        else f"FULL RUN ({len(gold_entries)} questions per model)"
    )

    installed = {k: v for k, v in LLM_MODELS.items() if is_installed(v)}
    skipped   = [v["label"] for k, v in LLM_MODELS.items() if k not in installed]

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"  MODE    : {mode_label}")
    print(f"  MODELS  : {len(installed)} installed, {len(skipped)} skipped")
    if skipped:
        for lbl in skipped:
            print(f"    - {lbl} (not installed)")
    print(f"  OUTPUT  : {REPORTS_DIR}/")
    print(f"{'=' * SEPARATOR_WIDTH}\n")

    if not installed:
        raise SystemExit("No generator models installed. Run: python scripts/13_install_models.py")

    # Load any summaries already saved from a previous interrupted run so the
    # final comparison table is always built from ALL completed models, not just
    # the ones that finished in this particular invocation. Without this, if you
    # finish 3 models, crash mid-model-4, restart and finish model-4, the final
    # comparison would only show model-4 because all_summaries starts empty.
    all_summaries = []
    if not DRY_RUN:
        for info in installed.values():
            sp = summary_path(info["label"])
            if sp.exists():
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    all_summaries.append(saved)
                    print(f"  Loaded existing summary for: {info['label']}")
                except (json.JSONDecodeError, OSError):
                    pass

    already_done_labels = {s["label"] for s in all_summaries}

    for key, info in installed.items():
        label = info["label"]

        # Skip models that already have a complete summary saved -- all 100
        # questions done, summary written. This is the model-level resume: if
        # SeaLLM and Qwen are fully done and the script crashed during Gemma,
        # restarting skips SeaLLM and Qwen entirely and starts from Gemma.
        if not DRY_RUN and label in already_done_labels:
            existing = load_existing_results(label)
            if len(existing) >= len(gold_entries):
                print(f"\n  Skipping {label} -- all {len(existing)} questions already done.")
                continue

        print(f"\n{SEPARATOR}")
        print(f"{BOLD}Starting llama-server for: {label}{RESET}")
        print(SEPARATOR)

        process = start_llama_server(info)
        print(f"  llama-server ready.")

        try:
            client     = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
            model_resp = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
            model_name = model_resp.json()["data"][0]["id"]
            print(f"  Model ID  : {model_name}")
            print(f"  Questions : {1 if DRY_RUN else len(gold_entries)}")
            print()

            results = evaluate_model(
                client, model_name, label, gold_entries,
                faiss_index, chunks, bm25_index, embedding_model,
                dry_run=DRY_RUN,
            )

            if not DRY_RUN:
                summary = summarise(results, label)
                # Replace any stale in-memory entry for this label with the
                # freshly computed one (handles the case where an incomplete
                # summary was loaded at startup and the model just finished)
                all_summaries = [s for s in all_summaries if s["label"] != label]
                all_summaries.append(summary)
                print_model_summary(summary)

        finally:
            print(f"\n  Stopping llama-server for: {label}")
            stop_llama_server(process)

    if DRY_RUN:
        print(f"\n{'=' * SEPARATOR_WIDTH}")
        print(f"{BOLD}DRY RUN COMPLETE{RESET}")
        print("All models started, generated, and stopped successfully.")
        print("If the answers above look correct, run without --dry-run for the full evaluation.")
        print(f"{'=' * SEPARATOR_WIDTH}")
        return

    # Final comparison and report
    print_final_comparison(all_summaries)
    write_final_report(all_summaries)

    total_gen_time = sum(
        s["timing_avg"]["generation_time"] * s["total_questions"]
        for s in all_summaries
        if s["timing_avg"].get("generation_time")
    )

    wall_clock_total = time.time() - wall_clock_start
    hours, rem       = divmod(int(wall_clock_total), 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\nApprox total generation time : {total_gen_time / 60:.1f} minutes")
    print(f"Total wall-clock runtime     : {hours}h {minutes}m {seconds}s")
    print("Generator comparison completed successfully.")


if __name__ == "__main__":
    main()