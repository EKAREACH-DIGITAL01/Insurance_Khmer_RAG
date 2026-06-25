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
# human-scored RAGAS evaluation in scripts/13_human_answer_evaluation.py.
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
# BLANK ANSWER POLICY:
#   A blank generated_answer is NOT a free pass. When answer_text is empty:
#     - answer_correctness: cosine similarity = 0.0 → score = 1
#     - answer_relevance:   cosine similarity = 0.0 → score = 1
#   This is enforced in score_answer() by passing a zero-vector for the
#   generated answer when answer_text.strip() is empty, rather than
#   returning None (which previously caused weighted_combined() to exclude
#   those metrics from the denominator, artificially inflating the score
#   for models with blank answers).
#
#   Rationale: a model that produced no output has definitively failed to
#   answer the question. Setting similarity = 0.0 → score = 1 is the
#   correct representation: "worst possible on this metric." It is NOT the
#   same as "metric is unavailable" (None), which is reserved for cases
#   where the metric genuinely cannot be computed (e.g. expected_answer is
#   missing from the gold set entirely).
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
EVALUATION_ROOT  = Path("data/09_evaluation")
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

# Safety / scope gates -- mirror app.py EXACTLY so a gold question that the
# live app would reject (canned message, no LLM call) is rejected here the
# same way, instead of being silently generated for. See app.py for the
# full calibration rationale behind each threshold.
MIN_SCORE              = 0.50  # below this (but >= OUT_OF_DOMAIN_THRESHOLD) => low_confidence prompt
OUT_OF_DOMAIN_THRESHOLD = 0.49  # raw FAISS best_score below this => out-of-domain, no generation
MIN_QUESTION_LENGTH    = 8      # shorter questions are treated as degenerate input

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

# ----------------------------------------------------------------------------
# Scope / domain gates -- duplicated from app.py.
# If you update any of these in app.py, update them here too. They must stay
# byte-for-byte identical so script 10 rejects exactly the same gold
# questions the live UI rejects (and with the same canned wording).
# ----------------------------------------------------------------------------
REDIRECT_TEXT = (
    "សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC) "
    "សម្រាប់ជំនួយបន្ថែម។"
)

SCOPE_VIOLATION_PATTERNS = [
    re.compile(r"(សូម)?ណែនាំ.{0,30}(ក្រុមហ៊ុន|ផលិតផល|គម្រោង)"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,30}ល្អ(បំផុត|ជាង)"),
    re.compile(r"មួយណា.{0,15}ល្អ(ជាង|បំផុត)?"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}និយម.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ពេញនិយម"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ល្បី.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}កេរ្តិ៍ឈ្មោះ.{0,10}ល្អ"),
    re.compile(r"គួរ.{0,15}(ទិញ|ជ្រើសរើស|ប្រើ).{0,30}ក្រុមហ៊ុន"),
    re.compile(r"ខ្ញុំគួរ.{0,20}(ជ្រើសរើស|ទិញ|ប្រើ|មើល)"),
    re.compile(r"មិន.{0,10}(សុំ|ចង់បាន).{0,10}ប្រឹក្សា"),
    re.compile(r"បើអ្នកជាខ្ញុំ"),
    re.compile(r"ត្រូវ(បង់|ចំណាយ|ខាត).{0,10}ប៉ុន្មាន"),
    re.compile(r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន"),
]


def contains_khmer_script(text):
    return bool(KHMER_SCRIPT_PATTERN.search(text))


def is_scope_violation(text):
    return any(pattern.search(text) for pattern in SCOPE_VIOLATION_PATTERNS)


def gate_question(question, faiss_results=None):
    """Mirror of app.py's run_pipeline_in_thread gating, in app.py's exact
    order. Returns the canned answer string the live UI would show if the
    question is rejected, or None if it should proceed to generation.

    The pre-retrieval gates (Khmer-script, length, scope) need no scores.
    The out-of-domain score gate needs the raw FAISS results, so it is only
    applied when faiss_results is passed (after the FAISS search)."""
    out_of_domain_msg = (
        "សំណួររបស់អ្នកហាក់ដូចជាមិនទាក់ទងនឹងច្បាប់ ឬបទប្បញ្ញត្តិធានារ៉ាប់រងកម្ពុជាទេ។ "
        "ជំនួយការនេះអាចឆ្លើយបានតែសំណួរទាក់ទងនឹងច្បាប់ធានារ៉ាប់រងប៉ុណ្ណោះ។ "
        f"{REDIRECT_TEXT}"
    )
    scope_msg = (
        "សំណួររបស់អ្នកស្នើសុំការណែនាំ ឬប្រៀបធៀបក្រុមហ៊ុន/ផលិតផលជាក់លាក់ "
        "ការផ្តល់ប្រឹក្សាផ្ទាល់ខ្លួន ឬការគណនាប្រាក់ផ្ទាល់ខ្លួន ដែលជំនួយការនេះ"
        "មិនអាចផ្តល់បានទេ។ "
        f"{REDIRECT_TEXT}"
    )

    if faiss_results is None:
        # Pre-retrieval gates (same order as app.py)
        if not contains_khmer_script(question):
            return out_of_domain_msg, "not_khmer"
        if len(question.strip()) < MIN_QUESTION_LENGTH:
            return out_of_domain_msg, "too_short"
        if is_scope_violation(question):
            return scope_msg, "scope_violation"
        return None
    else:
        # Post-FAISS score gate (before BM25/fusion, exactly like app.py)
        best_score = float(faiss_results[0][0]) if faiss_results else 0.0
        if best_score < OUT_OF_DOMAIN_THRESHOLD:
            return out_of_domain_msg, "low_score"
        return None

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
# Prompt -- duplicated from app.py's build_prompt() (INCLUDING the
# low_confidence branch). If you update the prompt in app.py, update it here
# too -- it must stay byte-for-byte identical so the LLM sees the same prompt
# in both places. low_confidence is passed as best_score < MIN_SCORE, exactly
# as app.py does.
# ----------------------------------------------------------------------------

def build_prompt(question, context, low_confidence=False):
    relevance_check = ""
    if low_confidence:
        relevance_check = """
IMPORTANT: The retrieved context below has LOW relevance confidence for this
question. Before answering, first check whether the context is actually
ABOUT the same topic as the question -- not just whether it shares similar
words or phrasing. If the context discusses a different subject, or the
question is not really about Cambodian insurance law at all (for example,
a question about the assistant itself, or unrelated small talk), you MUST
reply with the "not found" message below instead of extracting a loosely
related answer.
"""

    not_found_message = (
        f"រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។ {REDIRECT_TEXT}"
    )

    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.
{relevance_check}
Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Provide a complete and detailed answer. If the context lists multiple conditions, sub-points, exceptions, or requirements, enumerate ALL of them.
- Only combine information from multiple sources if EACH of them directly addresses the question asked. Do not stitch together unrelated facts from different sources just because they were retrieved -- if most of the retrieved sources don't actually answer the question, say so instead of forcing a combined answer out of all of them.
- Do not reinterpret a number, rate, or statistic for a purpose other than what it was stated for in the source (for example, a mortality or incidence rate expressed "per 1,000 people" is NOT the same as a total population count -- never restate a rate's denominator as if it were the literal answer to a "how many" question).
- Do not recommend, endorse, or compare any specific insurance product, company, or policy, even if the context mentions one and the question asks for a recommendation.
- Do not calculate, estimate, or infer a premium, payout, deductible, or any monetary amount that is not stated explicitly and directly in the context for that exact purpose.
- Do not give personalized financial or legal advice (e.g. "should I buy X", "what should I do in my situation"). If the question asks for a personal recommendation, a calculation, or advice tailored to the user's situation, decline and reply with the message below instead of attempting an answer.
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

    A blank answer has no numbers to check -- returns neutral 4, consistent
    with the original design (no numbers = nothing to verify, not a
    fabrication). The blank-answer penalty is applied separately via
    answer_correctness and answer_relevance (both forced to score 1).
    """
    answer_numbers = set(NUMBER_PATTERN.findall(generated_answer))
    if not answer_numbers:
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

    A blank answer fails the punctuation check AND has length_ratio = 0.0,
    which falls into the truncated branch at score = 1 (length_ratio < 0.40).
    No special-casing needed -- the existing logic handles it correctly.
    """
    stripped     = generated_answer.strip()
    ends_cleanly = bool(stripped) and stripped[-1] in KHMER_END_PUNCT

    if not expected_answer or len(expected_answer) == 0:
        return (5 if ends_cleanly else 2), ends_cleanly, None

    length_ratio = len(stripped) / len(expected_answer)

    if ends_cleanly:
        if 0.80 <= length_ratio <= 2.00:
            score = 5
        elif (0.60 <= length_ratio < 0.80) or (2.00 < length_ratio <= 3.00):
            score = 4
        else:
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

    IMPORTANT: None is ONLY acceptable when a metric genuinely cannot be
    computed (e.g. expected_answer is absent from the gold set, making
    answer_correctness unmeasurable). It must NEVER be used to represent
    "the model produced no output" -- that case must be scored as 1 (worst)
    on the affected metrics before this function is called.

    When a metric is genuinely None (missing expected_answer), we normalise
    by the sum of weights that actually had values so the result stays on
    the [1, 5] scale.
    """
    total        = 0.0
    weight_sum   = 0.0
    for key, weight in WEIGHTS.items():
        val = scores.get(key)
        if val is not None:
            total      += val * weight
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
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

    BLANK ANSWER HANDLING:
    When answer_text is empty, answer_correctness and answer_relevance are
    scored as 1 (worst) with similarity = 0.0, rather than None. This
    ensures blank answers are penalised in the combined score rather than
    silently excluded from the denominator. The None path is reserved for
    cases where the metric is genuinely uncomputable (missing expected_answer).

    Concretely: we encode a single space " " as a stand-in for a blank
    answer. After L2 normalisation a near-zero vector will have cosine
    similarity ≈ 0.0 against any real answer or question embedding, which
    maps to score = 1 under both threshold sets. This avoids the need for
    special-case branching in weighted_combined().
    """
    is_blank = not answer_text.strip()

    # 1. Context Relevance (deterministic -- unaffected by blank answer)
    ctx_rel = score_context_relevance(context_chunk_ids, expected_chunks)

    # 2. Faithfulness (number overlap -- blank answer has no numbers, scores 4)
    faith_score, faith_grounded, faith_total, faith_ungrounded = score_faithfulness(
        answer_text, context
    )

    # 3. Answer Correctness (BGE-M3 cosine vs expected_answer)
    #    Blank answer: encode " " → similarity ≈ 0.0 → score = 1.
    #    Missing expected_answer: score = None (genuinely uncomputable).
    ans_correct_score = None
    ans_correct_sim   = None
    if expected_answer and expected_answer.strip():
        # Use " " as proxy for blank so cosine similarity resolves to ~0.0
        answer_for_embed = answer_text if not is_blank else " "
        vecs = np.asarray(
            embedding_model.encode(
                [answer_for_embed, expected_answer], normalize_embeddings=True
            ),
            dtype="float32",
        )
        ans_correct_sim   = cosine_similarity(vecs[0], vecs[1])
        # Blank: clamp to 0.0 explicitly (space vector can have tiny
        # non-zero dot product from tokenisation artefacts)
        if is_blank:
            ans_correct_sim = 0.0
        ans_correct_score = similarity_to_score(ans_correct_sim, ANSWER_CORRECTNESS_THRESHOLDS)

    # 4. Answer Relevance (BGE-M3 cosine vs question)
    #    Blank answer: encode " " → similarity ≈ 0.0 → score = 1.
    #    Always computable (question is always present).
    ans_rel_score = None
    ans_rel_sim   = None
    answer_for_embed = answer_text if not is_blank else " "
    vecs = np.asarray(
        embedding_model.encode(
            [answer_for_embed, question], normalize_embeddings=True
        ),
        dtype="float32",
    )
    ans_rel_sim   = cosine_similarity(vecs[0], vecs[1])
    if is_blank:
        ans_rel_sim = 0.0
    ans_rel_score = similarity_to_score(ans_rel_sim, ANSWER_RELEVANCE_THRESHOLDS)

    # 5. Completeness (punctuation + length ratio)
    #    Blank answer: ends_cleanly = False, length_ratio = 0.0 → score = 1.
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
            "blank_answer":                  is_blank,
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


def summary_md_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / "generated_answer_results_summary.md"


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

        if not dry_run and question in existing:
            print(f"  [{i}/{total}] Skipping (already done): {question[:60]}...")
            continue

        print(f"  [{i}/{total}] {category} | {question[:70]}...")

        # --- Pre-retrieval gates (mirror app.py: Khmer-script, length, scope).
        # A gated question never reaches the LLM in the live app -- it returns
        # a fixed canned message. We record that exact canned message as the
        # "generated_answer" so the evaluation reflects what the UI would
        # actually answer for this gold question. ---
        gate_reason = None
        gated = gate_question(question)
        if gated is not None:
            answer_text, gate_reason = gated
            context           = ""
            context_chunk_ids = []
            first_token_time  = 0.0
            generation_time   = 0.0
            tokens_out        = 0
        else:
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

            # --- Out-of-domain score gate (mirror app.py: BEFORE BM25/fusion,
            # computed from raw FAISS best_score only) ---
            gated = gate_question(question, faiss_results=dense_results)
            if gated is not None:
                answer_text, gate_reason = gated
                context           = ""
                context_chunk_ids = []
                first_token_time  = 0.0
                generation_time   = 0.0
                tokens_out        = 0
            else:
                best_score = float(dense_results[0][0]) if dense_results else 0.0

                bm25_tokens  = khmer_aware_tokenize(question)
                bm25_scores  = bm25_index.get_scores(bm25_tokens)
                bm25_top     = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
                bm25_results = [(float(bm25_scores[j]), chunks[j]) for j in bm25_top]

                fused             = reciprocal_rank_fusion([dense_results, bm25_results])
                context_chunks    = select_context_chunks(fused)
                context           = build_context(context_chunks)
                context_chunk_ids = [c["chunk_id"] for _, c in context_chunks]

                # --- Generation ---
                # low_confidence mirrors app.py exactly: best_score < MIN_SCORE
                # (but >= OUT_OF_DOMAIN_THRESHOLD, since lower already gated out)
                prompt = build_prompt(
                    question, context, low_confidence=best_score < MIN_SCORE
                )
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

        if gate_reason is not None:
            print(f"    GATED ({gate_reason}) -- canned answer, no LLM call.")

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
        # Note: score_answer() now handles blank answer_text correctly --
        # blank produces score=1 for answer_correctness and answer_relevance
        # (not None), so weighted_combined() includes them in the denominator.
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
            "gate_reason":       gate_reason,
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

        append_result(label, entry_out)
        results.append(entry_out)

        blank_flag = " [BLANK]" if scored["detail"].get("blank_answer") else ""
        print(
            f"    combined={scored['combined']:.3f} | "
            f"ctx={scored['scores'].get('context_relevance')} "
            f"faith={scored['scores'].get('faithfulness')} "
            f"corr={scored['scores'].get('answer_correctness')} "
            f"rel={scored['scores'].get('answer_relevance')} "
            f"comp={scored['scores'].get('completeness')} | "
            f"ttft={first_token_time:.1f}s tok/s={tokens_per_s:.1f}"
            f"{blank_flag}"
        )

    return results


# ----------------------------------------------------------------------------
# Summarise results for one model
# ----------------------------------------------------------------------------

def summarise(results, label):
    valid = [r for r in results if "error" not in r and r.get("scores")]

    def avg_metric(key):
        # None means genuinely uncomputable (missing expected_answer).
        # Score=1 from blank answers is a real numeric value, not None,
        # so it IS included in this average -- that's the intended behaviour.
        vals = [
            r["scores"][key]
            for r in valid
            if r["scores"].get(key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    def timing_vals(key):
        return [
            r["timing"].get(key)
            for r in valid
            if r.get("timing") and r["timing"].get(key) is not None
        ]

    def avg_timing(key):
        vals = timing_vals(key)
        return round(sum(vals) / len(vals), 4) if vals else None

    def percentile(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        idx = (p / 100) * (len(s) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 4)

    scores_avg = {k: avg_metric(k) for k in WEIGHTS}

    combined_vals = [r["combined"] for r in valid if r.get("combined") is not None]
    combined_avg  = round(sum(combined_vals) / len(combined_vals), 4) if combined_vals else 0.0

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

    blank_count = sum(
        1 for r in valid
        if r.get("detail", {}).get("blank_answer") is True
    )

    summary = {
        "label":             label,
        "total_questions":   len(results),
        "valid_questions":   len(valid),
        "failed_questions":  len(results) - len(valid),
        "truncated_answers": truncated_count,
        "blank_answers":     blank_count,
        "scores_avg":        scores_avg,
        "combined_avg":      combined_avg,
        "per_category":      per_category,
        "timing_avg": {
            "first_token_time": timing_detail["first_token_time_avg"],
            "generation_time":  timing_detail["generation_time_avg"],
            "tokens_per_sec":   timing_detail["tokens_per_sec_avg"],
        },
        "timing_detail": timing_detail,
    }

    path = summary_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write per-model markdown summary
    write_model_summary_md(summary, label)

    return summary


# ----------------------------------------------------------------------------
# Per-model markdown report
# ----------------------------------------------------------------------------

def write_model_summary_md(summary, label):
    """Writes a human-readable markdown summary for a single model.
    Mirrors the terminal print_model_summary() output but in a persistent
    file, consistent with how scripts/08 and scripts/09 each produce a .md
    alongside their .json.
    """
    from tabulate import tabulate

    def fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    td = summary.get("timing_detail", {})

    lines = []
    lines.append(f"# Generator Evaluation Summary: {label}\n\n")

    # -- Quality scores --
    score_rows = [
        [k.replace("_", " ").title(), fmt(v, 4) if v is not None else "N/A"]
        for k, v in summary["scores_avg"].items()
    ]
    score_rows.append(["**Combined (weighted)**", f"**{summary['combined_avg']:.4f}**"])
    lines.append("## Proxy Metric Averages (1–5 scale)\n\n")
    lines.append(
        "> Blank answers score 1 on Answer Correctness and Answer Relevance "
        "(not excluded from the average).\n\n"
    )
    lines.append(tabulate(score_rows, headers=["Metric", "Avg Score"], tablefmt="github"))
    lines.append("\n\n")

    # -- Coverage --
    lines.append("## Coverage\n\n")
    lines.append(tabulate(
        [
            ["Total questions",   summary["total_questions"]],
            ["Valid (scored)",    summary["valid_questions"]],
            ["Failed (error)",    summary["failed_questions"]],
            ["Blank answers",     summary["blank_answers"]],
            ["Truncated answers", summary["truncated_answers"]],
        ],
        tablefmt="github",
    ))
    lines.append("\n\n")

    # -- Latency --
    lines.append("## Latency & Throughput\n\n")
    lines.append(tabulate(
        [
            ["Avg first-token time (s)",       fmt(td.get("first_token_time_avg"))],
            ["P50 first-token time (s)",       fmt(td.get("first_token_time_p50"))],
            ["P95 first-token time (s)",       fmt(td.get("first_token_time_p95"))],
            ["Avg total generation time (s)",  fmt(td.get("generation_time_avg"))],
            ["Avg decode speed (tok/s)",       fmt(td.get("tokens_per_sec_avg"))],
            ["Avg tokens out",                 fmt(td.get("tokens_out_avg"), 1)],
            ["Min tokens out",                 str(td.get("tokens_out_min", "N/A"))],
            ["Max tokens out",                 str(td.get("tokens_out_max", "N/A"))],
        ],
        tablefmt="github",
    ))
    lines.append("\n\n")

    # -- Per-category scores --
    lines.append("## Per-Category Scores\n\n")
    metric_keys = list(WEIGHTS.keys())
    cat_rows = []
    for cat, data in sorted(summary["per_category"].items()):
        row = [cat]
        for k in metric_keys:
            v = data["scores"].get(k)
            row.append(fmt(v, 4) if v is not None else "N/A")
        cat_rows.append(row)
    lines.append(tabulate(
        cat_rows,
        headers=["Category"] + [k.replace("_", " ").title() for k in metric_keys],
        tablefmt="github",
    ))
    lines.append("\n\n")

    # -- Per-category timing --
    lines.append("## Per-Category Timing (avg, seconds)\n\n")
    cat_timing_rows = []
    for cat, data in sorted(summary["per_category"].items()):
        t = data.get("timing", {})
        cat_timing_rows.append([
            cat,
            fmt(t.get("first_token_time_avg")),
            fmt(t.get("generation_time_avg")),
            fmt(t.get("tokens_out_avg"), 1),
        ])
    lines.append(tabulate(
        cat_timing_rows,
        headers=["Category", "Avg TTFT (s)", "Avg Gen Time (s)", "Avg Tokens Out"],
        tablefmt="github",
    ))
    lines.append("\n")

    md_path = summary_md_path(label)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"  Saved model markdown : {md_path}")


# ----------------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------------

def print_model_summary(summary):
    from tabulate import tabulate

    banner = "=" * SEPARATOR_WIDTH
    print(f"\n{banner}")
    print(f"{BOLD}GENERATOR: {summary['label']}{RESET}")
    print(banner)

    score_rows = [
        [k.replace("_", " ").title(), f"{v:.4f}" if v is not None else "N/A"]
        for k, v in summary["scores_avg"].items()
    ]
    score_rows.append(["── Combined (weighted) ──", f"{summary['combined_avg']:.4f}"])

    print(f"\n{SEPARATOR}")
    print(f"\t\t{BOLD}PROXY METRIC AVERAGES (1-5 scale){RESET}")
    print(SEPARATOR)
    print(tabulate(score_rows, headers=["Metric", "Avg Score"], tablefmt="grid"))

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
            ["Blank answers",               summary["blank_answers"]],
            ["Truncated answers",           summary["truncated_answers"]],
            ["Valid / Failed / Total",
             f"{summary['valid_questions']} / {summary['failed_questions']} / {summary['total_questions']}"],
        ],
        tablefmt="grid",
    ))

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
        ["── Blank answers ──"] + [str(s["blank_answers"]) for s in summaries]
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
            f"blank={s['blank_answers']} | "
            f"truncated={s['truncated_answers']}"
        )

    if len(ranked) > 1:
        gap = ranked[0]["combined_avg"] - ranked[1]["combined_avg"]
        if gap < 0.05:
            print(
                f"\n  NOTE: top two models are close (gap={gap:.4f}). "
                "Check TTFT, tok/s, blank and truncated counts as tiebreakers."
            )

    print(f"\n  {BOLD}RECOMMENDATION: {ranked[0]['label']}{RESET}")
    print(f"  → Use this model for human-scored evaluation in scripts/13_human_answer_evaluation.py")


def write_final_report(summaries):
    from tabulate import tabulate

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
            "BLANK ANSWERS score 1 on Answer Correctness and Answer Relevance "
            "(not excluded from averages). "
            "weighted_combined is normalised by the sum of weights that had "
            "values -- None is reserved for genuinely uncomputable metrics "
            "(missing expected_answer), not for blank generated answers. "
            "Run scripts/13_human_answer_evaluation.py on the winning model "
            "for human-verified RAGAS scores."
        ),
    }

    json_path = REPORTS_DIR / "generator_model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- Top-level markdown comparison report ---
    labels = [s["label"] for s in summaries]

    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    metric_keys = list(WEIGHTS.keys())
    metric_display = {
        "context_relevance":  "Context Relevance",
        "faithfulness":       "Faithfulness",
        "answer_correctness": "Answer Correctness",
        "answer_relevance":   "Answer Relevance",
        "completeness":       "Completeness",
    }

    score_rows = []
    for key in metric_keys:
        row = [metric_display[key]]
        for s in summaries:
            v = s["scores_avg"].get(key)
            row.append(fmt(v) if v is not None else "N/A")
        score_rows.append(row)
    combined_row = ["**Combined (weighted)**"] + [f"**{fmt(s['combined_avg'])}**" for s in summaries]
    score_rows.append(combined_row)

    latency_rows = []
    for label in ["── Avg TTFT (s) ──", "── P50 TTFT (s) ──", "── P95 TTFT (s) ──",
                  "── Avg gen time (s) ──", "── Avg tok/s ──", "── Blank answers ──",
                  "── Truncated answers ──"]:
        td_key_map = {
            "── Avg TTFT (s) ──": ("first_token_time_avg", 2),
            "── P50 TTFT (s) ──": ("first_token_time_p50", 2),
            "── P95 TTFT (s) ──": ("first_token_time_p95", 2),
            "── Avg gen time (s) ──": ("generation_time_avg", 2),
            "── Avg tok/s ──": ("tokens_per_sec_avg", 2),
        }
        if label in td_key_map:
            td_key, dec = td_key_map[label]
            row = [label] + [fmt(s.get("timing_detail", {}).get(td_key), dec) for s in summaries]
        elif "Blank" in label:
            row = [label] + [str(s["blank_answers"]) for s in summaries]
        else:
            row = [label] + [str(s["truncated_answers"]) for s in summaries]
        latency_rows.append(row)

    headers = ["Metric"] + labels

    md_lines = []
    md_lines.append("# Generator Model Comparison Report\n\n")
    md_lines.append(
        "> Blank answers score **1** on Answer Correctness and Answer Relevance "
        "(not excluded from averages). See `docs/SCORING_CALIBRATION.md`.\n\n"
    )

    md_lines.append("## Quality Scores (macro-avg, 1–5 scale)\n\n")
    md_lines.append(tabulate(score_rows, headers=headers, tablefmt="github"))
    md_lines.append("\n\n")

    md_lines.append("## Latency & Coverage\n\n")
    md_lines.append(tabulate(latency_rows, headers=headers, tablefmt="github"))
    md_lines.append("\n\n")

    md_lines.append("## Ranking\n\n")
    for i, s in enumerate(ranked, start=1):
        td = s.get("timing_detail", {})
        md_lines.append(
            f"{i}. **{s['label']}** — combined={s['combined_avg']:.4f} | "
            f"blank={s['blank_answers']} | "
            f"avg TTFT={fmt(td.get('first_token_time_avg'))}s\n"
        )
    md_lines.append(f"\n**Recommendation: {ranked[0]['label']}**\n\n")
    md_lines.append(
        f"Use this model for human-scored evaluation in "
        f"`scripts/13_human_answer_evaluation.py`.\n"
    )

    md_path = REPORTS_DIR / "generator_model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print(f"\nSaved final JSON report    : {json_path}")
    print(f"Saved final markdown report: {md_path}")
    print(
        f"Per-model results          : "
        f"{REPORTS_DIR}/<model>/generated_answer_results.jsonl"
    )
    print(
        f"Per-model summaries (JSON) : "
        f"{REPORTS_DIR}/<model>/generated_answer_results_summary.json"
    )
    print(
        f"Per-model summaries (MD)   : "
        f"{REPORTS_DIR}/<model>/generated_answer_results_summary.md"
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

    gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))
    if not gold_files:
        raise ValueError("No gold_questions.json files found under data/09_evaluation/.")

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