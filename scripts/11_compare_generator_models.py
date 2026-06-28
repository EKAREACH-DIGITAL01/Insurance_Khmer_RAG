import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

# macOS OpenMP guard -- MUST be set before importing faiss or torch (pulled in
# by sentence_transformers below). FAISS and PyTorch each load their own
# OpenMP runtime; on macOS the duplicate runtimes collide and SEGFAULT the
# instant the first real FAISS search + BGE-M3 encode run concurrently.
# KMP_DUPLICATE_LIB_OK lets the duplicate load proceed, and pinning OpenMP to a
# single thread removes the cross-runtime thread contention that triggers the
# crash. app.py sets exactly these at its top for the same reason -- keep them
# in sync.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss

faiss.omp_set_num_threads(1)

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
parser.add_argument(
    "--rescore",
    action="store_true",
    help=(
        "Do NOT generate anything. Reload every model's existing "
        "generated_answer_results(.jsonl) / _full.jsonl, recompute all metrics "
        "with the current scoring functions (rebuilding each answer's context "
        "from its saved context_chunk_ids), rewrite the JSONL in place, and "
        "regenerate every summary + the final comparison report. Use this to "
        "re-judge the already-generated answers for free after a scoring change "
        "-- it never starts llama-server."
    ),
)
args = parser.parse_args()
DRY_RUN = args.dry_run
RESCORE = args.rescore

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
MAX_CHUNKS_PER_DOC = 2
BM25_TOP_K        = TOP_K
RRF_K             = 60
NGRAM_SIZE        = 3
# Output-token cap. NOTE: unlike app.py (which uses a FIXED 1024 in production
# because it has no expected_answer to size against), this eval sizes the cap
# PER QUESTION to the expected answer's real token length, then clamps it
# between a floor and a ceiling. Rationale:
#   - Khmer tokenizes very inefficiently in these GGUF models (byte-fallback
#     splits each cluster into multiple tokens), so a long expected answer can
#     need well over 1024 output tokens -- a flat 1024 silently TRUNCATES them.
#   - On this hardware a larger cap measurably raises TTFT (the server reserves
#     KV/compute buffers up front), so a flat 2048 needlessly taxes the short
#     majority. Sizing to each answer gives every question the SMALLEST cap that
#     still fits its content -> minimum TTFT per question.
#   - The cap is identical for a given question only if tokenizers match; it is
#     computed with the CURRENTLY LOADED model's own tokenizer (via the live
#     server's /tokenize), so each model gets exactly the room its tokenizer
#     needs for the same content -- the fair basis for a model-vs-model compare.
# This is DELIBERATELY a divergence from app.py's fixed 1024: it trades
# production-truncation fidelity for not truncating the gold answers mid-eval.
MAX_TOKENS_FLOOR   = 1024  # never go below this, even for one-line answers
GEN_TOKEN_HEADROOM = 1.5   # the generated answer often runs longer than the
                           # reference; give it 50% slack plus room for EOS
MAX_TOKENS         = MAX_TOKENS_FLOOR  # fallback when token count is unavailable

# The ceiling is NOT a fixed guess. It only needs to cover the longest answer
# the eval actually generates against -- the longest expected_answer in the GOLD
# SET (not the longest corpus chunk; most chunks are never an expected answer).
# compute_gold_ceiling() tokenizes the longest gold answers with the currently
# loaded model's own tokenizer and sets the ceiling to (max gold-answer tokens
# x headroom) -- recomputed per model, since tokenizers differ. It is then
# bounded by SERVER_OUTPUT_HARD_CAP so prompt + output can never exceed the
# server's context window (which would trigger slow context-shifting /
# truncation).
SERVER_CTX_SIZE        = 16384  # must match start_chatbot.py / start_llama_server
PROMPT_TOKEN_MARGIN    = 256    # safety gap so prompt + max_tokens < ctx size
# Absolute upper bound on output tokens, independent of any single prompt:
# leaves room for a large (~9k) retrieval prompt within the context window.
# Per question we tighten this further using that question's real prompt length.
SERVER_OUTPUT_HARD_CAP = SERVER_CTX_SIZE - 9000  # ~7384 tokens of headroom
MAX_TOKENS_CEILING_FALLBACK = 2048  # used only if corpus tokenization fails

# Repetition penalty -- MUST stay identical to app.py's GEN_FREQUENCY_PENALTY
# (see that file for the full rationale). Small quantized models can fall into a
# degenerate repetition loop (the same sentence emitted until max_tokens);
# frequency_penalty breaks the loop while sparing normal legal-term reuse. Kept
# in sync so eval generation matches production generation exactly.
GEN_FREQUENCY_PENALTY = 0.3

# Transient generation failures: with --jinja, llama-server parses the model's
# streamed output through the chat-template PEG parser. When the model samples a
# token sequence the parser can't reconcile, the request aborts with errors like
# "does not match the expected peg-native format". It's stochastic (a re-request
# usually succeeds), so we re-roll a few times before recording a hard failure
# -- the automated equivalent of "delete the row and rerun".
GEN_MAX_RETRIES   = 3    # total attempts per question before giving up
GEN_RETRY_BACKOFF = 2.0  # seconds between attempts

# Safety / scope gates -- mirror app.py EXACTLY so a gold question that the
# live app would reject (canned message, no LLM call) is rejected here the
# same way, instead of being silently generated for. See app.py for the
# full calibration rationale behind each threshold.
MIN_SCORE              = 0.50  # below this (but >= OUT_OF_DOMAIN_THRESHOLD) => low_confidence prompt
OUT_OF_DOMAIN_THRESHOLD = 0.49  # raw FAISS best_score below this => out-of-domain, no generation
MIN_QUESTION_LENGTH    = 8      # shorter questions are treated as degenerate input

# Metric weights -- must sum to 1.0.
#
# answer_relevance (answer-vs-question cosine) was REMOVED from the weighted
# score. It is structurally capped (~0.87 ceiling, ~0.77 median on this corpus)
# because a question and its answer are different grammatical forms, so it
# penalised even byte-identical-to-gold answers and could never reach 5. Its
# job -- "does this actually answer the question" -- is done far better by
# answer_correctness (semantic match to the verbatim gold answer) here, and by
# the real RAGAS/human relevance judgement in the Stage-2 human eval
# (scripts/14). We STILL compute answer_relevance_similarity into `detail` for
# traceability; it just carries no weight. Its 0.15 was folded into
# answer_correctness (the gold answer is the strongest Stage-1 signal we have).
WEIGHTS = {
    "context_relevance":  0.20,
    "faithfulness":       0.25,
    "answer_correctness": 0.40,
    "completeness":       0.15,
}

# Khmer sentence-final punctuation -- used by completeness check
KHMER_END_PUNCT = {"។", "៕"}

# Trailing decoration that can legitimately follow a real sentence-final mark
# WITHOUT making the answer incomplete. The smaller models routinely close a
# fully-finished answer with markdown emphasis ("**"), a quote/list char, or a
# bracketed "closing tag" -- e.g. "(ចប់)", "(បញ្ចប់ ៕)", "(។)", a literal "(ចប់។)",
# or a bare trailing ")". A naive last-character check then misreads these as
# truncated. The completeness check strips this decoration before testing for a
# sentence-final mark, so decoration is judged by answer_correctness (where the
# embellishment lowers similarity), not mislabelled as truncation.
# NOTE: closing brackets are deliberately NOT in _TRAILING_DECOR. If they were,
# a trailing ")**" would have its ")" stripped first, destroying the closing
# bracket before _TRAILING_BRACKET could remove the whole "(...)" group -- which
# left an orphan "(" and mislabelled a complete answer as truncated. Brackets
# are handled only by _TRAILING_BRACKET (whole balanced group); _TRAILING_DECOR
# handles only markdown emphasis / quote chars / whitespace.
_TRAILING_DECOR   = re.compile(r"[\s*_`#>~]+$")
_TRAILING_BRACKET = re.compile(
    r"[(\[{（【「『][^()\[\]{}（）【】「」『』]*[)\]\}）】」』]\s*$"
)


def ends_with_sentence_final(text):
    """True if, after stripping trailing markdown/whitespace and any trailing
    bracketed closing-tag groups, the answer ends on a Khmer sentence-final mark
    (។ or ៕). Genuinely truncated answers (cut off mid-word with no final mark)
    have no such mark to expose, so they correctly remain False."""
    s = text.strip()
    prev = None
    while s and s != prev:
        prev = s
        s = _TRAILING_BRACKET.sub("", s).rstrip()
        s = _TRAILING_DECOR.sub("", s)
    return bool(s) and s[-1] in KHMER_END_PUNCT

# Arabic + Khmer numeral runs -- used by faithfulness proxy
NUMBER_PATTERN = re.compile(r"[0-9]+|[០១២៣៤៥៦៧៨៩]+")

# Khmer spelled-out cardinals -> Khmer digit. Used so the faithfulness proxy
# does NOT flag a digit in the answer as "fabricated" when the source merely
# spells the same number out in words (e.g. the answer writes "៣ ប្រភេទ" while
# the context says "បីប្រភេទ"). Longer words are listed before their prefixes
# so e.g. "ប្រាំបី" matches before "ប្រាំ". Covers 0-10 plus the common round
# numbers that appear in this legal corpus.
KHMER_WORD_TO_DIGIT = {
    "សូន្យ": "០", "មួយ": "១", "ពីរ": "២", "បី": "៣", "បួន": "៤",
    "ប្រាំបួន": "៩", "ប្រាំបី": "៨", "ប្រាំពីរ": "៧", "ប្រាំមួយ": "៦", "ប្រាំ": "៥",
    "ដប់": "១០", "ម្ភៃ": "២០", "សាមសិប": "៣០", "សែសិប": "៤០", "ហាសិប": "៥០",
    "មួយរយ": "១០០", "មួយពាន់": "១០០០",
}

# A list-ordinal marker at the start of a line (e.g. "៣.", "3)", "៤៖", or with a
# leading bullet like "- ៣." / "* ៤)"). The number in such a marker is the
# model's own list numbering, NOT a fact pulled from the source, so it must not
# count toward (or against) faithfulness. The optional leading bullet matters:
# the smaller models routinely emit dashed-AND-numbered list items ("- ៣. ...").
LIST_ORDINAL_PATTERN = re.compile(r"^\s*[-•*]?\s*[០១២៣៤៥៦៧៨៩0-9]+\s*[.)៖]")

KHMER_SCRIPT_PATTERN = re.compile(r"[ក-៿]")

# Provision-reference handling -- mirror app.py EXACTLY (see app.py for the full
# rationale). Maximal digit runs are pulled out of Khmer tokens; explicit
# មាត្រា/ប្រការ references become a boosted compound BM25 token; Arabic digits
# are normalized to the corpus's Khmer form.
DIGIT_RUN_PATTERN   = re.compile(r"[០១២៣៤៥៦៧៨៩0-9]+")
ARTICLE_REF_PATTERN = re.compile(r"(មាត្រា|ប្រការ)\s*([០១២៣៤៥៦៧៨៩0-9]+)")
ARTICLE_REF_BOOST   = 20
_ARABIC_TO_KHMER    = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


def to_khmer_digits(s):
    return s.translate(_ARABIC_TO_KHMER)

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
    # First-person "should I <do X>, yes or no?" personal decision -- see app.py.
    re.compile(r"ខ្ញុំគួរ.{0,40}(ឬ)?ទេ"),
    # Off-task translation request (out of scope) -- see app.py.
    re.compile(r"បកប្រែ"),
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
        # Evaluate SeaLLM over the FULL gold set (manual + AI questions, 400),
        # writing to the "..._full.jsonl" files. The 100 manual answers already
        # generated into that file are skipped on resume, so only the ~300 AI
        # questions are processed. The other three models stay on the manual-only
        # 100-question set (default scope). See main().
        "gold_scope": "full",
    },
    "2": {
        "local_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "label": "Qwen 3.5 9B",
        "extra_args": ["--jinja", "--reasoning", "off"],
    },
    "3": {
        # Sailor2-L = the 128K-context variant (the plain Sailor2-8B GGUF is
        # capped at 4096, too small for these prompts). See install_models.py.
        "local_filename": "Sailor2-L-8B-Chat-Q4_K_M.gguf",
        "label": "Sailor2-L 8B",
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
    # Compound provision-reference tokens first (boosted), then per-word digit
    # runs + n-grams. Mirrors app.py exactly.
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


LIST_MARKER_PATTERN = re.compile(
    r"^(?:[០-៩0-9]+\s*[.)៖]|[ក-អ]\s*[.)]|មាត្រា|ប្រការ|[-•])"
)


def clean_provision_text(text):
    """Fold PDF line-wrap newlines inside the verbatim provision text into
    spaces, while preserving the breaks that begin a heading or list item, so
    the deterministic provision answer reads as clean paragraphs/lists instead
    of one hard break per wrapped line. Mirrors app.py EXACTLY."""
    out = []
    force_break = False  # a blank line was seen -- don't fold across it
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            # Blank lines in the corpus are stray PDF artifacts (e.g. one sits
            # mid-list before "៤." in laws មាត្រា ៣). Don't emit a visible empty
            # line, but do stop the next line from folding back up.
            force_break = True
            continue
        # Fold this line into the previous one (a mid-sentence wrap) unless it
        # opens a new structural block (heading / numbered / lettered / dash
        # item) or a blank line just separated them.
        if out and not force_break and not LIST_MARKER_PATTERN.match(line):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
        force_break = False
    return "\n".join(out).strip()


def resolve_provision_lookup(question, context_results):
    """Deterministic verbatim answer for explicit provision lookups
    (មាត្រា N / ប្រការ N). Mirrors app.py EXACTLY: bypasses the LLM and returns
    the named provision's text from the document the question names, or a
    "no such provision" message when the named document lacks it, or None when
    no provision was named (fall back to the model)."""
    refs = []
    seen = set()
    for keyword, num in ARTICLE_REF_PATTERN.findall(question):
        num = to_khmer_digits(num)
        if (keyword, num) not in seen:
            seen.add((keyword, num))
            refs.append((keyword, num))
    if not refs:
        return None

    qn = re.sub(r"\s+", "", question)
    named_docs = {
        chunk.get("metadata", {}).get("document_name", "")
        for _score, chunk in context_results
        if re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", ""))
        and re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", "")) in qn
    }

    texts = []
    for keyword, num in refs:
        heading = re.compile(
            r"^\s*" + re.escape(keyword) + r"\s*" + re.escape(num)
            + r"(?![០១២៣៤៥៦៧៨៩0-9])"
        )
        match = None
        for _score, chunk in context_results:
            text = (chunk.get("text") or "").strip()
            if not heading.match(text):
                continue
            doc = chunk.get("metadata", {}).get("document_name", "")
            if named_docs and doc not in named_docs:
                continue
            match = clean_provision_text(text)
            break
        if match is not None:
            texts.append(match)
        elif named_docs:
            doc_title = next(iter(named_docs))
            return [
                f"រកមិនឃើញ{keyword} {num} នៅក្នុង{doc_title}ឡើយ។ {REDIRECT_TEXT}"
            ]
    return texts or None


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

# Per-model style correction. Sailor2 alone tends to dress answers up with an
# invented bold heading, markdown emphasis, injected source citations/dates
# pulled from chunk metadata (e.g. "(មាត្រា ១)", "(2023-10-17)"), and a literal
# closing tag like "(ចប់) ៕" -- which reads as embellishment next to the plain
# gold answer. This extra rule block is appended to the prompt ONLY for Sailor,
# so the base prompt the other models see stays byte-for-byte unchanged (and
# their already-generated results stay valid). Keyed on a case-insensitive
# "sailor" substring so it matches both the eval label ("Sailor2-L 8B") and the
# app's model id ("Sailor2-L-8B-Chat-Q4_K_M.gguf"). Must stay identical in app.py.
SAILOR_STYLE_RULES = """
- Do NOT add a title or heading, do NOT use bold or any markdown formatting, and do NOT cite the source document's name, article/provision number, or any date (for example, do NOT write "(មាត្រា ១)", "(2023-10-17)", or "(តាម គោលការណ៍ណែនាំ...)"). Answer in plain Khmer prose, or a plain numbered/dashed list, exactly as the source presents it.
- Do NOT write any closing word or tag such as "(ចប់)", "(បញ្ចប់)", or a trailing "៕". End only with ។ on the final sentence.
- Do NOT add framing, emphasis, or any information that is not in the context. Match the wording and scope of the source -- do not expand, summarize away, or restructure its meaning."""


def _is_sailor(model_id):
    return "sailor" in (model_id or "").lower()


def build_prompt(question, context, low_confidence=False, model_id=""):
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

    # Sailor-only style correction (empty for every other model, so their
    # prompt is unchanged). Appended at the end of the rules list below.
    style_rules = SAILOR_STYLE_RULES if _is_sailor(model_id) else ""

    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.
{relevance_check}
Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Provide a complete and detailed answer. If the context lists multiple conditions, sub-points, exceptions, or requirements, enumerate ALL of them.
- Always finish your answer with a complete sentence that ends in the Khmer sentence-final sign ។ (or ៕). Even when the answer is a list, the final item must end with ។. Never stop on a word, bracket, or list item without it.
- CRITICAL: If the question names a specific provision (មាត្រា N for a law or sub-decree, ប្រការ N for a prakas), you MUST reproduce that provision's ENTIRE text from the context word-for-word -- its heading line AND every following sentence, clause, condition, and sub-point. This rule applies EVEN IF the question is phrased as "និយាយអំពីអ្វី" / "មានន័យដូចម្តេច" / "about what" -- such phrasing does NOT permit a one-line topic summary. A reply that gives only the provision's subject or title is WRONG and forbidden. Quote the full provision.
- Only combine information from multiple sources if EACH of them directly addresses the question asked. Do not stitch together unrelated facts from different sources just because they were retrieved -- if most of the retrieved sources don't actually answer the question, say so instead of forcing a combined answer out of all of them.
- Do not reinterpret a number, rate, or statistic for a purpose other than what it was stated for in the source (for example, a mortality or incidence rate expressed "per 1,000 people" is NOT the same as a total population count -- never restate a rate's denominator as if it were the literal answer to a "how many" question).
- Do not recommend, endorse, or compare any specific insurance product, company, or policy, even if the context mentions one and the question asks for a recommendation.
- Do not calculate, estimate, or infer a premium, payout, deductible, or any monetary amount that is not stated explicitly and directly in the context for that exact purpose.
- Do not give personalized financial or legal advice (e.g. "should I buy X", "what should I do in my situation"). If the question asks for a personal recommendation, a calculation, or advice tailored to the user's situation, decline and reply with the message below instead of attempting an answer.{style_rules}
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


def _context_number_forms(context):
    """All numeric forms present in the context, expressed as Khmer digits:
    the digit runs themselves (Arabic normalised to Khmer) PLUS the digit form
    of any spelled-out Khmer cardinal that appears. This is the grounding set
    the answer's numbers are checked against."""
    forms = {to_khmer_digits(n) for n in NUMBER_PATTERN.findall(context)}
    for word, digit in KHMER_WORD_TO_DIGIT.items():
        if word in context:
            forms.add(digit)
    return forms


def _answer_fact_numbers(generated_answer):
    """The numbers in the answer that represent FACTS to be grounded, expressed
    as Khmer digits. Excludes list-ordinal markers (the model's own numbering of
    a list, e.g. a line beginning "៣." or "3)"), which are formatting, not facts
    drawn from the source. Arabic digits are normalised to Khmer so digit form
    never causes a false mismatch."""
    facts = set()
    for line in generated_answer.split("\n"):
        marker = LIST_ORDINAL_PATTERN.match(line)
        body = line[marker.end():] if marker else line
        for n in NUMBER_PATTERN.findall(body):
            facts.add(to_khmer_digits(n))
    return facts


def score_faithfulness(generated_answer, context):
    """Number-overlap proxy: what fraction of the FACT numbers in the generated
    answer also appear (in any equivalent form) in the retrieved context?
    Returns (score_1_to_5, grounded, total, ungrounded_list).

    Equivalence handling (added after false-flag evidence -- see
    docs/SCORING_CALIBRATION.md): Arabic and Khmer digits are normalised to a
    single form, a spelled-out Khmer cardinal in the context grounds its digit
    form in the answer (so an answer writing "៣" is grounded by a context that
    says "បី"), and list-ordinal markers in the answer are not counted as facts.
    This removes the bulk of false "fabricated number" hits on answers that are
    otherwise near-verbatim to the source.

    An answer with no fact-numbers to check scores 5, NOT a neutral 4. No
    numbers means there is no number to fabricate -- the answer is maximally
    faithful on the only axis this proxy measures. The old neutral-4 rule
    systematically PENALISED the safest, most concise answers (e.g. clean Khmer
    prose with no figures) while rewarding models that padded in many grounded
    list/citation numbers -- it ranked the cleanest model BELOW a wordier one
    purely on this artefact. Number FABRICATION (an answer number absent from
    context) is still what drives the score down; absence of numbers is not a
    fabrication. A blank answer likewise has no numbers (-> 5 here) but is
    penalised hard elsewhere via answer_correctness=1 and completeness=1.
    """
    answer_numbers = _answer_fact_numbers(generated_answer)
    if not answer_numbers:
        return 5, 0, 0, []
    context_numbers = _context_number_forms(context)
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
    """Measures whether the answer is a COMPLETE thought, not whether it is
    concise. Two signals:
    1. Does the answer end with Khmer sentence-final punctuation (។ or ៕)?
       Answers that end mid-word/mid-sentence hit max_tokens and are incomplete.
    2. Length ratio: len(generated) / len(expected) -- used ONLY as a
       truncation floor (too short relative to the gold passage).

    There is deliberately NO over-length ceiling anymore. The old ceiling
    penalised correct, complete answers purely for being phrased more verbosely
    than the gold passage (the gold answers are terse verbatim extracts), which
    double-counted against answer_correctness -- the metric that actually judges
    whether the extra words are right. Verbosity is not incompleteness, so a
    long-but-clean answer now scores 5. Genuine padding/repetition shows up as
    LOWER answer_correctness, where it belongs.

    A blank answer fails the punctuation check AND has length_ratio = 0.0,
    which falls into the truncated branch at score = 1 (length_ratio < 0.40).
    No special-casing needed -- the existing logic handles it correctly.
    """
    stripped     = generated_answer.strip()
    ends_cleanly = ends_with_sentence_final(stripped)

    if not expected_answer or len(expected_answer) == 0:
        return (5 if ends_cleanly else 2), ends_cleanly, None

    length_ratio = len(stripped) / len(expected_answer)

    if ends_cleanly:
        # Complete sentence: full credit unless it is suspiciously SHORT
        # relative to the gold passage (likely missing list items / clauses).
        if length_ratio >= 0.80:
            score = 5
        elif length_ratio >= 0.60:
            score = 4
        elif length_ratio >= 0.40:
            score = 3
        else:
            score = 2
    else:
        # No sentence-final punctuation -> almost certainly truncated.
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


def _free_port_8080():
    """Kill any llama-server already bound to port 8080 before we launch ours.

    A stale server left over from a crashed/manually-started run (commonly the
    bare llama.cpp DEFAULT of n_ctx=4096) keeps the port. Our new process then
    fails to bind, dies, and the readiness poll below silently attaches to the
    WRONG server -- which then 400s every prompt over 4096 tokens
    ('exceeds the available context size (4096)'). Freeing the port first
    guarantees we only ever talk to the server we just started."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return
    pids = [p for p in out.split() if p.isdigit()]
    if not pids:
        return
    print(f"  WARNING: port 8080 already in use (pid {', '.join(pids)}); "
          f"killing stale server(s) so we don't attach to the wrong context.")
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    # Force-kill anything still holding the port.
    try:
        still = subprocess.run(
            ["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()
        for pid in still:
            if pid.isdigit():
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass
    time.sleep(1)


def _verify_server(info, expected_ctx):
    """Verify the live server is OURS, and report its real context window.

    Two distinct things are checked via /props:

    1. STALE/WRONG SERVER -- the model the server has loaded must be the one we
       just launched. If `model_path` names a different file, we are talking to
       a leftover server on port 8080 (the bug this whole guard exists for), so
       we abort loudly. This is the reliable signal -- NOT the context size.

    2. MODEL CONTEXT CAP -- if the correct model reports n_ctx < expected, that
       is the MODEL'S OWN trained-context limit (llama.cpp clamps --ctx-size
       down to it; e.g. Sailor2-8B's GGUF is capped at 4096), not a stale
       server. We WARN (long prompts may 400) but do NOT abort, so a
       small-context model can still be evaluated on whatever fits."""
    label = info["label"]
    try:
        props = requests.get(f"{LLAMA_SERVER_ROOT}/props", timeout=10).json()
    except Exception as e:
        print(f"  WARNING: could not read /props to verify server for {label}: {e}")
        return None

    model_path = props.get("model_path") or props.get("model") or ""
    expected_file = info["local_filename"]
    if model_path and Path(model_path).name != expected_file:
        raise SystemExit(
            f"\nllama-server on port 8080 has '{Path(model_path).name}' loaded, "
            f"but we launched '{expected_file}' for {label}. This is a "
            f"wrong/stale server. Aborting -- make sure no other llama-server "
            f"is running on port 8080, then re-run."
        )

    n_ctx = (
        props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx")
    )
    if n_ctx is None:
        print(f"  WARNING: server did not report n_ctx for {label}; "
              f"could not verify context size.")
        return None
    n_ctx = int(n_ctx)
    if n_ctx < expected_ctx:
        print(f"  WARNING: {label} model caps context at n_ctx={n_ctx} "
              f"(requested {expected_ctx}). This is the GGUF's own trained "
              f"limit, not a stale server. Prompts longer than ~{n_ctx} tokens "
              f"will be rejected for this model.")
    else:
        print(f"  Context   : n_ctx={n_ctx} (verified)")
    return n_ctx


def start_llama_server(info):
    llama_bin = shutil.which("llama-server")
    if llama_bin is None:
        raise SystemExit("llama-server not found. Install llama.cpp and add it to PATH.")

    _free_port_8080()

    model_path = LLM_DIR / info["local_filename"]
    process = subprocess.Popen(
        [
            llama_bin, "-m", str(model_path),
            "-ngl", "999",
            "--ctx-size", str(SERVER_CTX_SIZE),
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
        # If our own process died (e.g. couldn't bind the port, or OOM at this
        # ctx size), stop -- otherwise the poll could latch onto a different
        # server still listening on 8080.
        if process.poll() is not None:
            raise SystemExit(
                f"llama-server process for {info['label']} exited during "
                f"startup (code {process.returncode}). Check the model loads "
                f"at --ctx-size {SERVER_CTX_SIZE} and that port 8080 is free."
            )
        try:
            r = requests.get(f"{LLAMA_SERVER_ROOT}/v1/models", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    _verify_server(info, SERVER_CTX_SIZE)
    return process


def stop_llama_server(process):
    """Stop the llama-server we launched, robustly -- even under Ctrl+C.

    SIGTERM first, then SIGKILL if it doesn't exit in time. The wait loop
    swallows a second KeyboardInterrupt (an impatient double Ctrl+C) so the
    cleanup always finishes instead of being abandoned mid-kill -- that
    abandonment is what used to leave a stale server bound to port 8080. As a
    final backstop we free the port directly, which also catches any orphaned
    child the process handle didn't account for."""
    try:
        process.terminate()
    except Exception:
        pass
    for _ in range(15):
        try:
            process.wait(timeout=1)
            break
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            # Ignore extra Ctrl+C while we're killing the server; keep going.
            continue
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass
    # Backstop: guarantee nothing is left listening on 8080 regardless of how
    # the handle behaved (silent no-op if the port is already free).
    _free_port_8080()


# ----------------------------------------------------------------------------
# Per-question generation + scoring
# ----------------------------------------------------------------------------

def count_server_tokens(text):
    """Real token count via the running llama-server's /tokenize endpoint --
    tokenized with the CURRENTLY LOADED model's own tokenizer, so it's exact
    for whichever generator is being evaluated (character heuristics are wildly
    wrong for Khmer's byte-fallback tokenization). Returns None if the server
    is unreachable / doesn't support it, so callers can fall back to the floor."""
    if not text or not text.strip():
        return 0
    try:
        resp = requests.post(
            f"{LLAMA_SERVER_ROOT}/tokenize", json={"content": text}, timeout=10
        )
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))
    except Exception:
        return None


def compute_gold_ceiling(gold_entries):
    """Derive the output-token ceiling from the GOLD ANSWERS the eval actually
    generates against, for the model currently loaded on the server.

    The ceiling only needs to cover the longest answer we'll ever score -- which
    is the longest expected_answer in the gold set, NOT the longest chunk in the
    corpus (most chunks are never an expected answer). To stay robust to future
    gold edits, this tokenizes EVERY gold answer with the live model's own
    tokenizer (cheap -- a few hundred local /tokenize calls) and takes the true
    maximum, rather than assuming character length predicts token length. The
    max is scaled by GEN_TOKEN_HEADROOM and bounded by SERVER_OUTPUT_HARD_CAP so
    a big answer can never collide with the context window. Falls back to
    MAX_TOKENS_CEILING_FALLBACK if the server can't tokenize.

    Because the ceiling is read from the data every run, adding a longer gold
    answer later automatically raises it -- the script self-adjusts and never
    silently truncates a newly-lengthened answer."""
    answers = [
        e.get("expected_answer", "") or ""
        for e in gold_entries
        if (e.get("expected_answer") or "").strip()
    ]

    max_tok = 0
    for a in answers:
        t = count_server_tokens(a)
        if t:
            max_tok = max(max_tok, t)

    if not max_tok:
        return MAX_TOKENS_CEILING_FALLBACK, 0

    ceiling = int(max_tok * GEN_TOKEN_HEADROOM)
    ceiling = max(MAX_TOKENS_FLOOR, min(ceiling, SERVER_OUTPUT_HARD_CAP))
    return ceiling, max_tok


def choose_max_tokens(expected_answer, ceiling, prompt=None):
    """Per-question output cap sized to the expected answer's real token length.

    Tokenizes the expected answer with the live model's own tokenizer, scales by
    GEN_TOKEN_HEADROOM, and clamps to [MAX_TOKENS_FLOOR, ceiling] (the
    gold-derived ceiling from compute_gold_ceiling). When the prompt is
    given, the cap is additionally tightened so prompt_tokens + max_tokens stays
    within the server context window -- a hard guard against context-shifting,
    independent of the corpus ceiling. Returns (max_tokens, expected_tokens,
    hit_ceiling)."""
    expected_tokens = count_server_tokens(expected_answer)
    if not expected_tokens:  # None (server error) or 0 (no expected answer)
        return MAX_TOKENS_FLOOR, expected_tokens or 0, False
    needed      = int(expected_tokens * GEN_TOKEN_HEADROOM)
    hit_ceiling = needed > ceiling
    max_tokens  = max(MAX_TOKENS_FLOOR, min(needed, ceiling))

    # Hard context-window guard: never let prompt + output exceed ctx size.
    if prompt is not None:
        prompt_tokens = count_server_tokens(prompt)
        if prompt_tokens:
            room = SERVER_CTX_SIZE - prompt_tokens - PROMPT_TOKEN_MARGIN
            if room < max_tokens:
                max_tokens = max(MAX_TOKENS_FLOOR, room)

    return max_tokens, expected_tokens, hit_ceiling


def generate_answer(client, model_name, prompt, max_tokens=MAX_TOKENS):
    """Streams the answer from llama-server. Returns (answer_text,
    first_token_time, generation_time, tokens_out).

    Retries on transient generation failures (see GEN_MAX_RETRIES) -- chiefly
    the intermittent llama.cpp chat-template PEG parser error that aborts a
    request when the model samples an output the --jinja parser can't reconcile.
    Each attempt re-rolls from scratch (temperature 0.2 makes the retry sample a
    different, usually-valid trajectory). The partial output of a failed attempt
    is discarded. Raises the last error only if every attempt fails."""
    last_err = None
    for attempt in range(1, GEN_MAX_RETRIES + 1):
        gen_start        = time.time()
        first_token_time = None
        answer_text      = ""
        tokens_out       = 0
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
                frequency_penalty=GEN_FREQUENCY_PENALTY,
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
        except Exception as e:
            last_err = e
            if attempt < GEN_MAX_RETRIES:
                print(
                    f"    Generation attempt {attempt}/{GEN_MAX_RETRIES} failed "
                    f"({e}); retrying in {GEN_RETRY_BACKOFF}s..."
                )
                time.sleep(GEN_RETRY_BACKOFF)

    raise last_err


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
# Filename suffix for the per-model output files. Set per-model in main() right
# before that model is evaluated. A model run over the FULL gold set (manual +
# AI questions) writes to "..._full.jsonl"/"..._full_summary.*" so it never
# collides with the manual-only 100-question files. Empty string = the normal
# manual-only files. Because models are evaluated strictly one at a time, a
# single module-level value is unambiguous for the whole of that model's run.
RESULTS_SUFFIX = ""


def model_dir(label):
    for info in LLM_MODELS.values():
        if info["label"] == label:
            return REPORTS_DIR / Path(info["local_filename"]).stem

    raise ValueError(f"Unknown model label: {label}")

def result_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"generated_answer_results{RESULTS_SUFFIX}.jsonl"


def summary_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"generated_answer_results{RESULTS_SUFFIX}_summary.json"


def summary_md_path(label):
    path = model_dir(label)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"generated_answer_results{RESULTS_SUFFIX}_summary.md"


def load_existing_results(label):
    """Loads already-completed answers for this model so the run can
    resume from where it left off if interrupted. Keyed by question_id so
    resume is robust to gold question-text edits (falls back to the question
    text for any legacy line that predates the question_id field)."""
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
            except json.JSONDecodeError:
                continue
            # A row that recorded a hard generation error (e.g. the transient
            # llama.cpp PEG-parser failure) is NOT "done" -- leaving it out of
            # the resume set means the next run regenerates it instead of
            # skipping it forever. The stale error row is harmlessly overwritten
            # by the fresh result (reorder_results_file dedupes last-write-wins).
            if entry.get("error"):
                continue
            done[entry.get("question_id") or entry["question"]] = entry
    return done


def append_result(label, entry, gold_entries=None):
    """Writes one completed answer to the per-model JSONL file immediately.

    When gold_entries is provided, the file is re-sorted into gold order right
    after the write (see reorder_results_file), so a question regenerated on a
    resumed run lands back in its original slot -- between its gold neighbours --
    the instant it's processed, not at the end of the whole run. The rewrite is
    a few hundred small rows, so the per-write cost is negligible next to a
    multi-second LLM generation, and the file is left fully ordered and
    crash-safe after every single question."""
    path = result_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if gold_entries is not None:
        reorder_results_file(label, gold_entries)


def reorder_results_file(label, gold_entries):
    """Rewrites the per-model JSONL so its rows follow the gold question order,
    keyed by question_id (falling back to question text for legacy lines).

    Answers are written by append_result() the instant they're generated, for
    crash safety -- which means a question regenerated on a later resumed run
    (e.g. after manually deleting its line) lands at the BOTTOM of the file
    instead of back in its original slot. This realigns the file to the gold
    order once a model is finished, so the on-disk row order always matches the
    gold set regardless of how many resume cycles produced it. Rows whose key
    isn't in the gold set (stray/legacy lines) are kept, appended after the
    ordered ones, so nothing is ever silently dropped."""
    path = result_path(label)
    if not path.exists():
        return

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Last write wins per key, mirroring load_existing_results()'s dedupe.
    by_key = {}
    for row in rows:
        by_key[row.get("question_id") or row.get("question")] = row

    ordered = []
    written = set()
    for entry in gold_entries:
        key = entry.get("question_id") or entry["question"]
        if key in by_key and key not in written:
            ordered.append(by_key[key])
            written.add(key)

    # Preserve any rows not in the gold set rather than dropping them.
    for key, row in by_key.items():
        if key not in written:
            ordered.append(row)

    with open(path, "w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    # Output-token ceiling derived from this model's tokenization of the
    # longest GOLD ANSWER (the real upper bound on what the eval generates),
    # bounded by the server context window. Computed once per model.
    model_ceiling, gold_max_tok = compute_gold_ceiling(gold_entries)
    print(
        f"  Output ceiling: {model_ceiling} tokens "
        f"(longest gold answer ~{gold_max_tok} tok x{GEN_TOKEN_HEADROOM} headroom, "
        f"bounded by ctx window)."
    )

    for i, entry in enumerate(entries, start=1):
        question        = entry["question"]
        question_id     = entry.get("question_id")
        expected_chunks = entry.get("expected_chunks", [])
        expected_answer = entry.get("expected_answer", "")
        category        = entry["category"]

        if not dry_run and (question_id or question) in existing:
            print(f"  [{i:>3}/{total}] {str(question_id):<18} | Skipping (already done): {question[:60]}...")
            continue

        print(f"  [{i:>3}/{total}] {str(question_id):<18} | {question[:70]}...")

        # --- Pre-retrieval gates (mirror app.py: Khmer-script, length, scope).
        # A gated question never reaches the LLM in the live app -- it returns
        # a fixed canned message. We record that exact canned message as the
        # "generated_answer" so the evaluation reflects what the UI would
        # actually answer for this gold question. ---
        gate_reason = None
        # Whether this answer actually went through the LLM. Provision fast-path
        # and gated canned answers bypass the model entirely, so they have no
        # meaningful latency/throughput and must be excluded from timing stats.
        llm_generated = True
        gated = gate_question(question)
        if gated is not None:
            answer_text, gate_reason = gated
            context           = ""
            context_chunk_ids = []
            first_token_time  = 0.0
            generation_time   = 0.0
            tokens_out        = 0
            llm_generated     = False
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
                llm_generated     = False
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

                # --- Deterministic provision fast-path (mirrors app.py) ---
                # When the question explicitly names a provision and we have its
                # exact chunk, answer verbatim and SKIP the LLM entirely -- the
                # same bypass the live app uses, so the evaluation reflects what
                # users actually get for these questions.
                provision_texts = resolve_provision_lookup(question, context_chunks)
                if provision_texts:
                    answer_text      = "\n\n".join(provision_texts)
                    first_token_time = 0.0
                    generation_time  = 0.0
                    tokens_out       = len(answer_text)
                    prompt           = None
                    llm_generated    = False
                else:
                    # --- Generation ---
                    # low_confidence mirrors app.py exactly: best_score < MIN_SCORE
                    # (but >= OUT_OF_DOMAIN_THRESHOLD, since lower already gated out)
                    prompt = build_prompt(
                        question, context,
                        low_confidence=best_score < MIN_SCORE,
                        model_id=label,
                    )
                    # Size the output cap to THIS answer's expected length
                    # (see choose_max_tokens). Sized with the live model's own
                    # tokenizer, clamped to [floor, ceiling].
                    gen_max_tokens, exp_tok, hit_ceiling = choose_max_tokens(
                        expected_answer, model_ceiling, prompt=prompt
                    )
                    if hit_ceiling:
                        print(
                            f"    NOTE: expected answer ~{exp_tok} tok needs "
                            f">{model_ceiling} (corpus ceiling); capped. Raise "
                            f"SERVER_CTX_SIZE/ctx-size if these truncate."
                        )
                    try:
                        answer_text, first_token_time, generation_time, tokens_out = generate_answer(
                            client, model_name, prompt, max_tokens=gen_max_tokens
                        )
                    except Exception as e:
                        print(f"    Generation failed: {e}")
                        entry_out = {
                            "question_id":      question_id,
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
                        append_result(label, entry_out, gold_entries)
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

        if llm_generated:
            decode_time  = max(generation_time - first_token_time, 0.001)
            tokens_per_s = tokens_out / decode_time if tokens_out > 0 else 0.0
            timing = {
                "first_token_time": round(first_token_time, 4),
                "generation_time":  round(generation_time,  4),
                "tokens_out":       tokens_out,
                "tokens_per_sec":   round(tokens_per_s,     4),
            }
        else:
            # Provision fast-path / gated canned answers never called the LLM,
            # so they have no real latency or throughput. Record None (not
            # fabricated zeros) so summarise() excludes them from the timing
            # averages -- otherwise a bypassed answer reports e.g. 259000 tok/s
            # and pollutes the cross-model latency comparison.
            tokens_per_s = 0.0
            timing = {
                "first_token_time": None,
                "generation_time":  None,
                "tokens_out":       None,
                "tokens_per_sec":   None,
            }

        entry_out = {
            "question_id":       question_id,
            "question":          question,
            "category":          category,
            "expected_chunks":   expected_chunks,
            "expected_answer":   expected_answer,
            "generated_answer":  answer_text,
            "context_chunk_ids": context_chunk_ids,
            "gate_reason":       gate_reason,
            "llm_generated":     llm_generated,
            "scores":            scored["scores"],
            "combined":          scored["combined"],
            "detail":            scored["detail"],
            "timing":            timing,
        }

        append_result(label, entry_out, gold_entries)
        results.append(entry_out)

        blank_flag = " [BLANK]" if scored["detail"].get("blank_answer") else ""
        # Fast-path / gated answers have no LLM timing -- show a marker instead
        # of a misleading 0.0s / huge tok/s.
        timing_str = (
            f"ttft={first_token_time:.1f}s tok/s={tokens_per_s:.1f}"
            if llm_generated
            else "no-LLM (fast-path/gated, excluded from timing)"
        )
        print(
            f"    combined={scored['combined']:.3f} | "
            f"ctx={scored['scores'].get('context_relevance')} "
            f"faith={scored['scores'].get('faithfulness')} "
            f"corr={scored['scores'].get('answer_correctness')} "
            f"rel={scored['scores'].get('answer_relevance')} "
            f"comp={scored['scores'].get('completeness')} | "
            f"{timing_str}"
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
        "> Answer Relevance is computed for traceability but NOT weighted. "
        "Blank answers score 1 on Answer Correctness (not excluded from the "
        "average).\n\n"
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


def load_all_rows(label):
    """Loads EVERY row from this model's result file for the current
    RESULTS_SUFFIX (errors included), for summarisation. Unlike
    load_existing_results(), it does not dedupe or drop error rows -- summarise()
    handles those itself."""
    path = result_path(label)
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def generate_all_summaries(label):
    """Writes a summary (json + md) for EVERY result file this model has on disk
    -- the manual-only `generated_answer_results.jsonl` and, when present, the
    full `generated_answer_results_full.jsonl`. Each writes its own matching
    `..._summary.json` / `..._summary.md` (the "_full" suffix flows through the
    path helpers via RESULTS_SUFFIX).

    Returns {suffix: summary_dict} for whichever files existed. The caller picks
    which one feeds the cross-model comparison (the model's primary scope)."""
    global RESULTS_SUFFIX
    saved_suffix = RESULTS_SUFFIX
    out = {}
    for suffix in ("", "_full"):
        RESULTS_SUFFIX = suffix
        rows = load_all_rows(label)
        if rows is None:
            continue  # this scope's result file doesn't exist for this model
        out[suffix] = summarise(rows, label)  # writes ..._summary.json / .md
    RESULTS_SUFFIX = saved_suffix
    return out


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
    print(f"  → Use this model for human-scored evaluation in scripts/14_human_answer_evaluation.py")


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
            "Faithfulness uses number-overlap against retrieved context, with "
            "Arabic/Khmer digit + Khmer number-word normalisation and "
            "list-ordinal exclusion to avoid false fabrication flags. "
            "Answer Correctness uses BGE-M3 cosine similarity vs the gold "
            "expected_answer (see docs/SCORING_CALIBRATION.md). Answer "
            "Relevance (answer-vs-question cosine) is computed for traceability "
            "but is NOT weighted -- it is structurally capped and penalised even "
            "gold-identical answers; the real relevance judgement is the Stage-2 "
            "human eval. Completeness checks sentence-final punctuation and a "
            "length-ratio FLOOR only (no over-length ceiling -- verbosity is not "
            "incompleteness). "
            "BLANK ANSWERS score 1 on Answer Correctness (not excluded from "
            "averages). weighted_combined is normalised by the sum of weights "
            "that had values -- None is reserved for genuinely uncomputable "
            "metrics (missing expected_answer), not for blank generated answers. "
            "Run scripts/14_human_answer_evaluation.py on the winning model "
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
        "> Weighted metrics: Context Relevance, Faithfulness, Answer Correctness, "
        "Completeness. Answer Relevance is shown for reference but NOT weighted. "
        "Blank answers score **1** on Answer Correctness (not excluded from "
        "averages). See `docs/SCORING_CALIBRATION.md`.\n\n"
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
        f"`scripts/14_human_answer_evaluation.py`.\n"
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
# Rescore mode -- re-judge already-generated answers, no generation
# ----------------------------------------------------------------------------

def rescore_all(chunks, embedding_model):
    """Recompute every metric for the answers ALREADY on disk, then rewrite the
    per-model JSONL, every summary, and the final comparison report. Never
    starts llama-server.

    The only thing the original scoring needed that the JSONL doesn't store
    verbatim is the retrieval `context` string -- but each row keeps its
    `context_chunk_ids`, so we rebuild the exact same context via build_context()
    from the chunk metadata (headers, dates and all), making the rescore
    bit-for-bit consistent with how the answer was scored at generation time.
    Rows that recorded a generation error (no answer) are left untouched."""
    global RESULTS_SUFFIX

    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    COMPARE_SUFFIX = ""
    all_summaries = []

    banner = "=" * SEPARATOR_WIDTH
    print(f"\n{banner}")
    print(f"{BOLD}RESCORE MODE -- recomputing metrics from existing answers "
          f"(no generation){RESET}")
    print(banner)

    for info in LLM_MODELS.values():
        label = info["label"]
        rescored_any = False

        for suffix in ("", "_full"):
            RESULTS_SUFFIX = suffix
            rows = load_all_rows(label)
            if rows is None:
                continue  # this scope's file doesn't exist for this model

            rescored = 0
            for row in rows:
                if row.get("error") or not row.get("generated_answer"):
                    continue
                context_ids = row.get("context_chunk_ids") or []
                context_chunks = [
                    (0.0, chunk_lookup[c]) for c in context_ids if c in chunk_lookup
                ]
                context = build_context(context_chunks)
                scored = score_answer(
                    row["generated_answer"],
                    row["question"],
                    row.get("expected_answer", ""),
                    row.get("expected_chunks", []),
                    context_ids,
                    context,
                    embedding_model,
                )
                row["scores"]   = scored["scores"]
                row["combined"] = scored["combined"]
                row["detail"]   = scored["detail"]
                rescored += 1

            path = result_path(label)
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {label} [{suffix or 'manual'}]: rescored {rescored} answers "
                  f"-> {path}")
            rescored_any = True

        if rescored_any:
            # Rewrites both ..._summary.json/.md for whichever files exist.
            summaries = generate_all_summaries(label)
            if COMPARE_SUFFIX in summaries:
                all_summaries.append(summaries[COMPARE_SUFFIX])

    RESULTS_SUFFIX = ""

    if not all_summaries:
        raise SystemExit(
            "No existing result files found to rescore. Run the full generation "
            "first (python scripts/11_compare_generator_models.py)."
        )

    print_final_comparison(all_summaries)
    write_final_report(all_summaries)
    print("\nRescore complete. Re-run scripts/12_diagnose_generator_scores.py "
          "to refresh the diagnostics.")


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
    # Force CPU to match app.py EXACTLY (app.py loads with device="cpu").
    # On Apple Silicon, an unspecified device auto-selects MPS, whose float
    # math differs subtly from CPU -- enough to flip FAISS top-K ordering on
    # near-tie chunks, which changes the selected context and therefore the
    # generated answer. Pinning CPU keeps retrieval bit-for-bit comparable.
    embedding_model = SentenceTransformer(EMBEDDING_HF_ID, device="cpu")

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(INDEX_PATH)

    print("Loading chunk metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Rescore mode short-circuits here: it only needs the embedding model (for
    # answer_correctness/relevance) and the chunk metadata (to rebuild context
    # for faithfulness). No FAISS search, no BM25, no llama-server.
    if RESCORE:
        rescore_all(chunks, embedding_model)
        return

    print("Building BM25 index...")
    bm25_corpus = [
        khmer_aware_tokenize(c.get("embedding_text") or c["text"])
        for c in chunks
    ]
    bm25_index = BM25Okapi(bm25_corpus)

    gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))
    if not gold_files:
        raise ValueError("No gold_questions.json files found under data/09_evaluation/.")

    # Only the first 25 questions of each category are purely human-authored
    # (question, expected_chunks, AND expected_answer). Question 26 onward were
    # AI-generated and contain bad questions / hallucinated answers, so they are
    # excluded from model comparison. We key off the trailing number in
    # question_id (e.g. law_qid_026) rather than list position so the cut-off
    # holds even if the files are reordered.
    HUMAN_AUTHORED_LIMIT = 25
    _QID_NUM_PATTERN = re.compile(r"(\d+)\s*$")

    def is_human_authored(q):
        match = _QID_NUM_PATTERN.search(str(q.get("question_id", "")))
        return bool(match) and int(match.group(1)) <= HUMAN_AUTHORED_LIMIT

    # Two scopes:
    #   gold_entries_manual -- the 100 purely human-authored questions (first 25
    #     per category). The default set used by Qwen / Sailor / Llama.
    #   gold_entries_full   -- ALL questions (manual + AI, 400). Used only by
    #     models flagged "gold_scope": "full" (currently SeaLLM), written to the
    #     separate "..._full.jsonl" files.
    gold_entries_manual = []
    gold_entries_full = []
    for gf in gold_files:
        category = gf.parent.name
        with open(gf, "r", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            entry = {**q, "category": category}
            gold_entries_full.append(entry)
            if is_human_authored(q):
                gold_entries_manual.append(entry)

    def scope_for(info):
        """Returns (results_suffix, gold_entries) for a model based on its
        configured gold_scope. Full-scope models write to the separate
        "..._full" files over the complete question set; everyone else uses the
        manual-only set and the default filenames."""
        if info.get("gold_scope") == "full":
            return "_full", gold_entries_full
        return "", gold_entries_manual

    # Default gold_entries (manual) drives the run-wide mode label; per-model
    # scope is resolved inside the loop.
    gold_entries = gold_entries_manual

    mode_label = (
        "DRY RUN (1 question per model)"
        if DRY_RUN
        else f"FULL RUN ({len(gold_entries_manual)} manual / "
             f"{len(gold_entries_full)} full questions)"
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

    global RESULTS_SUFFIX

    # The cross-model comparison ALWAYS uses each model's manual-only
    # `generated_answer_results.jsonl` (suffix ""), so every model is ranked on
    # the same 100 human-authored questions. SeaLLM still RUNS over the full set
    # and writes its `..._full.jsonl` (for separate analysis), but that file is
    # never fed into the comparison. See scope_for() (run scope) vs this (compare
    # scope) -- they are deliberately decoupled.
    COMPARE_SUFFIX = ""

    all_summaries = []
    if not DRY_RUN:
        for info in installed.values():
            RESULTS_SUFFIX = COMPARE_SUFFIX
            sp = summary_path(info["label"])
            if sp.exists():
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    all_summaries.append(saved)
                    print(f"  Loaded existing summary for: {info['label']}")
                except (json.JSONDecodeError, OSError):
                    pass
        RESULTS_SUFFIX = ""

    already_done_labels = {s["label"] for s in all_summaries}

    for key, info in installed.items():
        label = info["label"]
        RESULTS_SUFFIX, gold_entries = scope_for(info)

        if not DRY_RUN and label in already_done_labels:
            existing = load_existing_results(label)
            if len(existing) >= len(gold_entries):
                print(f"\n  Skipping {label} -- all {len(existing)} questions already done.")
                # Still (re)write summaries for every result file this model has,
                # so both generated_answer_results(.jsonl) and its _full variant
                # always have an up-to-date *_summary.json/.md even on a no-op run.
                # The comparison always uses the manual-only "" summary.
                primary_suffix = COMPARE_SUFFIX
                summaries = generate_all_summaries(label)
                if primary_suffix in summaries:
                    all_summaries = [s for s in all_summaries if s["label"] != label]
                    all_summaries.append(summaries[primary_suffix])
                continue

        print(f"\n{SEPARATOR}")
        print(f"{BOLD}Starting llama-server for: {label}{RESET}")
        print(SEPARATOR)

        # process is assigned INSIDE the try so the finally below always stops
        # the server -- including when Ctrl+C lands during startup itself.
        process = None
        try:
            process = start_llama_server(info)
            print(f"  llama-server ready.")

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
                # Summarise EVERY result file this model has (manual "" and, for
                # SeaLLM, full "_full"), each into its own *_summary.json/.md.
                primary_suffix = COMPARE_SUFFIX
                summaries = generate_all_summaries(label)
                # The cross-model comparison ALWAYS uses the manual-only "" scope
                # (the 100 human-authored questions), even for full-scope models
                # like SeaLLM -- so every model is ranked on the same questions.
                summary = summaries.get(primary_suffix) or summarise(results, label)
                all_summaries = [s for s in all_summaries if s["label"] != label]
                all_summaries.append(summary)
                print_model_summary(summary)

        finally:
            if process is not None:
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
    try:
        main()
    except KeyboardInterrupt:
        # The per-model finally has already stopped the current server (and
        # freed port 8080). Exit cleanly instead of dumping a traceback; the
        # backstop at the next launch will catch anything left regardless.
        print("\nInterrupted by user. llama-server stopped; progress saved.")
        _free_port_8080()
        raise SystemExit(130)