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
# Generate Model Answers
# ----------------------------------------------------------------------------
# PURPOSE: run every installed generator model over the gold question set
# (25 questions x 4 categories = 100 questions per model), retrieve the context
# for each question exactly as app.py would, generate the answer, and store the
# result. This script does NOT score or rank anything -- scoring is done by an
# LLM judge (see scripts/14_prepare_llm_judge_input.py and
# docs/llm_judge_prompt.md).
#
# PARITY: retrieval, prompting, gating and token limits mirror app.py exactly,
# so a generated answer here is the answer the live chatbot would have given.
# See docs/generation_robustness_and_parity.md.
#
# OUTPUT (per model, under reports/generator_model_comparison/<model>/)
#   generated_answer_results.jsonl   one row per question: the question, the
#                                    retrieved context_chunk_ids, the gold
#                                    expected_answer, the generated_answer and
#                                    timing. No scores.
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
#   python scripts/13_generate_model_answers.py            # full run
#   python scripts/13_generate_model_answers.py --dry-run  # 1 question/model
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
TOP_K             = 15
CONTEXT_K         = 6
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
MIN_SCORE              = 0.49  # same relevance floor used by the live app
OUT_OF_DOMAIN_THRESHOLD = 0.49  # raw FAISS best_score below this => out-of-domain, no generation
MIN_QUESTION_LENGTH    = 8      # shorter questions are treated as degenerate input




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


OFFICIAL_STATISTICS_PATTERN = re.compile(
    r"(ស្ថិតិ|statistics|ឆ្នាំ|ប្រចាំឆ្នាំ|២០២[០-៩]|202[0-9]|បុព្វលាភធានារ៉ាប់រងដុល)",
    re.IGNORECASE,
)


def contains_khmer_script(text):
    return bool(KHMER_SCRIPT_PATTERN.search(text))


def is_scope_violation(text):
    for pattern in SCOPE_VIOLATION_PATTERNS:
        if not pattern.search(text):
            continue
        if pattern.pattern == r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន" and OFFICIAL_STATISTICS_PATTERN.search(text):
            continue
        return True
    return False


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


HTML_ARTIFACT_PATTERN = re.compile(
    r"(?:data-tip=|aria-label=|<button|</button|<svg|</svg|<div|</div|<ul|</ul|<li|</li)",
    re.IGNORECASE,
)
CONTEXT_SOURCE_NOISE_PATTERN = re.compile(
    r"^\s*(?:"
    r"Source file:|OCR method:|\[Page \d+\]|"
    r"Source URL:|URL:|File URL:|Download(?: file)?:|"
    r"Logo:|Image:|Image file:|Asset:|https?://"
    r")",
    re.IGNORECASE,
)
CONTEXT_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def clean_context_text_for_prompt(text):
    """Strip retrieval metadata/UI artifacts before the generator sees context."""
    lines = []
    for raw_line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if CONTEXT_SOURCE_NOISE_PATTERN.search(line):
            continue
        if HTML_ARTIFACT_PATTERN.search(line):
            continue
        line = CONTEXT_URL_PATTERN.sub("", line).strip()
        if not line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


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
        header_parts = [f"[Source {rank}] {doc_name}", f"ប្រភេទ: {doc_type}"]
        if release_date and release_date != "N/A":
            header_parts.append(f"កាលបរិច្ឆេទ: {release_date}")
        if chapter and chapter != "N/A":
            header_parts.append(f"ជំពូក/ផ្នែក: {chapter}")
        if section and section != "N/A":
            header_parts.append(f"ចំណុច: {section}")
        if legal_unit and legal_unit != "N/A":
            header_parts.append(f"ឯកតាច្បាប់: {legal_unit}")
        context_text = clean_context_text_for_prompt(chunk.get("text") or "")
        if not context_text:
            continue
        header = " | ".join(header_parts)
        context_parts.append(f"{header}\n{context_text}")
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
                    # low_confidence mirrors app.py exactly. MIN_SCORE currently
                    # equals OUT_OF_DOMAIN_THRESHOLD, so normal low-score cases
                    # are refused before generation.
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
            # fabricated zeros) so a bypassed answer does not report e.g.
            # 259000 tok/s and pollute the latency figures.
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
            "timing":            timing,
        }

        append_result(label, entry_out, gold_entries)
        results.append(entry_out)

        # This script GENERATES only -- scoring is the LLM judge's job (see
        # scripts/14 and 15), so there are no quality numbers to report here.
        # The progress line therefore describes the generation itself.
        blank_flag = " [BLANK]" if not answer_text.strip() else ""
        # Fast-path / gated answers have no LLM timing -- show a marker instead
        # of a misleading 0.0s / huge tok/s.
        timing_str = (
            f"ttft={first_token_time:.1f}s tok/s={tokens_per_s:.1f}"
            if llm_generated
            else "no-LLM (fast-path/gated, excluded from timing)"
        )
        print(
            f"    chars={len(answer_text)} "
            f"ctx_chunks={len(context_chunk_ids)} | "
            f"{timing_str}"
            f"{blank_flag}"
        )

    return results


# ----------------------------------------------------------------------------
# Summarise results for one model
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Per-model markdown report
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------------











# ----------------------------------------------------------------------------
# Rescore mode -- re-judge already-generated answers, no generation
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    wall_clock_start = time.time()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run: python scripts/08_generate_embeddings.py"
        )

    print("Loading embedding model (BGE-M3)...")
    # Force CPU to match app.py EXACTLY (app.py loads with device="cpu").
    # On Apple Silicon, an unspecified device auto-selects MPS, whose float
    # math differs subtly from CPU -- enough to flip FAISS top-K ordering on
    # near-tie chunks, which changes the selected context and therefore the
    # generated answer. Pinning CPU keeps retrieval bit-for-bit comparable.
    embedding_model = SentenceTransformer(
        EMBEDDING_HF_ID,
        device="cpu",
        local_files_only=True,
    )

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

    # Only the first 25 questions of each category belong to the CORE set, where
    # the question, expected_chunks and expected_answer were each written and
    # checked individually. Question 26 onward form the EXTENDED set, whose
    # quality is uneven -- it contains weak questions and some answers that do
    # not hold up -- so it is excluded from generator comparison. We key off the
    # trailing number in question_id (e.g. law_qid_026) rather than list position
    # so the cut-off holds even if the files are reordered.
    CORE_SET_LIMIT = 25
    _QID_NUM_PATTERN = re.compile(r"(\d+)\s*$")

    def is_core_set(q):
        match = _QID_NUM_PATTERN.search(str(q.get("question_id", "")))
        return bool(match) and int(match.group(1)) <= CORE_SET_LIMIT

    # Every model is evaluated on the SAME set: the 100 core gold questions
    # (first 25 per category). The extended set is not used here.
    gold_entries = []
    for gf in gold_files:
        category = gf.parent.name
        with open(gf, "r", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            if is_core_set(q):
                gold_entries.append({**q, "category": category})

    mode_label = (
        "DRY RUN (1 question per model)"
        if DRY_RUN
        else f"FULL RUN ({len(gold_entries)} core gold questions per model)"
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
        raise SystemExit("No generator models installed. Run: python install_models.py")

    generated_counts = {}

    for key, info in installed.items():
        label = info["label"]

        # Resume: a model whose result file already covers the whole gold set is
        # skipped outright. Partial files resume inside evaluate_model(), which
        # keys off question_id.
        if not DRY_RUN:
            existing = load_existing_results(label)
            if len(existing) >= len(gold_entries):
                print(f"\n  Skipping {label} -- all {len(existing)} answers already generated.")
                generated_counts[label] = len(existing)
                continue

        print(f"\n{SEPARATOR}")
        print(f"{BOLD}Starting llama-server for: {label}{RESET}")
        print(SEPARATOR)

        # process is assigned INSIDE the try so the finally below always stops
        # the server -- including when Ctrl+C lands during startup itself.
        process = None
        try:
            process = start_llama_server(info)
            print("  llama-server ready.")

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
                generated_counts[label] = len(results)
                print(f"\n  {label}: {len(results)} answers -> {result_path(label)}")

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

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"{BOLD}GENERATION COMPLETE{RESET}")
    for label, n in generated_counts.items():
        print(f"  {label:22s} {n:3d} answers")
    print(f"{'=' * SEPARATOR_WIDTH}")

    wall_clock_total = time.time() - wall_clock_start
    hours, rem       = divmod(int(wall_clock_total), 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal wall-clock runtime : {hours}h {minutes}m {seconds}s")
    print("\nNext: score these answers with an LLM judge --")
    print("  python scripts/14_prepare_llm_judge_input.py")
    print("  then follow docs/llm_judge_prompt.md")


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
