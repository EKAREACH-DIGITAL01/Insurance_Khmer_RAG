import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set HF_HOME before any HF import so all HF assets stay inside the
# project folder. See install_models.py for the full
# explanation of why this must come before the imports below.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_APP_DIR, ".env"))
except Exception:
    pass

_HF_CACHE_DIR = os.path.join(_APP_DIR, "models", "retriever", ".cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR

import base64
import html
import json
import queue
import re
import shutil
import threading
import time
import uuid
from urllib.parse import quote

import numpy as np
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

import history

transformers_logging.disable_progress_bar()

try:
    from scripts.irc_website_categories import WEBSITE_CATEGORY_LABEL_KH
except Exception:
    WEBSITE_CATEGORY_LABEL_KH = {}

def first_csv_value(value):
    return next((part.strip() for part in (value or "").split(",") if part.strip()), "")


def env_value(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_base_url(url):
    return (url or "http://localhost:8080/v1").strip().rstrip("/")


def derive_generator_root(base_url):
    if base_url.endswith("/v1"):
        return base_url[:-3].rstrip("/")
    if base_url.endswith("/v1/openai"):
        return base_url[: -len("/v1/openai")].rstrip("/")
    return base_url.rstrip("/")


DEFAULT_GENERATOR_ROOT = "http://localhost:8080"
HOSTED_DEFAULT_GENERATOR_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HOSTED_DEFAULT_GENERATOR_MODEL = "gemini-3.7-flash"
IS_HOSTED_RUNTIME = bool(
    env_value(
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_SERVICE_NAME",
        "RENDER",
        "K_SERVICE",
        "STREAMLIT_SHARING_MODE",
    )
)
DEFAULT_GENERATOR_BASE_URL = (
    HOSTED_DEFAULT_GENERATOR_BASE_URL
    if IS_HOSTED_RUNTIME
    else f"{DEFAULT_GENERATOR_ROOT}/v1"
)
LLAMA_SERVER_URL = normalize_base_url(
    env_value(
        "GENERATOR_BASE_URL",
        "OPENAI_BASE_URL",
        "GEMINI_BASE_URL",
        default=DEFAULT_GENERATOR_BASE_URL,
    )
)
LLAMA_SERVER_ROOT = normalize_base_url(
    env_value("GENERATOR_ROOT_URL", default=derive_generator_root(LLAMA_SERVER_URL))
)
GENERATOR_MODEL = env_value(
    "GENERATOR_MODEL",
    "OPENAI_MODEL",
    "GEMINI_MODEL",
    "GOOGLE_MODEL",
    default=HOSTED_DEFAULT_GENERATOR_MODEL if IS_HOSTED_RUNTIME else "",
)
MODEL = os.getenv("CHATBOT_MODEL") or GENERATOR_MODEL or "Generator"
USES_LOCAL_LLAMA_SERVER = LLAMA_SERVER_ROOT == DEFAULT_GENERATOR_ROOT
EMBEDDING_LOCAL_FILES_ONLY = env_bool(
    "EMBEDDING_LOCAL_FILES_ONLY",
    default=USES_LOCAL_LLAMA_SERVER,
)
if USES_LOCAL_LLAMA_SERVER:
    GENERATOR_API_KEY = env_value("GENERATOR_API_KEY", "OPENAI_API_KEY", default="not-needed")
else:
    GENERATOR_API_KEY = env_value(
        "GENERATOR_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        default=first_csv_value(os.getenv("GEMINI_API_KEYS")),
    )
DEPLOYMENT_COMMIT = env_value("RAILWAY_GIT_COMMIT_SHA", "GIT_COMMIT_SHA", default="")

TOP_K = 15
CONTEXT_K = 6  # how many chunks actually go into the LLM prompt (<= TOP_K)
MAX_CHUNKS_PER_DOC = 2  # cap near-duplicate chunks from the same document.
                        # NOTE: must stay > 1. Many laws are a single source
                        # document split into per-article chunks, so two
                        # DIFFERENT questions about adjacent articles of the
                        # same law (e.g. "មាត្រា ១ - គោលបំណង" vs
                        # "មាត្រា ២ - គោលដៅ") retrieve the same top chunk under
                        # dense search -- the embedder can't separate the
                        # near-synonym Khmer keywords. With a cap of 1 only that
                        # single top chunk reaches the LLM, so both questions
                        # collapse to an identical answer. Allowing 2 article
                        # chunks from the same document into the prompt lets the
                        # model answer from the article whose header actually
                        # matches the question. 2 is the minimum that fixes the
                        # collapse (both adjacent articles reach the model) while
                        # adding the least prefill/TTFT; CONTEXT_K still caps the
                        # total, so prompt size stays bounded.
DOC_FOCUS_LOOKAHEAD = 8
DOC_FOCUS_MIN_HITS = 3
DOC_FOCUS_CONTEXT_K = 4
BM25_TOP_K = TOP_K  # lexical candidate pool size, matched to FAISS's TOP_K
                     # so neither ranker dominates the fused pool just by
                     # having a bigger candidate list to draw from
RRF_K = 60  # standard Reciprocal Rank Fusion constant from the IR
            # literature (Cormack et al. 2009) -- not something tuned on
            # this corpus, just the conventional default
NGRAM_SIZE = 3  # character n-gram size for Khmer-script BM25 tokens
# Maximal runs of digits -- Khmer (U+17E0-17E9) or ASCII -- pulled out of
# Khmer-script tokens so article numbers match lexically despite no spacing.
DIGIT_RUN_PATTERN = re.compile(r"[០១២៣៤៥៦៧៨៩0-9]+")
# Explicit provision references ("មាត្រា ១០", "មាត្រា១០", "ប្រការ ៤") emitted
# as a single compound token ("មាត្រា១០"), normalizing away the optional space
# so the query and the corpus heading match regardless of spacing. Laws and
# sub-decrees number provisions with "មាត្រា"; prakas use "ប្រការ" -- same
# structure, different keyword -- so both are matched. A bare digit run alone
# can't surface the right chunk: it's one term against dozens of shared
# common-word n-grams from the document title, and "មាត្រា ១០" is short with
# little gram overlap, so it ranks ~40th. The compound is near-unique (only the
# provision heading and chunks that cross-reference it), but to overcome the
# n-gram noise it has to carry weight, so it's emitted ARTICLE_REF_BOOST times
# -- enough to lift the matching chunk into the BM25 top-K candidate pool that
# rank-based RRF fusion draws from. No effect on questions that don't name a
# provision number (no match -> no tokens emitted -> normal behaviour).
ARTICLE_REF_PATTERN = re.compile(r"(មាត្រា|ប្រការ)\s*([០១២៣៤៥៦៧៨៩0-9]+)")
# The corpus writes every provision number in Khmer digits ("មាត្រា ៦៧"), but
# users may type Arabic ("មាត្រា 67"). Normalize Arabic -> Khmer so the typed
# number matches the heading and the BM25 compound token regardless of script.
_ARABIC_TO_KHMER = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


def to_khmer_digits(s):
    return s.translate(_ARABIC_TO_KHMER)
ARTICLE_REF_BOOST = 20
DIAG_DISPLAY_K = 10  # how many fused chunks the diagnostics page SHOWS --
                      # presentation-only. The fused candidate pool itself
                      # (up to BM25_TOP_K + TOP_K unique chunks if FAISS and
                      # BM25 barely overlap) and select_context_chunks()'s
                      # CONTEXT_K selection are both untouched by this; it
                      # only trims what's rendered in the "CHUNKS" cards and
                      # "FULL SOURCE DETAILS" section below them.
MIN_SCORE = 0.49
# Hard floor below MIN_SCORE: scores under this are treated as "not even
# in-domain" and skip the LLM call entirely (out-of-domain questions like
# "who is Messi?" still return *some* top-10 from FAISS -- it always
# returns k results -- just with low scores; without this gate that
# irrelevant context still gets sent through a full ~75-150s generation).
# Tune empirically: should sit below the worst score seen on genuine
# Case 2 questions (valid domain, weak coverage) and above what garbage
# queries actually score on your index -- check the diagnostics page after
# trying a few out-of-domain questions and adjust if needed.
#
# Calibration log (raise/lower as more real data comes in):
#   "who is messi?" (caught separately by the Khmer-script gate now)  -> 0.295
#   "តើអ្នកស្រឡាញ់ខ្ញុំទេ?" (valid Khmer grammar, no topical relevance) -> 0.3959
#   "ហេតុអ្វីបានជាអ្នកឆ្លើយមិនចេញ?" (meta-question about the assistant)  -> 0.457
#       -- still hallucinated a confident wrong answer even with the
#          low-confidence relevance-check prompt instruction added.
#   "ប្រទេសកម្ពុជាសម្បូរទៅដោយមនុស្សប្រុសទេ?" (off-topic chit-chat)        -> 0.474
#       -- hallucinated a garbled answer stitching together 5 unrelated
#          chunks instead of declining.
#   "ខ្ញុំពិបាកចិត្តណាស់ជួយខ្ញុំផង" (distress, not a legal question)      -> 0.484
#       -- this one DID correctly decline. The only success in this band.
#   Raised from 0.42 to 0.49 (just under MIN_SCORE): 3 of 4 real
#   non-insurance questions tested in the 0.42-0.50 band produced
#   hallucinated/garbled output despite the relevance-check instruction,
#   not just "low confidence" ones. That band has proven to be more
#   dangerous than useful on this model/quantization, so this trades away
#   some recall on weak-but-real questions (still unconfirmed whether any
#   exist in this exact range) in favor of not fabricating answers. If a
#   genuinely valid (if weak) question starts getting rejected here,
#   that's the signal to lower this again -- but until that's observed,
#   the evidence favors precision over recall in this band.
OUT_OF_DOMAIN_THRESHOLD = 0.49
# Catches degenerate inputs ("Hi", "ok") that the score gate alone misses --
# very short/low-information queries can embed to a deceptively "moderate"
# similarity against *something* in a large corpus purely from vector-space
# geometry, not real relevance. No legitimate Khmer insurance-law question
# is this short, so this is a safe, score-independent pre-check.
MIN_QUESTION_LENGTH = 8
# Deterministic language gate: if a question contains zero Khmer-script
# characters, it's not a Khmer question -- no embedding similarity needed
# to know that. This catches English chit-chat ("do you love me?", "who is
# Messi?") that the score gate alone let through inconsistently, since
# short English phrases can land deceptively close to *something* in a
# multilingual embedding space. Khmer Unicode block: U+1780-U+17FF.
KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")


def contains_khmer_script(text):
    return bool(KHMER_SCRIPT_PATTERN.search(text))


def khmer_aware_tokenize(text):
    """BM25 tokenizer for mixed Khmer/Latin/numeric text.

    Khmer has no spaces between words, so naive whitespace splitting
    under-tokenizes a Khmer sentence into one giant "word" per run --
    nearly useless for the exact term matching BM25 depends on. Without a
    Khmer word segmenter available in this environment, this falls back to
    character n-grams (size NGRAM_SIZE) for any whitespace-delimited token
    that contains Khmer script. This is a standard fallback for CJK/
    Khmer-adjacent scripts under BM25: it can't match whole words, but it
    reliably matches shared substrings -- article numbers embedded in
    Khmer text, repeated legal terminology, partial name matches -- which
    is exactly the lexical signal dense embeddings tend to under-weight.

    Latin words and digit runs (company names like "Prudential", article
    numbers, dates) are kept as whole tokens rather than n-grammed, since
    those already have natural word boundaries and benefit more from
    exact whole-token match than from being shredded into n-grams.

    NOTE: this is a first-pass tokenizer, not validated against a proper
    Khmer word segmenter. If 10_compare_retrieval_modes.py's hybrid numbers
    don't show a real improvement over dense-only, this tokenizer -- not
    necessarily the hybrid approach itself -- is the first thing to
    re-examine."""
    tokens = []
    # Compound provision-reference tokens first (see ARTICLE_REF_PATTERN). Done
    # over the whole text, not per-word, since the keyword ("មាត្រា"/"ប្រការ")
    # and its number sit in separate whitespace tokens in the corpus heading
    # ("មាត្រា ១០"). Keeping the keyword in the compound prevents a "មាត្រា ៥"
    # query from matching a "ប្រការ ៥" heading in another document.
    for keyword, ref_num in ARTICLE_REF_PATTERN.findall(text):
        tokens.extend([keyword + to_khmer_digits(ref_num)] * ARTICLE_REF_BOOST)
    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            # Khmer is unspaced, so an entire question collapses into one
            # whitespace-token. Article numbers ("មាត្រា១០") then get glued to
            # surrounding script and shredded into n-grams, so the
            # distinguishing digit run never survives as a standalone token to
            # match the corpus -- where "មាត្រា ១០" keeps "១០" separate. Pull
            # every maximal digit run (Khmer ០-៩ U+17E0-17E9 or ASCII 0-9) out
            # as its own token first, so the article number matches lexically
            # regardless of spacing on either side.
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


# ----------------------------------------------------------------------------
# Disclaimer & scope-boundary text (Internship Proposal §6.5, §11)
# ----------------------------------------------------------------------------
# IMPORTANT: this Khmer wording is a first draft, not reviewed by a native
# Khmer-speaking domain expert. Per the proposal's own §9.3 human evaluation
# panel requirement, this text must be reviewed/corrected before any
# external demonstration -- treat it as a placeholder for that review, not
# as final legal-disclaimer copy.
DISCLAIMER_TEXT = (
    "ចំណាំ៖ ចម្លើយនេះមានគោលបំណងផ្តល់ព័ត៌មានទូទៅសម្រាប់ការអប់រំប៉ុណ្ណោះ "
    "មិនមែនជាការផ្តល់ប្រឹក្សាផ្នែកហិរញ្ញវត្ថុ ឬច្បាប់ផ្ទាល់ខ្លួនទេ។ "
    "សម្រាប់ករណីជាក់លាក់របស់អ្នក សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ "
    "ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC)។"
)

REDIRECT_TEXT = (
    "សូមទាក់ទងភ្នាក់ងារធានារ៉ាប់រងដែលមានអាជ្ញាប័ណ្ណ ឬនិយ័តករធានារ៉ាប់រងកម្ពុជា (IRC) "
    "សម្រាប់ជំនួយបន្ថែម។"
)

NOT_KHMER_QUESTION_TEXT = (
    "សូមសួរសំណួរជាភាសាខ្មែរដែលពាក់ព័ន្ធនឹងធានារ៉ាប់រងកម្ពុជា"
)

# ----------------------------------------------------------------------------
# Deterministic scope-violation gate
# ----------------------------------------------------------------------------
# Why this exists as a SEPARATE gate, not just prompt rules in build_prompt():
# the OUT_OF_DOMAIN_THRESHOLD calibration log above already proved that
# asking this 7B model to police itself via prompt instructions alone is
# unreliable once retrieval finds TOPICALLY RELEVANT chunks -- e.g. "which
# insurance company should I buy from" still retrieves real prakas/law text
# about licensed insurers, so best_score stays well above both thresholds
# and the low-confidence relevance-check instruction never even fires.
# A scope violation (asking for a specific recommendation, advice framed as
# "if you were me", or a personal monetary calculation) is a property of
# the QUESTION ITSELF, not of how relevant the retrieved context is, so it
# needs its own deterministic check -- same philosophy as the Khmer-script
# and length gates above -- rather than depending on retrieval score or
# trusting the prompt rules as the only line of defense. The in-prompt
# rules in build_prompt() are kept as a second layer for phrasing this
# gate doesn't catch, not removed.
#
# These patterns are a first pass based on observed failures during manual
# testing, not an exhaustive adversarial test set -- expect to extend this
# list as more bypass phrasings are found (see Internship Proposal §11's
# requirement to test scope boundaries "under creative user queries").
SCOPE_VIOLATION_PATTERNS = [
    # "(please) recommend a company/product/plan" -- an imperative
    # recommendation request directed AT the assistant. Deliberately
    # requires "ណែនាំ" to co-occur near "ក្រុមហ៊ុន/ផលិតផល/គម្រោង" rather than
    # triggering on "ណែនាំ" alone, since that word also appears constantly
    # in this corpus's own document titles ("គោលការណ៍ណែនាំ" = "guidance").
    re.compile(r"(សូម)?ណែនាំ.{0,30}(ក្រុមហ៊ុន|ផលិតផល|គម្រោង)"),
    # "which company/product is the best / better"
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,30}ល្អ(បំផុត|ជាង)"),
    re.compile(r"មួយណា.{0,15}ល្អ(ជាង|បំផុត)?"),
    # Popularity / ranking / reputation framing -- functionally the same as
    # "which is best" but without the literal word "ល្អបំផុត"/"ណែនាំ", so the
    # patterns above miss it (e.g. "which company is most commonly used",
    # "which company has the best reputation"). Each still requires a
    # company/product/plan word to co-occur, same precision trade-off as
    # the patterns above.
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}និយម.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ពេញនិយម"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}ល្បី.{0,15}ជាងគេ"),
    re.compile(r"(ក្រុមហ៊ុន|ផលិតផល|គម្រោង).{0,40}កេរ្តិ៍ឈ្មោះ.{0,10}ល្អ"),
    # "should I buy/choose/use from which company"
    re.compile(r"គួរ.{0,15}(ទិញ|ជ្រើសរើស|ប្រើ).{0,30}ក្រុមហ៊ុន"),
    # First-person "what should I choose/buy/use/look for" -- personal
    # product-selection advice framing, WITHOUT requiring a company to
    # co-occur (catches generic phrasing like "what should I choose?").
    # Deliberately anchored on literal "ខ្ញុំគួរ" ("I should") rather than
    # "គួរ" alone, since the corpus answers regulatory facts in third
    # person ("the insured may...", "the company must..."), so this is
    # unlikely to false-positive on legitimate retrievable content.
    #
    # The verb list is deliberately narrower than "any verb" -- it's
    # scoped to product-SELECTION verbs (choose/buy/use/look-at), not
    # general "what should I do" framing. A future claims-process corpus
    # (Internship Proposal §4's "step-by-step guidance on how to prepare
    # a claim" sub-domain) would legitimately need to answer first-person
    # "what should I do to file a claim" questions, which this pattern is
    # intentionally narrow enough not to catch.
    re.compile(r"ខ្ញុំគួរ.{0,20}(ជ្រើសរើស|ទិញ|ប្រើ|មើល)"),
    # "I don't want personal advice, but..." -- saying this is itself a
    # strong signal the question is advice-shaped, even when the user
    # explicitly disclaims wanting advice while still asking for it.
    re.compile(r"មិន.{0,10}(សុំ|ចង់បាន).{0,10}ប្រឹក្សា"),
    # "if you were me" -- explicitly asking for the assistant's own personal
    # pick. This phrasing essentially never appears in regulatory text, so
    # it's safe as a standalone trigger with no co-occurrence requirement.
    re.compile(r"បើអ្នកជាខ្ញុំ"),
    # "how much must I personally pay/spend" -- a personal monetary
    # calculation request, as opposed to a question about a fee amount
    # that IS stated as a fact in the corpus (e.g. "what is the annual
    # license fee" is fine; "how much must I pay per month" is not, since
    # nothing in the corpus states a personal monthly figure to retrieve).
    re.compile(r"ត្រូវ(បង់|ចំណាយ|ខាត).{0,10}ប៉ុន្មាន"),
    # Personal/product premium questions stay out of scope; official
    # statistics questions are allowed by is_scope_violation().
    re.compile(r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន"),
    # First-person "should I <do X>, yes or no?" -- a personal decision the
    # user wants the assistant to make for them (e.g. "should I file a claim
    # now?", "should I renew?"). Anchored on "ខ្ញុំគួរ" ("I should") closing
    # with a yes/no marker ("ទេ"/"ឬទេ"), so it stays first-person and
    # decision-shaped. The corpus answers regulatory facts in third person and
    # never contains "ខ្ញុំគួរ", so this does not false-positive on legitimate
    # retrievable content. Broader than the product-SELECTION verb pattern
    # above, which only catches choose/buy/use/look-at.
    re.compile(r"ខ្ញុំគួរ.{0,40}(ឬ)?ទេ"),
    # Off-task requests that are not Cambodian-insurance-law questions at all:
    # asking the assistant to translate a word/phrase. Translation is a
    # general-purpose task outside this assistant's scope; "បកប្រែ" does not
    # appear in legitimate questions about the corpus's content.
    re.compile(r"បកប្រែ"),
]


OFFICIAL_STATISTICS_PATTERN = re.compile(
    r"(ស្ថិតិ|statistics|ឆ្នាំ|ប្រចាំឆ្នាំ|២០២[០-៩]|202[0-9]|បុព្វលាភធានារ៉ាប់រងដុល)",
    re.IGNORECASE,
)


def is_scope_violation(text):
    """Deterministic, score-independent check for questions asking the
    assistant to do something explicitly out of scope per the Internship
    Proposal's §4/§11 boundaries: recommend or compare a specific company
    or product, give advice framed as a personal choice, or calculate a
    personal monetary amount. See the comment block above for why this is
    a separate gate rather than relying on prompt instructions alone."""
    for pattern in SCOPE_VIOLATION_PATTERNS:
        if not pattern.search(text):
            continue
        if pattern.pattern == r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន" and OFFICIAL_STATISTICS_PATTERN.search(text):
            continue
        return True
    return False


MAX_USED_SOURCES = 3

# Repetition penalty for generation. Small quantized 7B-9B models can fall into
# a degenerate REPETITION LOOP -- emitting the same sentence over and over until
# they hit max_tokens (a stochastic failure at temperature 0.2; the same
# question can answer cleanly on another run). frequency_penalty scales with how
# many times a token has already been produced, so it breaks a runaway loop fast
# while leaving normal legal-term reuse (which only repeats a handful of times)
# largely untouched -- unlike presence_penalty, which flatly penalizes ANY
# reused token and would distort Khmer legal vocabulary. MUST stay identical to
# scripts/13_generate_model_answers.py so eval and production generate the
# same way. 0.3 is conservative; raise toward 0.6-0.8 if loops still appear.
GEN_FREQUENCY_PENALTY = 0.6

# --- Dynamic output-token cap (production) ----------------------------------
# The eval (scripts/13) sizes the output cap to each gold answer's token length.
# Production has NO expected answer -- the client can ask anything -- so we use
# the runtime signal that IS available: the retrieved context. The answer is
# grounded in (and faithfully bounded by) the chunks we retrieved, so we size
# the cap to the LARGEST retrieved chunk x headroom. A query that pulls a big
# provision gets a big cap (won't truncate); one that pulls small chunks gets a
# small cap (faster first token, since the server reserves buffers to the cap).
# Clamped to [floor, hard cap] and further bounded so prompt + output can never
# exceed the context window.
MAX_TOKENS_FLOOR    = 512    # enough for legal lists, low enough to avoid demo-time runaways
GEN_TOKEN_HEADROOM  = 1.25   # source-backed answers need room for Khmer list wording
SERVER_CTX_SIZE     = 16384  # must match start_chatbot.py's --ctx-size
PROMPT_TOKEN_MARGIN = 256    # safety gap so prompt + max_tokens < ctx size
# Absolute output ceiling independent of any one prompt: leaves room for a large
# (~9k) retrieval prompt inside the context window. Per query we tighten this
# further using that query's real prompt length.
SERVER_OUTPUT_HARD_CAP = SERVER_CTX_SIZE - 9000  # ~7384 tokens
DEMO_OUTPUT_HARD_CAP = 4096

# Transient generation failures: with --jinja, llama-server parses streamed
# output through the chat-template PEG parser and intermittently aborts with
# "does not match the expected peg-native format" when the model samples an
# output it can't reconcile. It's stochastic, so we re-roll -- but only while
# NOTHING has been streamed to the UI yet (once tokens are on screen we can't
# cleanly restart). Kept in sync with scripts/13_generate_model_answers.py.
GEN_MAX_RETRIES   = 3
GEN_RETRY_BACKOFF = 2.0

# Which precomputed vector_db/<subfolder>/ the live app loads.
#
# BGE-M3 is the production retriever -- it won the three-way comparison in
# reports/embedding_model_comparison/ -- and is not selectable at runtime.
# The other two indices exist only so that comparison can be re-run:
# scripts/08_generate_embeddings.py builds every installed model into its
# own subfolder (vector_db/BAAI-BGE-M3/, vector_db/multilingual-e5-large/,
# vector_db/MiniLM-paraphrase-multilingual-L12-v2/). Repointing this
# constant is therefore a deliberate change of retriever, not a runtime
# option, and needs no rebuild of anything already generated -- but the
# reports under reports/ then no longer describe what the app runs.
ACTIVE_EMBEDDING_MODEL_DIR = "BAAI-BGE-M3"

VECTOR_DB_ROOT = "vector_db"

PHASES = [
    "encoding_query",
    "searching_faiss",
    "searching_bm25",
    "fusing_results",
    "building_context",
    "constructing_prompt",
    "preparing_response",
    "generating_response",
    "completed",
]
PHASE_LABELS = {
    "encoding_query": (
        "<b>Encoding Query</b> - "
        "<i>BGE-M3 is converting the question into an embedding vector</i>"
    ),
    "searching_faiss": (
        "<b>Searching FAISS Index</b> - "
        "<i>FAISS is searching the vector database for relevant chunks</i>"
    ),
    "searching_bm25": (
        "<b>Searching BM25 Index</b> - "
        "<i>Lexical keyword search is running alongside semantic search</i>"
    ),
    "fusing_results": (
        "<b>Fusing Rankings</b> - "
        "<i>Semantic and keyword rankings are being combined</i>"
    ),
    "building_context": (
        "<b>Building Retrieval Context</b> - "
        "<i>Retrieved chunks are being assembled into context</i>"
    ),
    "constructing_prompt": (
        "<b>Constructing Prompt</b> - "
        "<i>Context, instructions, and question are being merged</i>"
    ),
    "preparing_response": (
        "<b>Preparing Response</b> - "
        f"<i>{MODEL} is processing the prompt and preparing the first token</i>"
    ),
    "generating_response": (
        "<b>Generating Response</b> - "
        f"<i>{MODEL} is generating the answer from retrieved context</i>"
    ),
    "completed": ("<b>Completed</b>"),
}

CHATBOT_NAME = "ជំនួយការធានារ៉ាប់រងកម្ពុជា"
CHATBOT_SUBTITLE = "Cambodia Insurance Assistant"
# Welcome-state copy: a formal Khmer greeting, then the offer to help. The
# second line is the invitation to type, so the middle of the welcome screen
# asks for exactly the one thing the screen is for.
WELCOME_GREETING = "សូមគោរពជម្រាបសួរ"
WELCOME_SUBLINE = "តើមានអ្វីខ្ញុំអាចជួយដល់លោក លោកស្រីដែរឬទេ?"
# Names the document types actually in the corpus, so the screen says what can
# be asked, not just that it will help. Kept as its own line rather than folded
# into the line above: that one asks what the user needs, and a sentence doing
# both jobs buries the question.
WELCOME_HINT = "សូមសួរអំពីច្បាប់ អនុក្រឹត្ត សេចក្តីប្រកាស និងសេចក្តីណែនាំ។"
IRC_LOGO_PATH = os.path.join(_APP_DIR, "assets", "logo", "logo.png")
FAVICON_PATH = os.path.join(_APP_DIR, "assets", "logo", "favicon.png")


def image_data_uri(path, mime_type):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


IRC_LOGO_DATA_URI = image_data_uri(IRC_LOGO_PATH, "image/png")
IRC_LOGO_HTML = f'<img class="kc-logo" src="{IRC_LOGO_DATA_URI}" alt="IRC logo" />'

favicon = Image.open(FAVICON_PATH)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("is_generating", False)
st.session_state.setdefault("stop_event", None)
st.session_state.setdefault("active_run", None)
# Holds the real queue.Queue and the numeric phase-tracking state for an
# in-flight generation, NOT just a rendered snapshot of it (that's what
# active_run is for). Storing the actual objects means ANY future script
# run -- including the one triggered by navigating to Diagnostics and
# back -- can literally resume draining the same background thread's
# output where a previous, now-killed run left off, instead of the thread
# being orphaned with nobody left to read its queue.
st.session_state.setdefault("pipeline_out_q", None)
st.session_state.setdefault("pipeline_state", None)
# The conversation the working "messages" list belongs to. None means a new
# chat nobody has spoken in yet -- the row is created lazily on the first
# message, so opening the app or clicking "ថ្មី" never leaves empty
# conversations littering the sidebar.
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("renaming_id", None)

history.init_db()

# Which page is showing lives in the URL, not in session state, so every
# answer's diagnostics has its own address: ?view=diagnostics&msg=<message id>.
# That address is openable cold -- render_diagnostics_page() falls back to
# looking the message up in the database when the current session doesn't
# have it -- so it can be reloaded, bookmarked, or handed to someone else.
DIAG_MSG_ID = st.query_params.get("msg")
IS_DIAGNOSTICS = st.query_params.get("view") == "diagnostics" and bool(DIAG_MSG_ID)


def open_chat():
    st.query_params.clear()

st.set_page_config(
    page_title=CHATBOT_NAME,
    page_icon=favicon,
    layout="wide" if IS_DIAGNOSTICS else "centered",
    # Diagnostics is a full-width single-purpose readout with its own "Back
    # to chat" control -- the conversation rail is noise there.
    initial_sidebar_state="collapsed" if IS_DIAGNOSTICS else "expanded",
)


def process_feedback_query():
    """Persist a reaction from the answer action row, then clean the URL."""
    msg_id = st.query_params.get("feedback_msg")
    value = st.query_params.get("feedback")
    if not msg_id or value not in {"up", "down"}:
        return

    if history.record_feedback(msg_id, value):
        for msg in st.session_state["messages"]:
            if str(msg.get("id")) == str(msg_id):
                msg["feedback"] = value
                msg["feedback_updated_at"] = int(time.time())
                break

    st.query_params.clear()
    st.rerun()


process_feedback_query()


def ensure_conversation(question):
    """Opens a conversation row for the working thread if it doesn't have one.

    Called the moment a question is accepted, not when its answer lands. A
    conversation begins when you ask, and the rail renders before the pipeline
    runs -- creating the row at commit time instead left the rail showing the
    new conversation as unselected for the whole length of the generation, and
    only catching up on some later rerun."""
    if st.session_state["conversation_id"] is None:
        st.session_state["conversation_id"] = history.create_conversation(
            history.derive_title(question)
        )
    return st.session_state["conversation_id"]


def commit_message(msg):
    """The single write-through point for finished messages.

    Every path that produces a message -- normal answer, interrupted, errored,
    out-of-domain -- goes through here rather than appending to session state
    directly, so there is no way to add a message that silently fails to
    persist."""
    st.session_state["messages"].append(msg)
    history.append_message(ensure_conversation(msg["question"]), msg)

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

PAGE_BG = "var(--ink)" if IS_DIAGNOSTICS else "var(--parchment)"
PAGE_FG = "var(--mist)" if IS_DIAGNOSTICS else "var(--stone)"
BUTTON_HOVER_COLOR = "var(--signal)" if IS_DIAGNOSTICS else "var(--lacquer)"
# Streamlit's default block-container top padding is 96px, sized for pages
# that put a header bar there. Neither page here does: diagnostics opens on
# its Back button, and chat opens straight onto the thread, so on both that
# padding is dead space. Chat goes to zero and does its own spacing in
# .kc-thread, which is what lets the first question sit level with the rail's
# brand instead of a screen-height below it.
TOP_PADDING_CSS = (
    '[data-testid="stAppViewContainer"] .block-container '
    "{ padding-top: 1.2rem !important; }"
    if IS_DIAGNOSTICS
    # Chat owns both ends: 9px above (which, plus the first message row's own
    # 18px margin, lands the opening question level with the rail's logo) and
    # enough below to clear the fixed chat bar and disclaimer.
    else '[data-testid="stAppViewContainer"] .block-container '
    "{ padding-top: 9px !important; padding-bottom: 150px !important; }"
)


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: #0E2F20;
  --ink-soft: #123D2A;
  --ink-line: #246344;
  --parchment: #F7FAF8;
  --parchment-soft: #FFFFFF;
  --irc-primary: #02913F;
  --irc-secondary: #5AA87B;
  --irc-gold: #B19F50;
  --irc-white: #F9FAF9;
  --lacquer: var(--irc-primary);
  --lacquer-deep: #006F31;
  --lacquer-mist: rgba(2,145,63,0.10);
  --brass: var(--irc-gold);
  --brass-soft: rgba(177,159,80,0.34);
  --stone: #163326;
  --mist: #A9B9AF;
  --signal: var(--irc-secondary);
  --danger: #C0524A;
  /* Conversation rail. A tint of --parchment rather than the --ink surface
     the diagnostics page uses: the rail is a quiet index the eye passes over
     on its way to the answer, and a full dark panel next to a near-white
     thread pulls attention the wrong way. It reads as a recessed shelf of the
     same material, separated by tone alone. */
  --rail: #EDF2EE;
  --rail-hover: #E2EAE4;
  --rail-line: rgba(22,51,38,0.10);
  /* Live width of the rail, in px, kept in sync by the observer in
     sync_rail_width(). Fixed-position chrome (top bar, chat bar) is
     positioned against the chat area rather than the viewport, so it can't
     run under the rail or center itself against the wrong box. 0 is the
     correct fallback: if the script never runs, the rail is not there. */
  --kc-rail-w: 0px;
  /* Side-panel toggle mark, inlined as a data URI (no external requests). */
  --kc-panel-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2.5'/%3E%3Cline x1='9.5' y1='4' x2='9.5' y2='20'/%3E%3Cpolyline points='16.2,9.5 13.7,12 16.2,14.5'/%3E%3C/svg%3E");
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
/* The toolbar is kept in the layout because Streamlit nests the control that
   re-opens a collapsed rail inside it; its own actions are hidden below. */
[data-testid="stToolbar"] {{
  background: transparent !important;
  padding: 0 !important;
}}
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {{display: none !important;}}
[data-testid="stExpandSidebarButton"] {{
  display: inline-flex !important;
  position: fixed !important;
  top: 10px !important;
  left: 12px !important;
  z-index: 1000 !important;
}}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stExpandSidebarButton"] svg {{
  color: var(--stone) !important;
  fill: var(--stone) !important;
  opacity: 0.55;
}}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
  width: 19px !important;
  height: 19px !important;
  font-size: 0 !important;
  background-color: var(--stone);
  -webkit-mask: var(--kc-panel-icon) center / contain no-repeat;
  mask: var(--kc-panel-icon) center / contain no-repeat;
  opacity: 0.6;
  transform: scaleX(-1);  /* chevron points the way the rail will move */
}}
[data-testid="stDecoration"] {{display: none !important;}}
[data-testid="stStatusWidget"] {{display: none !important;}}
[data-testid="stForm"] div[data-testid="InputInstructions"] {{display: none !important;}}
/* The header is flattened to nothing rather than removed. Streamlit renders
   control that RE-OPENS a collapsed rail into this header, and nowhere else
   -- `display: none` here left the rail collapsible but not reopenable, with
   a full page reload the only way back. Its decorative contents are hidden
   individually above. */
header, [data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}}


html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {PAGE_BG} !important;
  color: {PAGE_FG};
  font-family: 'IBM Plex Sans', sans-serif;
}}

{TOP_PADDING_CSS}

.kc-welcome-wrap {{ margin-top: 26vh; text-align: center; }}
.kc-greeting {{
  font-family: 'Moul', serif;
  font-size: 27px;
  line-height: 1.75;
  color: var(--lacquer);
}}
.kc-greeting-sub {{
  font-family: 'Hanuman', sans-serif;
  font-size: 16px;
  line-height: 1.9;
  color: var(--stone);
  opacity: 0.75;
  margin-top: 10px;
}}
.kc-greeting-hint {{
  font-family: 'Hanuman', sans-serif;
  font-size: 13px;
  line-height: 1.9;
  color: var(--stone);
  opacity: 0.5;
  margin-top: 4px;
}}
.kc-logo {{
  display: block;
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  object-fit: contain;
  border-radius: 50%;
}}

/* welcome-only input: in-flow under the title, NOT pinned. Arrow is laid out
   as a real adjacent column rather than absolutely positioned, since the
   latter kept landing in the wrong place. */
[data-testid="stForm"] {{
  background: var(--parchment-soft) !important;
  border: 1px solid var(--brass-soft) !important;
  border-radius: 26px !important;
  padding: 6px 16px !important;
  box-shadow: 0 4px 18px rgba(58,53,43,0.12);
  max-width: 560px;
  width: 100%;
  margin: 32px auto 0 auto !important;
}}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
  align-items: center !important;
  gap: 0.5rem !important;
}}
[data-testid="stForm"] [data-testid="column"] {{ padding: 0 !important; }}
[data-testid="stForm"] [data-testid="stWidgetLabel"] {{ display: none; }}
[data-testid="stForm"] [data-testid="stTextInput"],
[data-testid="stForm"] [data-testid="stTextInput"] > div,
[data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stBaseButton-secondary"],
[data-testid="stForm"] [data-testid="stTextInput"] div[class*="BaseInput"] {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input {{
  background: transparent !important;
  border: none !important;
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  color: var(--stone);
  padding: 10px 8px;
  box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {{
  color: rgba(22, 51, 38, 0.58) !important;
  opacity: 1 !important;
}}
[data-testid="stForm"] [data-baseweb="input"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

[data-testid="stForm"] [data-baseweb="input"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {{ box-shadow: none !important; }}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] {{
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
  background: var(--lacquer) !important;
  color: var(--parchment-soft) !important;
  border: none !important;

  width: 42px !important;
  height: 42px !important;
  min-width: 42px !important;

  border-radius: 50% !important;

  margin-right: 0px !important;
  margin-left: 10px !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  font-size: 28px !important;
  font-weight: 800 !important;
  line-height: 1 !important;

  box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button::before {{
    content: "↑";
    font-size: 20px;
    font-weight: 500;
    color: white;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
  background: var(--lacquer-deep) !important;
  transform: translateY(-1px);
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:focus-visible {{
  outline: 2px solid var(--brass);
  outline-offset: 2px;
}}

/* The chat area carries no top chrome at all -- the identity lives in the
   rail (.kc-brand), so the thread starts at the top of its own column. */
/* The helper scripts (sync_rail_width, disable_rail_resize, scroll_to_bottom,
   the keydown guard) each render as a 0-height iframe, but they are still flex
   children of the page's vertical block -- which has a 16px gap -- so every one
   of them contributed a full gap of dead space above the first message even at
   zero height. Taking them out of flow drops those gaps while leaving the
   iframes live; `display: none` would stop their scripts from running at all. */
/* Every element in a vertical block spends a 16px flex gap, including the ones
   that render nothing: the helper-script iframes, the injected <style> block,
   the fixed-position disclaimer, and the empty .kc-thread div. Left in flow
   they stacked up into a screenful of dead space above the first question --
   space that was then scrollable. Taken out of flow they cost nothing, and the
   iframes stay live (display:none would stop their scripts running). */
[data-testid="stAppViewContainer"] [data-testid="stElementContainer"]:has(> iframe.stIFrame),
[data-testid="stAppViewContainer"] [data-testid="stElementContainer"]:has(style),
[data-testid="stAppViewContainer"] [data-testid="stElementContainer"]:has(.kc-disclaimer),
[data-testid="stAppViewContainer"] [data-testid="stElementContainer"]:has(.kc-thread) {{
  position: absolute !important;
  top: 0 !important;
  height: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
}}

/* .kc-thread carries NO spacing. It looks like a wrapper around the message
   list but it never was one: Streamlit sanitises each st.markdown block on its
   own, so the opening <div> is auto-closed immediately and renders as an empty
   standalone box -- the messages are siblings after it, not children. Any
   padding put here therefore became a ~160px empty box sitting above the
   thread, which is exactly the dead space that pushed the first question down
   the screen. The real spacing lives on .block-container. */
.kc-thread {{ padding: 0; margin: 0; }}

.kc-msg-user-row {{ display: flex; justify-content: flex-end; margin: 18px 0 6px 0; }}
.kc-msg-user {{
  background: var(--lacquer);
  color: var(--parchment-soft);
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  line-height: 1.7;
  padding: 10px 16px;
  border-radius: 14px 14px 2px 14px;
  max-width: 80%;
  white-space: pre-wrap;
}}

.kc-msg-assistant-row {{ margin: 4px 0 22px 0; }}
.kc-msg-assistant {{
  font-family: 'Hanuman', sans-serif;
  font-size: 16.5px;
  line-height: 1.48;
  color: var(--stone);
  white-space: pre-line;
  min-height: 1.4em;
  overflow-wrap: anywhere;
}}
.kc-msg-assistant br {{
  line-height: 1.15;
}}

/* Sources read as part of the answer, not as a separate labelled panel: body
   type, ordinary bullets, and the document name as a plain link. The brass
   rules and monospace chunk ids that used to be here announced a system
   detail (which chunk was retrieved) in the loudest colour on the page. */
.kc-sources-title {{
  font-family: 'Hanuman', sans-serif;
  font-size: 14px;
  color: var(--stone);
  opacity: 0.75;
  margin-top: 14px;
}}
.kc-sources-list {{
  margin: 4px 0 0 0;
  padding-left: 20px;
  list-style: disc;
}}
.kc-sources-list li {{
  font-family: 'Hanuman', sans-serif;
  font-size: 14px;
  line-height: 1.45;
  color: var(--stone);
  margin: 2px 0;
}}
.kc-sources-list li::marker {{ color: var(--stone); opacity: 0.45; }}
.kc-sources-list a {{
  color: var(--lacquer);
  text-decoration: none;
  border-bottom: 1px solid transparent;
}}
.kc-sources-list a:hover {{
  color: var(--lacquer-deep);
  border-bottom-color: var(--lacquer-deep);
}}
.kc-sources-list a:focus-visible {{
  outline: 2px solid var(--brass);
  outline-offset: 2px;
}}
.kc-sources-list svg,
.kc-msg-assistant svg:not(.kc-inline-svg) {{
  max-width: 18px;
  max-height: 18px;
}}

/* Pinned under the chat bar, sharing its exact left/right bracket so the two
   read as one unit sitting in the chat area. */
.kc-disclaimer {{
  position: fixed;
  left: var(--kc-rail-w, 0px);
  right: 0;
  bottom: 14px;
  margin: 0 auto;
  width: min(560px, calc(100vw - var(--kc-rail-w, 0px) - 48px));
  z-index: 999;
  font-family: 'Hanuman', sans-serif;
  font-size: 10.5px;
  line-height: 1.75;
  text-align: center;
  color: var(--stone);
  opacity: 0.5;
}}

/* ---- Answer action row ----
   Four controls of the same kind in one flex row: they align by construction,
   and each carries a plain title attribute so the tooltips match. */
.kc-actions {{
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  overflow: visible;
}}
.kc-act {{
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  padding: 0;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--stone);
  opacity: 0.55;
  cursor: pointer;
  transition: background 120ms ease, opacity 120ms ease, color 120ms ease;
}}
/* Tooltip. The four controls are plain elements, so they'd otherwise fall back
   to the browser's native title bubble -- which is what made the copy icon look
   different from its neighbours in the first place. Drawing it here gives all
   four the same styled tooltip, matching the ones Streamlit renders elsewhere
   in the app. */
.kc-act::after {{
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--parchment-soft);
  color: var(--stone);
  font-family: 'Hanuman', sans-serif;
  font-size: 12px;
  line-height: 1.7;
  padding: 3px 10px;
  border-radius: 6px;
  white-space: nowrap;
  /* The page is near-white too, so the tooltip needs an edge and a shadow to
     separate from it -- colour alone won't do the work here. */
  border: 1px solid var(--rail-line);
  box-shadow: 0 2px 8px rgba(22,51,38,0.12);
  opacity: 0;
  pointer-events: none;
  transition: opacity 120ms ease;
  z-index: 1000;
}}
.kc-act:hover::after,
.kc-act:focus-visible::after {{ opacity: 1; }}

/* The diagnostics control is an anchor, and Streamlit colours markdown links
   its own link-blue with a rule that outranks .kc-act -- so it is restated
   here for the anchor specifically, or one of the four icons would be blue. */
a.kc-act, a.kc-act:hover, a.kc-act:visited {{
  color: var(--stone) !important;
  text-decoration: none !important;
}}
.kc-act svg {{ width: 18px; height: 18px; }}
.kc-act:hover {{
  background: rgba(22,51,38,0.07);
  color: var(--stone);
  opacity: 1;
}}
.kc-act:focus-visible {{
  outline: 2px solid var(--brass);
  outline-offset: 1px;
}}
/* Copy confirms itself for a moment; the reactions stay lit while chosen. */
.kc-act-copy.is-done,
.kc-act-thumb.is-on {{
  color: var(--lacquer);
  opacity: 1;
}}
/* A chosen reaction is a SOLID icon, not just a recoloured outline. The icons
   are drawn as strokes with `fill="none"`, so recolouring alone tinted the
   line and left the glyph hollow -- too quiet to read as "this is selected"
   at 18px. Filling the same shape is what makes the state obvious. */
.kc-act-thumb.is-on svg {{
  fill: currentColor;
}}

/* Streamlit wraps every element (markdown, button, etc.) in its own
   container that carries built-in vertical spacing, and a st.button's
   wrapper carries noticeably MORE of it than a plain st.markdown HTML
   block does. That's what produced a large visible gap specifically
   between the "Details" button and the disclaimer right below it --
   compare to the out-of-domain answer path, which has no button between
   the answer text and the disclaimer and looks fine as-is. The exact
   wrapper testid Streamlit uses for this has varied across versions, so
   this targets every plausible candidate rather than guessing one -- same
   defensive multi-candidate approach as scroll_to_bottom() above. */
[data-testid="stButton"],
[data-testid="element-container"]:has([data-testid="stButton"]),
[data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
  margin-bottom: -14px !important;
}}

[data-testid="stButton"] button {{
  background: transparent !important;
  border: none !important;
  color: var(--brass) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 0 !important;
  box-shadow: none !important;
}}
[data-testid="stButton"] button:hover {{ color: {BUTTON_HOVER_COLOR} !important; text-decoration: underline; }}

/* ---------- Persistent pinned chat bar (idle + generating share one bar) ----------
   st.chat_input anchors itself in Streamlit's persistent "stBottom" slot,
   which isn't reliably cleared on runs where it's skipped -- that caused an
   old input to ghost on screen. A plain st.container has no such
   special-cased persistence. Both idle/generating states render through
   the SAME st.container(key="kc_chatbar") call, with a borderless st.form
   nested inside purely for widget grouping -- the outer container is what
   carries the CSS-targetable "st-key-kc_chatbar" class reliably (a
   borderless st.form's own key does not reliably expose the same class).
   Only the inner content (text field vs status text) and the submit
   button's icon differ between states, toggled purely with CSS via
   :has(), so the outer box itself never moves or resizes. */
.st-key-kc_chatbar {{
  position: fixed !important;
  /* Centred on the chat area, not the viewport: left/right bracket the space
     beside the rail and `margin: auto` centres within that bracket, so the bar
     sits under the thread it belongs to at any rail width (including 0, when
     the rail is collapsed). Sits above the pinned disclaimer. */
  left: var(--kc-rail-w, 0px) !important;
  right: 0 !important;
  bottom: 72px !important;
  transform: none !important;
  margin: 0 auto !important;
  contain: none !important;
  filter: none !important;
  will-change: auto !important;
  z-index: 999;
  width: min(560px, calc(100vw - var(--kc-rail-w, 0px) - 48px));
}}
.st-key-kc_chatbar [data-testid="stForm"] {{
  margin: 0 !important;
}}
.st-key-kc_chatbar [data-testid="stHorizontalBlock"] {{
  align-items: center !important;
  gap: 0.5rem !important;
}}
.st-key-kc_chatbar [data-testid="column"] {{ padding: 0 !important; }}

.kc-stopbar-text {{
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  color: var(--stone);
  opacity: 0.55;
  display: flex;
  align-items: center;
  min-height: 42px;
  padding: 0 8px;

  position: relative;
  top: -5px;
}}

/* Stop-mode icon swap: detected via the .kc-stopbar-text marker that only
   exists in the form's DOM while generating. Same 42px circular button as
   the send arrow -- only the glyph inside changes. */
.st-key-kc_chatbar:has(.kc-stopbar-text) [data-testid="stFormSubmitButton"] button {{
  background: var(--lacquer) !important;
  position: relative !important;
}}
.st-key-kc_chatbar:has(.kc-stopbar-text) [data-testid="stFormSubmitButton"] button::before {{
  content: "" !important;
  position: absolute !important;
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;
  width: 13px;
  height: 13px;
  background: var(--parchment-soft);
  border-radius: 3px;
}}
.st-key-kc_chatbar:has(.kc-stopbar-text) [data-testid="stFormSubmitButton"] button:hover {{
  background: var(--lacquer-deep) !important;
}}

.kc-alert {{
  background: var(--lacquer-mist);
  border: 1px solid var(--lacquer);
  color: var(--lacquer-deep);
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 12.5px;
  padding: 8px 12px;
  border-radius: 4px;
  margin: 4px 0 10px 0;
  display: inline-block;
}}

/* ---------- Progress ladder ---------- */
.kc-ladder {{
  border-left: 2px solid var(--brass-soft);
  padding-left: 14px;
  display: flex;
  flex-direction: column;
  gap: 9px;
  margin: 6px 0 18px 0;
}}
.kc-step-row {{ display: flex; align-items: center; }}
.kc-step-row-pending {{ opacity: 0.4; }}
.kc-step-icon {{
  width: 14px; height: 14px;
  border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 9px; color: var(--ink);
  margin-right: 9px; flex-shrink: 0;
}}
.kc-step-done {{ background: var(--signal); }}
.kc-step-pending {{ background: var(--brass-soft); }}
.kc-step-active {{ background: var(--lacquer); animation: kc-pulse 1s ease-in-out infinite; }}
@keyframes kc-pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.5; transform: scale(0.75); }} }}
.kc-step-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--stone); }}
.kc-step-time {{ margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--brass); }}

/* ---------- Diagnostics page (ink register) ---------- */
.kc-diag-header {{
  text-align: center;
  border-bottom: 1px solid var(--ink-line);
  margin-top: 22px;
  padding-bottom: 14px;
  margin-bottom: 18px;
}}

.kc-diag-title {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px;
  font-weight: 700;
  color: var(--brass);
  text-align: center;
}}
.kc-diag-question {{
  font-family: 'Moul', serif;
  font-size: 20px;
  font-weight: 700;
  color: var(--brass);
}}

/* Provenance badge + legend -- each chunk card's badge is a miniature
   seal showing which retrieval engine(s) "stamped" that chunk as
   relevant: brass ring = FAISS only, lacquer ring = BM25 only, ring AND
   center both split brass/lacquer = found and confirmed by both. */
.kc-diag-src-name-row {{ display: flex; align-items: center; gap: 7px; }}
.kc-diag-src-name-row svg {{ flex-shrink: 0; }}
.kc-diag-legend {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin: 8px 0 16px 0;
}}
.kc-diag-legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 999px;
  padding: 5px 12px 5px 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.02em;
  color: var(--mist);
}}
.kc-diag-legend-item svg {{ width: 16px; height: 16px; flex-shrink: 0; }}

/* Pipeline rail -- the four SYSTEM/RETRIEVAL/CONTEXT/GENERATION blocks
   read top-to-bottom as a real sequence (question -> vector -> ranking ->
   context -> tokens), so this echoes the live progress ladder's visual
   language (connecting line + dot per stage) for that same pipeline,
   instead of presenting four unrelated cards.
   The line is a background-image gradient (not border-left) specifically
   so it can fade out softly at the bottom rather than ending in a hard
   edge. The dot is a real flex element rendered by console_block() (see
   kc-pipeline-dot below) with a deterministic negative margin computed
   from this container's own padding-left + line width, rather than a
   guessed absolute-position offset. */
.st-key-kc_diag_pipeline {{
  padding-left: 26px;
  display: flex;
  flex-direction: column;
  background: linear-gradient(to bottom, var(--brass-soft), var(--brass-soft) 85%, transparent);
  background-repeat: no-repeat;
  background-size: 2px 100%;
  background-position: left top;
}}
.kc-pipeline-title {{ display: flex; align-items: center; }}
.kc-pipeline-dot {{
  display: inline-block;
  width: 9px;
  height: 9px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--brass);
  margin-left: -30px;
  margin-right: 14px;
  box-shadow: 0 0 0 4px var(--ink), 0 0 8px rgba(177, 159, 80, 0.45);
}}

.kc-console-block {{
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 14px;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.kc-console-block:hover {{
  transform: translateY(-2px);
  border-color: var(--brass);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
.kc-console-title {{
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
  letter-spacing: 0.08em; color: var(--brass); text-transform: uppercase;
  margin: 6px 0 8px 0;
}}
.kc-row {{
  display: flex; justify-content: space-between; gap: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  padding: 6px 0; border-bottom: 1px dashed rgba(157,182,171,0.15);
}}
.kc-row:last-child {{ border-bottom: none; }}
.kc-row-label {{ color: var(--mist); opacity: 0.85; }}
.kc-row-value {{ color: var(--parchment-soft); text-align: right; }}

/* ---------- Diagnostics source cards (right column, ink register) ---------- */
.kc-diag-src-list {{ display: flex; flex-direction: column; gap: 10px; }}
.kc-diag-src-card {{
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 6px;
  padding: 14px 16px;
  opacity: 0;
  animation: kc-rise 0.5s ease forwards;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.kc-diag-src-card:hover {{
  transform: translateY(-2px);
  border-color: var(--brass);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
.kc-diag-src-top {{ display: flex; align-items: flex-start; gap: 12px; }}
.kc-diag-src-rank {{
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--brass);
  width: 20px; flex-shrink: 0; padding-top: 2px;
}}
.kc-diag-src-main {{ flex: 1; min-width: 0; }}
.kc-diag-src-name {{ font-family: 'Hanuman', sans-serif; font-size: 14px; color: var(--parchment-soft); line-height: 1.5; }}
.kc-diag-src-id {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--brass); margin-top: 4px; }}
.kc-diag-src-scores {{
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 92px;
}}
.kc-diag-src-score-row {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
}}
.kc-diag-src-score-key {{
  color: var(--mist);
  opacity: 0.65;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.kc-diag-src-score-val {{ color: var(--parchment-soft); }}
.kc-diag-src-score-row--rrf .kc-diag-src-score-key,
.kc-diag-src-score-row--rrf .kc-diag-src-score-val {{
  color: var(--signal);
  font-weight: 600;
  font-size: 13px;
}}
.kc-diag-src-bar {{ width: 100%; height: 5px; background: var(--ink-line); border-radius: 3px; overflow: hidden; margin-top: 10px; }}
.kc-diag-src-fill {{
  height: 100%; width: 0; border-radius: 3px;
  background: linear-gradient(90deg, var(--brass), var(--lacquer));
  animation: kc-fill 0.9s cubic-bezier(.2,.8,.2,1) forwards;
}}
@keyframes kc-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes kc-fill {{ from {{ width: 0; }} to {{ width: var(--w, 0%); }} }}

.kc-detail-grid {{
  display: grid; grid-template-columns: auto 1fr; gap: 4px 14px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; margin-bottom: 12px;
}}
.kc-detail-grid .k {{ color: var(--brass); }}
.kc-detail-grid .v {{ color: var(--parchment-soft); }}
.kc-detail-text {{
  font-family: 'Hanuman', sans-serif; font-size: 14px; line-height: 1.85; color: var(--mist);
  border-top: 1px dashed var(--ink-line); padding-top: 12px; white-space: pre-wrap;
}}

/* Full Source Details -- back to native st.expander: simple, reliable,
   guaranteed clickable. Styled to match the ink card system. The
   .kc-diag-page wrapping div used to scope these never actually contained
   the expanders in the real DOM (opening/closing a div across separate
   st.markdown calls doesn't nest the content in between), so every rule
   below was previously matching nothing. st.expander is only used on this
   one page in the whole app, so these are safe to leave unscoped. */
[data-testid="stExpander"] {{
  background: var(--ink-soft) !important;
  border: 1px solid var(--ink-line) !important;
  border-radius: 6px !important;
  margin-bottom: 10px !important;
  overflow: hidden;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
[data-testid="stExpander"]:hover {{
  transform: translateY(-2px);
  border-color: var(--brass) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
/* Force every background inside the expander to transparent first -- we
   don't know exactly which element/attribute Streamlit uses to paint the
   "open" state in this version, so rather than guess a single selector,
   blank everything out and re-apply our own color via more specific rules
   below (which win on specificity regardless of source order). */
[data-testid="stExpander"] * {{
  background-color: transparent !important;
}}
[data-testid="stExpander"] summary {{
  font-family: 'IBM Plex Mono', monospace !important;
  color: var(--parchment-soft) !important;
  font-size: 12px !important;
  padding: 12px 16px !important;
  transition: background-color 0.15s ease, color 0.15s ease;
}}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details[open] summary,
[data-testid="stExpander"] summary[aria-expanded="true"],
[data-testid="stExpander"] [aria-expanded="true"] summary {{
  background-color: rgba(177,159,80,0.12) !important;
  color: var(--brass) !important;
}}
[data-testid="stExpander"] summary svg {{ fill: var(--brass) !important; }}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
  padding: 4px 16px 16px 16px !important;
  border-top: 1px solid var(--ink-line);
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}

@media (max-width: 560px) {{
  .kc-thread {{
    padding-bottom: 168px;
  }}
  .kc-greeting {{
    font-size: 20px;
  }}
  .kc-welcome-wrap {{
    margin-top: 16vh;
  }}
}}

/* ---------- Conversation rail ----------
   The rail sits on --rail, a tint of the parchment used by the
   thread, separated by tone rather than by a change of material. Active rows
   are marked with the brass left rule used by .kc-source-chip, tying the rail
   to the rest of the system. Everything below is scoped under
   [data-testid="stSidebar"] so it can't leak into the chat surface's own
   button and input styling. */
[data-testid="stSidebar"] {{
  background: var(--rail) !important;
  border-right: 1px solid var(--rail-line);
  /* Fixed width, not resizable. The rail holds one thing -- a list of
     conversation titles -- and it is sized so the brand lockup sits on one
     line; letting it be dragged only offers the chance to break that. It is
     either open at this width or closed. Locking all three width properties
     overrides the inline width Streamlit's drag handle writes. */
  width: 290px !important;
  min-width: 290px !important;
  max-width: 290px !important;
  flex: 0 0 290px !important;
  /* Pinned to the viewport: the rail is navigation, and navigation that
     scrolls away with the conversation is navigation you have to scroll back
     to find. Sticky rather than fixed so it stays a flex sibling of the chat
     column -- fixed would take it out of flow and let the thread run beneath
     it. */
  position: sticky !important;
  top: 0 !important;
  align-self: flex-start !important;
  height: 100vh !important;
}}
/* The drag handle itself is removed by disable_rail_resize() -- it carries no
   test id and only a build-generated class name, so there is no stable
   selector to write here. */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  background: var(--rail) !important;
}}
/* Streamlit reserves a 60px band at the top of the rail for the collapse
   control alone, which is pure dead space above the brand. The band is
   collapsed to nothing and the control is pinned into the brand's own row --
   brand left, collapse right, on one line at the very top. .kc-brand keeps a
   right inset so a long title never runs under the control. */
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  position: relative !important;
}}
/* Pinned to the rail's top-right corner, level with the brand on its left --
   one row holding identity and the control that hides it. */
[data-testid="stSidebarCollapseButton"] {{
  position: absolute !important;
  /* Centred on the brand's two-line block -- level with the gap between the
     Khmer title and the English line, not with the top of the rail. */
  top: 30px !important;
  /* Aligns the GLYPH -- not the button box -- with the right end of the rail's
     divider rules, so the icon terminates the line. The glyph sits 6px inside
     its own hit area (button margin plus centring slack), so the box is pushed
     that far past the line to put the mark itself on it. The hit area still
     ends well inside the rail. */
  right: -6px !important;
  z-index: 3;
  /* Streamlit reveals this only on hover. It stays visible here: it is the
     only way to collapse the rail, and a control you have to discover by
     hovering is not one users find. */
  visibility: visible !important;
  opacity: 1 !important;
}}
[data-testid="stSidebarCollapseButton"] button {{
  padding-right: 0 !important;
}}
/* Streamlit's glyph here is a double chevron, which reads as "skip to the
   end" rather than "hide this panel". Swapped for the standard side-panel
   mark -- a panel outline with the rail picked out and a chevron pointing
   the way it will move. Drawn as a mask so it takes its colour from the
   rule above rather than carrying its own. */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {{
  width: 19px !important;
  height: 19px !important;
  font-size: 0 !important;
  background-color: var(--stone);
  -webkit-mask: var(--kc-panel-icon) center / contain no-repeat;
  mask: var(--kc-panel-icon) center / contain no-repeat;
  opacity: 0.6;
}}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding-top: 0 !important;
}}
/* Streamlit's collapse chevron is near-invisible against the rail. */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg {{
  color: var(--stone) !important;
  fill: var(--stone) !important;
  opacity: 0.55;
}}

.kc-brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 44px 16px 10px;
}}
.kc-brand .kc-logo {{ width: 34px; height: 34px; }}
.kc-brand-title {{
  font-family: 'Moul', serif;
  /* Sized to clear the drag floor on one line -- see the rail's min-width. */
  font-size: 11.5px;
  line-height: 1.7;
  color: var(--lacquer);
  white-space: nowrap;
}}
.kc-brand-sub {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 8.5px;
  letter-spacing: 0.02em;
  color: var(--brass);
  margin-top: 2px;
  white-space: nowrap;
}}

.kc-rail-label {{
  /* Hanuman, not the mono face the chat surface uses for its eyebrows: IBM
     Plex Mono has no Khmer coverage, so a mono label renders through a
     fallback face at a size that clips the diacritics. */
  font-family: 'Hanuman', sans-serif;
  font-size: 11.5px;
  line-height: 2;
  letter-spacing: 0.04em;
  color: var(--brass);
  padding: 14px 10px 10px 10px;
  border-top: 1px solid var(--rail-line);
  margin-top: 10px;
}}
/* Centred and set down from the heading: with no rows to head, a line hugging
   the left edge reads as a truncated first entry rather than as the state of
   an empty list. */
.kc-rail-empty {{
  font-family: 'Hanuman', sans-serif;
  font-size: 13px;
  line-height: 1.7;
  color: var(--stone);
  opacity: 0.55;
  padding: 34px 10px 6px 10px;
  text-align: center;
}}

/* Row buttons: tertiary = an inactive conversation, primary = the open one. */
[data-testid="stSidebar"] .stButton button {{
  font-family: 'Hanuman', sans-serif !important;
  font-size: 13.5px !important;
  font-weight: 400 !important;
  text-align: left !important;
  justify-content: flex-start !important;
  /* Khmer stacks diacritics above and below the baseline; the tighter
     line-heights used on Latin UI text clip them. */
  line-height: 1.9 !important;
  padding: 6px 10px !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  /* The chat surface styles buttons as mono uppercase brass links; the rail's
     rows are navigation carrying Khmer sentences, so both of those inherited
     text treatments have to be explicitly undone here. */
  letter-spacing: 0 !important;
  text-transform: none !important;
  transition: background 120ms ease, color 120ms ease;
}}
/* Undo the chat surface's negative button margin, which exists to tighten the
   "មើលព័ត៌មានលម្អិត" link under an answer and would overlap the rail's rows. */
[data-testid="stSidebar"] [data-testid="stButton"],
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
  margin-bottom: 0 !important;
}}
/* Long Khmer titles are truncated with an ellipsis rather than wrapping, so
   every row is the same height. The line box has to clear Khmer's stacked
   diacritics first -- at a tighter line-height, `overflow: hidden` crops
   them. */
[data-testid="stSidebar"] .stButton button p {{
  /* Streamlit sets the font on this inner <p>, not on the button, so the
     button-level family never reaches the label -- Khmer fell back to a Latin
     face that has no room for its stacked diacritics and cropped them. */
  font-family: 'Hanuman', sans-serif !important;
  font-size: 13.5px !important;
  line-height: 2.1 !important;
}}
/* Streamlit wraps the label in two nested flex divs that centre it, which is
   what actually positions the text -- text-align on the button never reaches
   it. The rows are nav targets, so the label starts at the row's own padding
   edge, in line with the section heading above it. */
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button > div > span {{
  width: 100% !important;
  justify-content: flex-start !important;
}}
[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {{
  width: 100% !important;
  text-align: left !important;
}}
[data-testid="stSidebar"] .stButton button p {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
[data-testid="stSidebar"] .stButton button[kind="tertiary"] {{
  background: transparent !important;
  color: var(--stone) !important;
  border: none !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
  /* The chat surface underlines its buttons because they read as links; the
     rail's rows are full-width nav targets and use a fill instead. */
  text-decoration: none !important;
  background: transparent !important;
}}

/* ---- Conversation row ----
   Hover and selection are painted on the row wrapper, so the fill runs the
   full width of the row and the overflow menu sits inside it rather than
   stranded on the rail beside it. Selection is the fill alone -- an extra
   rule down the edge was one marker too many for a two-state list. */
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_"] {{
  border-radius: 8px;
  transition: background 120ms ease;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_"]:hover {{
  background: var(--rail-hover);
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_active"],
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_active"]:hover {{
  background: var(--lacquer-mist);
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_active"] .stButton button {{
  color: var(--lacquer-deep) !important;
  font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stButton button:focus-visible {{
  outline: 2px solid var(--brass) !important;
  outline-offset: 1px;
}}

/* New chat is the same shape as a conversation row -- it belongs to the same
   list and does the same kind of thing -- distinguished only by the accent
   colour and the leading +. A solid full-width CTA here would outweigh the
   conversations it sits above. */
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] {{
  margin-top: 10px !important;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button {{
  color: var(--lacquer) !important;
  font-weight: 500 !important;
  align-items: center !important;
}}
/* The label carries a 2.1 line-height so Khmer diacritics clear; the icon
   does not, so without an explicit centre the two sit on different lines. */
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button > div,
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button > div > span {{
  align-items: center !important;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button p {{
  overflow: visible !important;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button:hover {{
  background: var(--rail-hover) !important;
  color: var(--lacquer-deep) !important;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_new"] button [data-testid="stIconMaterial"] {{
  font-size: 17px !important;
  line-height: 2.1 !important;
  margin-right: 2px;
}}

[data-testid="stSidebar"] .stButton button:disabled {{
  opacity: 0.45 !important;
}}

/* Per-row overflow menu. Hidden until the row is hovered -- a column of dots
   down a list of titles is noise, and the menu is only ever wanted for the row
   the pointer is already on. It stays visible while its own menu is open
   (aria-expanded), or the trigger would vanish under the panel it opened, and
   while focused, so keyboard users can reach it at all. */
[data-testid="stSidebar"] [data-testid="stPopover"] button {{
  background: transparent !important;
  color: var(--stone) !important;
  border: 1px solid transparent !important;
  padding: 7px 4px !important;
  min-height: 0 !important;
  justify-content: center !important;
  opacity: 0;
  transition: opacity 120ms ease;
}}
[data-testid="stSidebar"] [class*="st-key-kc_rail_row_"]:hover [data-testid="stPopover"] button,
[data-testid="stSidebar"] [data-testid="stPopover"] button[aria-expanded="true"],
[data-testid="stSidebar"] [data-testid="stPopover"] button:focus-visible {{
  opacity: 1;
}}
/* The popover's dropdown chevron doubles up with the "⋯" glyph that already
   says "more actions". */
[data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stIconMaterial"] {{
  display: none !important;
}}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {{
  background: var(--rail-hover) !important;
  color: var(--lacquer) !important;
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
  background: var(--parchment-soft) !important;
  color: var(--stone) !important;
  border: 1px solid var(--rail-line) !important;
  font-family: 'Hanuman', sans-serif !important;
  font-size: 13px !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
  gap: 2px !important;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: 2px !important;
}}
/* Menu items: icon, then label, left aligned, one per line. */
[class*="st-key-kc_rail_rename_open_"] [data-testid="stButton"],
[class*="st-key-kc_rail_delete_"] [data-testid="stButton"] {{
  margin-bottom: 0 !important;
}}
[class*="st-key-kc_rail_rename_open_"] button,
[class*="st-key-kc_rail_delete_"] button {{
  background: transparent !important;
  border: none !important;
  min-height: 36px !important;
  font-family: 'Hanuman', sans-serif !important;
  font-size: 13.5px !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  text-align: left !important;
  justify-content: flex-start !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 7px 10px !important;
  border-radius: 7px !important;
  color: var(--stone) !important;
}}
[class*="st-key-kc_rail_rename_open_"] button p,
[class*="st-key-kc_rail_delete_"] button p {{
  font-family: 'Hanuman', sans-serif !important;
  font-size: 13.5px !important;
  line-height: 2.1 !important;
}}
[class*="st-key-kc_rail_rename_open_"] button:hover {{
  background: var(--rail-hover) !important;
  color: var(--stone) !important;
  text-decoration: none !important;
}}
/* Delete is the only destructive item on the rail, and it reads as one.
   Declared after the shared menu-item colour above so it wins. */
[class*="st-key-kc_rail_delete_"] button,
[class*="st-key-kc_rail_delete_"] button:hover {{
  color: var(--danger) !important;
}}
[class*="st-key-kc_rail_delete_"] button:hover {{
  background: rgba(192,82,74,0.10) !important;
  text-decoration: none !important;
}}
[class*="st-key-kc_rail_delete_"] button [data-testid="stIconMaterial"] {{
  color: var(--danger) !important;
}}
[class*="st-key-kc_rail_rename_open_"] button [data-testid="stIconMaterial"],
[class*="st-key-kc_rail_delete_"] button [data-testid="stIconMaterial"] {{
  font-size: 17px !important;
  opacity: 0.7;
}}

/* Inline rename editor -- sits exactly where the row's title was. */
[data-testid="stSidebar"] [class*="st-key-kc_rail_editing_"] input {{
  padding: 5px 9px !important;
  border: 1px solid var(--lacquer) !important;
  border-radius: 8px !important;
  box-shadow: 0 0 0 3px var(--lacquer-mist) !important;
}}

/* The popover panel ships with padding sized for form content; a two-item
   menu only needs enough to keep the rows off the edges. */
[data-testid="stPopoverBody"] {{
  padding: 8px !important;
  min-width: 0 !important;
}}
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {{
  gap: 4px !important;
}}
[data-testid="stPopoverBody"] [data-testid="stElementContainer"] {{
  margin-bottom: 0 !important;
}}

</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


import streamlit.components.v1 as components

components.html(
    """
    <script>
    window.parent.document.addEventListener(
        'keydown',
        function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
                e.stopImmediatePropagation();
            }
        },
        true
    );
    </script>
    """,
    height=0,
)


def sync_rail_width():
    """Publishes the rail's live width to the parent document as --kc-rail-w.

    The fixed-position top bar, chat bar, and disclaimer are positioned
    against the chat area rather than the viewport, which means they need a
    number CSS cannot derive on its own: the sidebar's width is set by
    Streamlit at runtime, changes when the user drags or collapses it, and is
    animated in between. A ResizeObserver covers drag and collapse; the rAF
    pass that follows a transitionstart covers the animation frames in
    between, so the bars travel with the rail instead of snapping at the end.

    Runs inside a 0-height components.html iframe, so it reaches back out to
    window.parent -- same approach as the keydown listener and
    scroll_to_bottom() below."""
    components.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          function measure() {
            var rail = doc.querySelector('[data-testid="stSidebar"]');
            // Collapsed rails are translated off-screen rather than removed,
            // so the right edge -- clamped at 0 -- is what the chat area
            // actually starts at, in both states.
            var w = rail ? Math.max(0, rail.getBoundingClientRect().right) : 0;
            doc.documentElement.style.setProperty('--kc-rail-w', w + 'px');
          }
          function track() {
            var start = null;
            function step(t) {
              if (start === null) start = t;
              measure();
              if (t - start < 600) window.requestAnimationFrame(step);
            }
            window.requestAnimationFrame(step);
          }
          measure();
          var rail = doc.querySelector('[data-testid="stSidebar"]');
          if (rail) {
            new window.parent.ResizeObserver(measure).observe(rail);
            rail.addEventListener('transitionstart', track);
          }
          // The collapse control lives outside the rail, and clicking it is
          // the one case where the rail is about to move before any observer
          // on it has fired.
          doc.addEventListener('click', function (e) {
            if (e.target.closest('[data-testid="stSidebarCollapseButton"], '
                                 + '[data-testid="stSidebarCollapsedControl"]')) {
              track();
            }
          }, true);
          window.parent.addEventListener('resize', measure);
        })();
        </script>
        """,
        height=0,
    )


def disable_rail_resize():
    """Removes Streamlit's sidebar drag handle.

    The rail is a fixed width by design, so the handle does nothing but put a
    col-resize cursor on the border and invite a drag that CSS then refuses.
    It cannot be hidden from the stylesheet: Streamlit gives it no data-testid
    and only a build-generated emotion class that changes between releases.
    What it does have, unmistakably, is `cursor: col-resize` -- so it's found
    by that computed style instead, which no version bump can rename.

    A MutationObserver re-runs the sweep because Streamlit re-mounts chrome on
    rerun, which would otherwise restore the handle after the first pass."""
    components.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          function sweep() {
            var all = doc.querySelectorAll('div');
            for (var i = 0; i < all.length; i++) {
              var el = all[i];
              if (el.dataset.kcResizeOff) continue;
              var cur = window.parent.getComputedStyle(el).cursor;
              if (cur === 'col-resize' || cur === 'ew-resize') {
                el.dataset.kcResizeOff = '1';
                el.style.setProperty('display', 'none', 'important');
                el.style.setProperty('pointer-events', 'none', 'important');
                el.style.setProperty('cursor', 'default', 'important');
              }
            }
          }
          sweep();
          var pending = null;
          new window.parent.MutationObserver(function () {
            if (pending) return;
            pending = window.setTimeout(function () { pending = null; sweep(); }, 80);
          }).observe(doc.body, {childList: true, subtree: true});
        })();
        </script>
        """,
        height=0,
    )


def scroll_to_bottom():
    """Scrolls the actual app viewport (not this throwaway iframe) down to
    the latest content. Runs inside a 0-height components.html iframe, so it
    has to reach back out to window.parent to find Streamlit's real
    scrolling element -- same trick as the keydown listener above.

    Streamlit's actual scrolling container has varied across versions
    ("stAppViewContainer", "stMain", plain document scroll...), so rather
    than guess one selector, this just scrolls every plausible candidate
    plus the window itself. Harmless no-ops for whichever aren't actually
    scrollable."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const win = window.parent;
            const candidates = [
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('.main'),
                doc.scrollingElement,
                doc.body,
            ];
            candidates.forEach(function(el) {
                if (el) {
                    try { el.scrollTo({ top: el.scrollHeight, behavior: "auto" }); }
                    catch (e) {}
                }
            });
            try { win.scrollTo({ top: doc.body.scrollHeight, behavior: "auto" }); }
            catch (e) {}
        })();
        </script>
        """,
        height=0,
    )


# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------


def count_tokens(text):
    """Real token count via llama-server's native /tokenize endpoint (not an
    estimate -- character-based heuristics are especially wrong for Khmer).
    Returns None if the server doesn't support it / is unreachable."""
    tokenize_url = env_value("GENERATOR_TOKENIZE_URL")
    if not tokenize_url and not USES_LOCAL_LLAMA_SERVER:
        return None
    tokenize_url = tokenize_url or f"{LLAMA_SERVER_ROOT}/tokenize"
    try:
        resp = requests.post(tokenize_url, json={"content": text}, timeout=10)
        resp.raise_for_status()
        tokens = resp.json().get("tokens", [])
        return len(tokens)
    except Exception:
        return None


def choose_max_tokens_live(context_results, prompt_tokens=None):
    """Per-query output cap for production (no expected answer available).

    Sizes the cap to the LARGEST retrieved chunk's token length x headroom --
    the answer is grounded in the retrieved context, so its faithful length is
    bounded by its biggest source chunk. Clamped to [MAX_TOKENS_FLOOR,
    SERVER_OUTPUT_HARD_CAP] and, when prompt_tokens is known, further tightened
    so prompt + output stays within the server context window (a hard guard
    against slow context-shifting). Falls back to the floor if tokenization is
    unavailable. Mirrors scripts/13's choose_max_tokens, but keyed off the
    retrieved chunks instead of a gold answer."""
    chunk_tokens = []
    for _score, chunk in context_results:
        t = count_tokens(chunk.get("text") or "")
        if t:
            chunk_tokens.append(t)

    base = max(chunk_tokens) if chunk_tokens else 0
    if not base:
        max_tokens = MAX_TOKENS_FLOOR
    else:
        want = int(base * GEN_TOKEN_HEADROOM)
        max_tokens = max(MAX_TOKENS_FLOOR, min(want, SERVER_OUTPUT_HARD_CAP))

    if prompt_tokens:
        room = SERVER_CTX_SIZE - prompt_tokens - PROMPT_TOKEN_MARGIN
        if room < max_tokens:
            max_tokens = max(MAX_TOKENS_FLOOR, room)

    return min(max_tokens, DEMO_OUTPUT_HARD_CAP)


GENERATED_SOURCE_SECTION_PATTERN = re.compile(
    r"^\s*(?:មើលប្រភពឯកសារ|ប្រភពឯកសារ|ប្រភព|Sources?|ឯកសារយោង)(?:\s*[:：-]|$)",
    re.IGNORECASE,
)
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


def _dedupe_consecutive_lines(lines):
    deduped = []
    recent = []
    for line in lines:
        key = re.sub(r"\s+", "", line)
        if key and key in recent[-3:]:
            continue
        deduped.append(line)
        if key:
            recent.append(key)
    return deduped


def _dedupe_repeated_sentences(text):
    """Remove short local-model loops without deleting legitimate lists."""
    pieces = re.split(r"([។៕!?])", text)
    sentences = []
    for i in range(0, len(pieces), 2):
        body = pieces[i]
        punct = pieces[i + 1] if i + 1 < len(pieces) else ""
        if body or punct:
            sentences.append(body + punct)

    out = []
    recent = []
    for sentence in sentences:
        key = re.sub(r"\s+", "", sentence)
        if len(key) > 30 and key in recent[-4:]:
            continue
        out.append(sentence)
        if key:
            recent.append(key)
    return "".join(out)


def _trim_to_last_complete_sentence(text):
    """Hide a dangling tail when generation stopped mid-sentence.

    This does not force a period onto incomplete text. It keeps the last
    complete Khmer sentence when doing so preserves most of the answer, which
    reads better and is less misleading during a live demo.
    """
    text = text.strip()
    if not text or text.endswith(("។", "៕", ".", "!", "?")):
        return text
    last = max(text.rfind("។"), text.rfind("៕"))
    if last >= 0 and last >= len(text) * 0.55:
        return text[: last + 1].strip()
    return text


def normalize_answer_text(text, final=True):
    """Clean model output before display/storage."""
    if not text:
        return ""
    cleaned_lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if HTML_ARTIFACT_PATTERN.search(line):
            break
        if GENERATED_SOURCE_SECTION_PATTERN.search(line):
            break
        cleaned_lines.append(line)
    cleaned_lines = _dedupe_consecutive_lines(cleaned_lines)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-•]\s*$\n?", "", cleaned)
    cleaned = _dedupe_repeated_sentences(cleaned)
    cleaned = fix_common_generation_typos(cleaned)
    if final:
        cleaned = _trim_to_last_complete_sentence(cleaned)
    return cleaned.strip()


def fix_common_generation_typos(text):
    """Correct recurring Khmer term corruption from local generation.

    These strings are not present in the corpus; they are model-side spelling
    drift when English insurance class labels are rewritten into Khmer. Keep
    this narrow so real source text is not broadly rewritten.
    """
    replacements = {
        "ធានរបងខាងតូច": "ធានារ៉ាប់រងខ្នាតតូច",
        "ធានរបងឯកជន": "ធានារ៉ាប់រងអាយុជីវិត",
        "ធានរបងទូទៅ": "ធានារ៉ាប់រងទូទៅ",
        "ធានរបង": "ធានារ៉ាប់រង",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


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


def has_repetition_loop(text):
    """Detect runaway repeated spans while streaming."""
    compact = re.sub(r"\s+", " ", normalize_answer_text(text, final=False))
    if len(compact) < 220:
        return False
    for size in (45, 65, 90, 120):
        if len(compact) < size * 4:
            continue
        tail = compact[-size:]
        if len(tail.strip()) >= size * 0.8 and compact.count(tail) >= 4:
            return True
    sentences = [s.strip() for s in re.split(r"[។៕!?]", compact) if len(s.strip()) > 35]
    return len(sentences) >= 5 and len(set(sentences[-5:])) <= 2


def render_answer_html(answer, final=True):
    return f'<div class="kc-msg-assistant">{html.escape(normalize_answer_text(answer, final=final))}</div>'


def get_loaded_model():
    if GENERATOR_MODEL:
        return GENERATOR_MODEL
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["id"]
    except Exception:
        return "No Model Loaded"


@st.cache_resource(show_spinner=False)
def resolve_vector_db():
    """Resolves the FAISS index, chunk metadata and embedding model info
    for ACTIVE_EMBEDDING_MODEL_DIR.

    There is exactly one supported layout: the per-model subfolder
    vector_db/<ACTIVE_EMBEDDING_MODEL_DIR>/ that
    scripts/08_generate_embeddings.py builds. The embedding model is not
    selectable at runtime -- BGE-M3 won the comparison in
    reports/embedding_model_comparison/ and is what the app ships with.

    This used to fall back to two older flat layouts and, failing those,
    to *assuming* BGE-M3 with a guessed hf_id. That was a silent
    correctness hazard: a wrong guess encodes queries with a different
    model than built the index, which doesn't error -- it just returns
    quietly bad retrievals. Anything short of a complete, current index
    is now a hard failure with the command that rebuilds it.

    Returns (faiss_path, metadata_path, embedding_model_info)."""
    db_dir = f"{VECTOR_DB_ROOT}/{ACTIVE_EMBEDDING_MODEL_DIR}"
    faiss_path = f"{db_dir}/khmer_insurance.index"
    metadata_path = f"{db_dir}/khmer_insurance_metadata.json"
    info_path = f"{db_dir}/embedding_model_info.json"

    missing = [
        os.path.basename(p)
        for p in (faiss_path, metadata_path, info_path)
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"{db_dir}/ is missing {', '.join(missing)}. "
            f"Run: python scripts/08_generate_embeddings.py"
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return faiss_path, metadata_path, info


@st.cache_resource(show_spinner=False)
def load_embedding_model_info():
    _, _, info = resolve_vector_db()
    return info


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    info = load_embedding_model_info()
    kwargs = {
        "trust_remote_code": bool(info.get("trust_remote_code")),
        "local_files_only": EMBEDDING_LOCAL_FILES_ONLY,
    }
    return SentenceTransformer(info["hf_id"], device="cpu", **kwargs)


@st.cache_resource
def load_index():
    faiss_path, _, _ = resolve_vector_db()
    import faiss

    faiss.omp_set_num_threads(1)
    return faiss.read_index(faiss_path)


@st.cache_resource
def load_metadata():
    _, metadata_path, _ = resolve_vector_db()
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_bm25_index(_chunks):
    """Builds the lexical BM25 index in memory at startup, directly from
    the same chunk metadata FAISS already loads. Unlike the FAISS index
    (built once offline by scripts/08_generate_embeddings.py and shipped
    as a committed artifact in vector_db/), BM25 over a few hundred chunks
    tokenizes and indexes in well under a second -- there's no real
    benefit to persisting it to disk, and building it fresh here means it
    can never silently go stale relative to the corpus the way a separate
    committed BM25 artifact could if someone regenerates embeddings but
    forgets to rebuild it.

    The leading underscore on `_chunks` is Streamlit's convention for
    cache_resource arguments that shouldn't be hashed for the cache key
    (the chunk list is large; hashing it on every rerun would be
    wasteful) -- it's still used completely normally inside the function.
    """
    corpus_tokens = [
        khmer_aware_tokenize(chunk.get("embedding_text") or chunk["text"])
        for chunk in _chunks
    ]
    return BM25Okapi(corpus_tokens)


def encode_query(query, embedding_model, embedding_model_info):
    """Encodes the live user question as a QUERY. For e5-style models
    (multilingual-e5-large), prepends 'query: ' prefix to the raw question."""
    embed_start = time.time()
    if embedding_model_info.get("query_prefix"):
        query = embedding_model_info["query_prefix"] + query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )
    embed_time = time.time() - embed_start
    return query_embedding, embed_time


def search_faiss(query_embedding, index, chunks):
    search_start = time.time()
    scores, indices = index.search(
        query_embedding,
        k=TOP_K,
    )
    search_time = time.time() - search_start
    results = [(score, chunks[idx]) for score, idx in zip(scores[0], indices[0])]
    return results, search_time


def search_bm25(query, bm25_index, chunks):
    search_start = time.time()
    tokenized_query = khmer_aware_tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
    results = [(float(scores[i]), chunks[i]) for i in top_indices]
    search_time = time.time() - search_start
    return results, search_time


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Combines multiple ranked result lists -- each a list of (score,
    chunk) tuples already sorted best-first -- into one fused ranking via
    Reciprocal Rank Fusion: fused_score(chunk) = sum(1 / (k + rank)) over
    every list the chunk appears in, where rank is that chunk's 1-indexed
    position within each individual list.

    RRF is used instead of blending the raw scores directly because FAISS
    cosine similarity and BM25 scores live on completely different,
    mutually incomparable numeric scales -- min-max normalizing and
    weighting them against each other would need its own from-scratch
    calibration (exactly the kind of fragile, corpus-specific tuning that
    already cost real iteration here once already, see the
    OUT_OF_DOMAIN_THRESHOLD calibration log above). RRF only cares about
    RANK POSITION within each individual list, which needs no calibration
    at all and is the standard, well-established way to fuse rankings
    from heterogeneous retrievers.

    IMPORTANT: the fused score this returns is NOT a similarity score and
    is not comparable in scale to either FAISS cosine scores or BM25
    scores -- it must never be used for the OUT_OF_DOMAIN_THRESHOLD /
    MIN_SCORE safety gates, which stay calibrated against raw FAISS scores
    only (see run_pipeline_in_thread -- best_score/min_score/spread_score
    are computed from the raw FAISS `results` list before fusion ever
    happens). Fusion only changes which chunks get selected as context."""
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[cid], chunk_lookup[cid]) for cid in ranked_chunk_ids]


def build_context(results):
    context_start = time.time()
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
    context = "\n\n".join(context_parts)
    context_length = len(context)
    context_time = time.time() - context_start
    return context, context_length, context_time


# Per-model style correction. Sailor2 alone tends to dress answers up with an
# invented bold heading, markdown emphasis, injected source citations/dates
# pulled from chunk metadata (e.g. "(មាត្រា ១)", "(2023-10-17)"), and a literal
# closing tag like "(ចប់) ៕" -- which reads as embellishment next to the plain
# gold answer. This extra rule block is appended to the prompt ONLY for Sailor,
# so the base prompt the other models see stays byte-for-byte unchanged. Keyed
# on a case-insensitive "sailor" substring so it matches both the eval label
# ("Sailor2-L 8B") and the app's model id ("Sailor2-L-8B-Chat-Q4_K_M.gguf"). Must
# stay identical to scripts/13_generate_model_answers.py.
SAILOR_STYLE_RULES = """
- Do NOT add a title or heading, do NOT use bold or any markdown formatting, and do NOT cite the source document's name, article/provision number, or any date (for example, do NOT write "(មាត្រា ១)", "(2023-10-17)", or "(តាម គោលការណ៍ណែនាំ...)"). Answer in plain Khmer prose, or a plain numbered/dashed list, exactly as the source presents it.
- Do NOT write any closing word or tag such as "(ចប់)", "(បញ្ចប់)", or a trailing "៕". End only with ។ on the final sentence.
- Do NOT add framing, emphasis, or any information that is not in the context. Match the wording and scope of the source -- do not expand, summarize away, or restructure its meaning."""


def _is_sailor(model_id):
    return "sailor" in (model_id or "").lower()


def build_prompt(question, context, low_confidence=False, model_id=""):
    relevance_check = ""
    if low_confidence:
        # Triggered when best_score sits below MIN_SCORE but above
        # OUT_OF_DOMAIN_THRESHOLD -- i.e. retrieval wasn't confident enough
        # to reject outright, but isn't a clean match either. The risk in
        # this band specifically isn't "no context was found" (the model
        # already handles that case via the rule below) -- it's that the
        # retrieved chunks share VOCABULARY with the question without
        # sharing its actual TOPIC (e.g. a meta-question about the
        # assistant itself landing on legal complaint-procedure documents
        # purely because both use the word "answer/respond"). A 7B model
        # under low temperature tends to pattern-match surface phrasing
        # from such chunks into a confident-sounding but wrong answer
        # rather than recognizing the mismatch, so this band gets an
        # explicit instruction to check topical alignment before relying
        # on the context at all.
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
- Use the source wording as much as possible. Prefer copying the relevant Khmer phrase/sentence from the context instead of paraphrasing it.
- If only one source directly answers the question, use only that source. Do not combine other sources just because they were retrieved.
- Provide a concise professional answer. If the context lists multiple directly relevant conditions, sub-points, exceptions, or requirements, enumerate them once and keep their order.
- Do not repeat the same sentence, phrase, list item, or explanation.
- Do not write a "Sources", "ប្រភព", "មើលប្រភពឯកសារ", citation, disclaimer, HTML, button text, or UI/action text. The application will show sources separately after your answer.
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


# A line in the corpus is the START of a new structural block (heading, a
# numbered item "១.", a lettered item "ក.", a bullet) -- versus a PDF
# line-wrap artifact in the middle of a sentence. Verbatim provision text is
# stored with a hard "\n" at every PDF line break, so without this distinction
# every wrapped line renders as its own jagged line under `white-space:
# pre-wrap`. We keep the break before structural lines and fold the rest.
LIST_MARKER_PATTERN = re.compile(
    r"^(?:[០-៩0-9]+\s*[.)៖]|[ក-អ]\s*[.)]|មាត្រា|ប្រការ|[-•])"
)


def clean_provision_text(text):
    """Fold PDF line-wrap newlines inside the verbatim provision text into
    spaces, while preserving the breaks that begin a heading or list item, so
    the deterministic provision answer reads as clean paragraphs/lists instead
    of one hard break per wrapped line."""
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


SOURCE_NOISE_PATTERN = re.compile(
    r"^(?:Source file:|OCR method:|\[Page \d+\]|https?://|data-tip=|aria-label=)",
    re.IGNORECASE,
)


def clean_reference_text(text):
    """Prepare retrieved source text for direct, faithful display."""
    lines = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or SOURCE_NOISE_PATTERN.search(line):
            continue
        if HTML_ARTIFACT_PATTERN.search(line):
            break
        lines.append(line)
    cleaned = clean_provision_text("\n".join(lines))
    cleaned = normalize_answer_text(cleaned)
    return cleaned


def find_context_by_doc_id(context_results, document_id):
    for _score, chunk in context_results:
        if chunk.get("metadata", {}).get("document_id") == document_id:
            return chunk
    return None


def find_contexts_by_doc_id(context_results, document_id):
    return [
        chunk
        for _score, chunk in context_results
        if chunk.get("metadata", {}).get("document_id") == document_id
    ]


def question_matches_management(question):
    return bool(
        re.search(
            r"(ថ្នាក់ដឹកនាំ|រចនាសម្ព័ន្ធ|management)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_publication(question):
    return bool(
        re.search(
            r"(publication|ការបោះពុម្ព|បោះពុម្ពផ្សាយ|របាយការណ៍ប្រចាំឆ្នាំ|annual report)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_policy_count(question):
    return bool(
        re.search(
            r"(ចំនួនបណ្ណសន្យា|បណ្ណសន្យារ៉ាប់រង|number of polic|policy count)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_insurance_company_count(question):
    if not re.search(r"(ប៉ុន្មាន|ចំនួន|how many|number of)", question, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(គ្រឹះស្ថានធានារ៉ាប់រង|ក្រុមហ៊ុនធានារ៉ាប់រង|"
            r"insurance institution|insurance compan)",
            question,
            re.IGNORECASE,
        )
    )


def question_matches_management_profile(question):
    if not re.search(
        r"(ប្រវត្តិ|ជីវប្រវត្តិ|background|biography|profile|cv)",
        question,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"(ប៊ូ\s*ច័ន្ទភីរូ|ច័ន្ទភីរូ|Bou\s*Chanphirou|Chanphirou)",
            question,
            re.IGNORECASE,
        )
    )


def find_result_by_doc_id(results, document_id):
    for score, chunk in results:
        if chunk.get("metadata", {}).get("document_id") == document_id:
            return score, chunk
    return None


def find_results_by_doc_id(results, document_id, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        if chunk.get("metadata", {}).get("document_id") != document_id:
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_bou_chanphirou_profile_results(results, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("document_id", "")
        doc_name = metadata.get("document_name", "")
        source_url = metadata.get("source_url", "")
        is_profile_doc = (
            doc_id == "website_file_ceab89ff2b0c"
            or "Bou Chanphirou_ Updataed Biography.pdf" in doc_name
            or "Bou Chanphirou_ Updataed Biography.pdf" in source_url
        )
        if not is_profile_doc:
            continue
        haystack = " ".join(
            str(metadata.get(key, ""))
            for key in ("document_id", "document_name", "source_url")
        )
        haystack += " " + (chunk.get("text") or "")[:400]
        if "Bou Chanphirou" not in haystack and "BOU Chanphirou" not in haystack:
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_policy_count_results(results, limit=None):
    matches = []
    seen = set()
    required_terms = [
        "ចំនួនបណ្ណសន្យារ៉ាប់រង",
        "ធានារ៉ាប់រងទូទៅ",
        "ធានារ៉ាប់រងអាយុជីវិត",
        "ធានារ៉ាប់រងខ្នាតតូច",
        "256,460",
        "208,450",
        "1,043,565",
        "952,759",
    ]
    for score, chunk in results:
        text = chunk.get("text") or ""
        if not all(term in text for term in required_terms):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


POLICY_COUNT_SOURCE_DOC_ID = "website_file_386481a18166"
INSURANCE_COMPANY_COUNT_SOURCE_DOC_ID = "website_file_db047ce07365"


def find_policy_count_results_in_corpus(limit=1):
    corpus = globals().get("chunks", [])
    candidates = [
        (1.0, chunk)
        for chunk in corpus
        if chunk.get("metadata", {}).get("document_id") == POLICY_COUNT_SOURCE_DOC_ID
    ]
    return find_policy_count_results(candidates, limit=limit)


def _is_insurance_company_count_chunk(chunk):
    text = chunk.get("text") or ""
    return (
        chunk.get("metadata", {}).get("document_id")
        == INSURANCE_COMPANY_COUNT_SOURCE_DOC_ID
        and "ក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន" in text
        and ("៣៩" in text or "39" in text)
    )


def find_insurance_company_count_results(results, limit=None):
    matches = []
    seen = set()
    for score, chunk in results:
        if not _is_insurance_company_count_chunk(chunk):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append((score, chunk))
        if limit and len(matches) >= limit:
            break
    return matches


def find_insurance_company_count_results_in_corpus(limit=1):
    corpus = globals().get("chunks", [])
    candidates = [(1.0, chunk) for chunk in corpus if _is_insurance_company_count_chunk(chunk)]
    return find_insurance_company_count_results(candidates, limit=limit)


def curated_candidate_results(question, results):
    """Official website facts that are clean enough to protect in context.

    These are intentionally narrow. The regular threshold still handles normal
    open-ended retrieval, but these structured website assets were getting
    dropped by hybrid ranking or rejected by the raw FAISS gate despite being
    the exact source for common demo questions.
    """
    candidates = []
    if question_matches_management(question):
        if result := find_result_by_doc_id(results, "website_homepage_management_structure"):
            candidates.append(result)
    if re.search(r"(បុព្វលាភ|premium|ស្ថិតិ)", question, re.IGNORECASE):
        if result := find_result_by_doc_id(
            results,
            "website_homepage_gross_premium_statistics",
        ):
            candidates.append(result)
    if question_matches_policy_count(question):
        policy_count_results = find_policy_count_results(results, limit=1)
        if not policy_count_results:
            policy_count_results = find_policy_count_results_in_corpus(limit=1)
        candidates.extend(policy_count_results)
    if question_matches_insurance_company_count(question):
        company_count_results = find_insurance_company_count_results(results, limit=1)
        if not company_count_results:
            company_count_results = find_insurance_company_count_results_in_corpus(limit=1)
        candidates.extend(company_count_results)
    if question_matches_publication(question):
        candidates.extend(
            find_results_by_doc_id(
                results,
                "website_publication_catalog",
                limit=2,
            )
        )
    if question_matches_management_profile(question):
        candidates.extend(find_bou_chanphirou_profile_results(results, limit=2))
    return candidates


def has_curated_source_candidate(question, results):
    return bool(curated_candidate_results(question, results))


def augment_curated_context(question, context_results, candidate_results):
    preferred = curated_candidate_results(question, candidate_results)
    if not preferred:
        return context_results

    merged = []
    seen = set()
    for score, chunk in preferred:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append((score, chunk))
        if len(merged) >= CONTEXT_K:
            break
    return merged


def extract_management_answer(context_results):
    chunk = find_context_by_doc_id(context_results, "website_homepage_management_structure")
    if not chunk:
        return None
    names = []
    in_khmer_list = False
    for raw in (chunk.get("text") or "").split("\n"):
        line = raw.strip()
        if line.startswith("ថ្នាក់ដឹកនាំ"):
            in_khmer_list = True
            continue
        if not in_khmer_list:
            continue
        if not line or line.startswith("Official management profile asset"):
            break
        if line.startswith("នាយកដ្ឋាន"):
            names.append(line)
        elif " - " in line:
            names.append(line)
    if not names:
        return None
    items = "\n".join(f"{i}. {name}" for i, name in enumerate(names, start=1))
    return f"ថ្នាក់ដឹកនាំ ន.ធ.ក. មានដូចខាងក្រោម៖\n{items}។"


def extract_publication_answer(question, context_results):
    chunks = find_contexts_by_doc_id(context_results, "website_publication_catalog")
    if not chunks or not question_matches_publication(question):
        return None

    chunks = sorted(chunks, key=lambda chunk: chunk.get("chunk_id", ""))
    text = "\n".join(chunk.get("text") or "" for chunk in chunks)
    annual_only = bool(
        re.search(
            r"(របាយការណ៍ប្រចាំឆ្នាំ|annual report)",
            question,
            re.IGNORECASE,
        )
    )

    annual_items = []
    other_items = []
    section = None
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("របាយការណ៍ប្រចាំឆ្នាំ") or line.startswith("Annual reports"):
            section = "annual"
            continue
        if line.startswith("ការបោះពុម្ពផ្សាយផ្សេងទៀត") or line.startswith("Other publications"):
            section = "other"
            continue
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if section == "annual":
            annual_items.append(item)
        elif section == "other":
            other_items.append(item)

    items = annual_items if annual_only else annual_items + other_items
    if not items:
        return None

    heading = (
        "របាយការណ៍ប្រចាំឆ្នាំរបស់ ន.ធ.ក. ដែលមានក្នុងគេហទំព័រ រួមមាន៖"
        if annual_only
        else "ការបោះពុម្ពផ្សាយ និងរបាយការណ៍របស់ ន.ធ.ក. ដែលមានក្នុងគេហទំព័រ រួមមាន៖"
    )
    body = "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))
    return f"{heading}\n{body}។"


POLICY_COUNT_ROW_PATTERN = re.compile(
    r"\|\s*(?:[១២៣]\.\s*)?([^|]*ធានារ៉ាប់រង[^|]+?)\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*\|"
)
POLICY_COUNT_TOTAL_PATTERN = re.compile(
    r"\|\s*សរុប\s*\|\s*([0-9,]+)(?:\.00)?\s*\|\s*([0-9,]+)(?:\.00)?\s*\|"
)


def _stat_int(value):
    return int(value.replace(",", ""))


def _stat_number(value):
    return f"{_stat_int(value):,}"


def _trend_word(current_value, previous_value):
    current = _stat_int(current_value)
    previous = _stat_int(previous_value)
    if current > previous:
        return "កើនឡើង"
    if current < previous:
        return "ថយចុះ"
    return "មិនប្រែប្រួល"


def extract_policy_count_answer(question, context_results):
    if not question_matches_policy_count(question):
        return None
    chunk = None
    for _score, candidate in context_results:
        text = candidate.get("text") or ""
        if find_policy_count_results([(_score, candidate)], limit=1):
            chunk = candidate
            break
    if not chunk:
        return None

    text = chunk.get("text") or ""
    section_start = text.find("ចំនួនបណ្ណសន្យារ៉ាប់រង")
    if section_start < 0:
        return None
    section_end = text.find("ទ្រព្យសកម្មសរុប", section_start)
    text = text[section_start : section_end if section_end >= 0 else len(text)]
    rows = []
    for label, current, previous in POLICY_COUNT_ROW_PATTERN.findall(text):
        label = re.sub(r"\s+", " ", label).strip()
        rows.append((label, current, previous))
    total_match = POLICY_COUNT_TOTAL_PATTERN.search(text)
    if total_match:
        rows.append(("សរុប", total_match.group(1), total_match.group(2)))
    if not rows:
        return None

    lines = [
        "យោងតាមតារាងចំនួនបណ្ណសន្យារ៉ាប់រង ឆ្នាំ២០២១ ប្រៀបធៀបនឹងឆ្នាំ២០២០ មានការប្រែប្រួលដូចខាងក្រោម៖"
    ]
    for idx, (label, current, previous) in enumerate(rows, start=1):
        trend = _trend_word(current, previous)
        lines.append(
            f"{idx}. {label}: {trend}ពី {_stat_number(previous)} បណ្ណ "
            f"ក្នុងឆ្នាំ២០២០ ទៅ {_stat_number(current)} បណ្ណ ក្នុងឆ្នាំ២០២១"
        )
    return "\n".join(lines) + "។"


def extract_insurance_company_count_answer(question, context_results):
    if not question_matches_insurance_company_count(question):
        return None
    chunk = None
    for _score, candidate in context_results:
        if _is_insurance_company_count_chunk(candidate):
            chunk = candidate
            break
    if not chunk:
        return None

    text = chunk.get("text") or ""
    if "ក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន ៣៩" not in text:
        return None

    return (
        "គិតត្រឹមឆ្នាំ២០២៥ ទីផ្សារធានារ៉ាប់រងកម្ពុជាមានក្រុមហ៊ុនធានារ៉ាប់រងសរុបចំនួន ៣៩ ក្នុងនោះ៖\n"
        "1. ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិតចំនួន ១៣\n"
        "2. ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅចំនួន ១៨\n"
        "3. ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចចំនួន ៧\n"
        "4. ក្រុមហ៊ុនធានារ៉ាប់រងបន្តចំនួន ១។\n"
        "ក្នុងចំណោមក្រុមហ៊ុនទាំងនេះ មានក្រុមហ៊ុនបរទេសចំនួន ២០, "
        "ក្រុមហ៊ុនក្នុងស្រុកចំនួន ១៧ និងក្រុមហ៊ុនចម្រុះរដ្ឋនិងឯកជនចំនួន ២។"
    )


def extract_bou_chanphirou_profile_answer(question, context_results):
    if not question_matches_management_profile(question):
        return None
    chunks = find_contexts_by_doc_id(context_results, "website_file_ceab89ff2b0c")
    if not chunks:
        return None
    text = "\n".join(chunk.get("text") or "" for chunk in chunks)
    if "Bou Chanphirou" not in text and "BOU Chanphirou" not in text:
        return None
    return (
        "ប្រវត្តិសង្ខេបរបស់ឯកឧត្តម ប៊ូ ច័ន្ទភីរូ មានដូចខាងក្រោម៖\n"
        "1. ឯកឧត្តមជាអគ្គនាយកនៃនិយ័តករធានារ៉ាប់រងកម្ពុជា។\n"
        "2. ក្រោយការបង្កើតអាជ្ញាធរសេវាហិរញ្ញវត្ថុមិនមែនធនាគារ នៅដើមឆ្នាំ 2021 "
        "ឯកឧត្តមត្រូវបានតែងតាំងជាអគ្គនាយកដំបូងនៃ ន.ធ.ក.។\n"
        "3. ឯកឧត្តមមានបទពិសោធន៍ដឹកនាំក្នុងវិស័យរដ្ឋាភិបាល 26 ឆ្នាំ។ ពីឆ្នាំ 1995 ដល់ 2002 "
        "ឯកឧត្តមធ្វើការនៅក្រសួងពាណិជ្ជកម្ម ជាអ្នកចរចាពាណិជ្ជកម្មសេវាកម្មក្នុងអាស៊ាន។\n"
        "4. នៅឆ្នាំ 2002 ឯកឧត្តមបានចូលរួមក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ ហើយបានបំពេញការងារ"
        "ជិត 20 ឆ្នាំក្នុងវិស័យធានារ៉ាប់រង។ មុនតួនាទីបច្ចុប្បន្ន ឯកឧត្តមជាអគ្គនាយករងទទួលបន្ទុក"
        "ធានារ៉ាប់រង នៃអគ្គនាយកដ្ឋានឧស្សាហកម្មហិរញ្ញវត្ថុ ក្រសួងសេដ្ឋកិច្ច និងហិរញ្ញវត្ថុ។\n"
        "5. ឯកឧត្តមមានតួនាទីសំខាន់ក្នុងការជំរុញការកើតឡើងនៃធានារ៉ាប់រងអាយុជីវិត និង"
        "ធានារ៉ាប់រងខ្នាតតូចក្នុងរយៈពេលជាង 10 ឆ្នាំកន្លងមក និងត្រូវបានឧស្សាហកម្មមើលឃើញថា"
        "ជាអ្នកសម្រេចចិត្តសំខាន់ម្នាក់ក្នុងការគាំទ្រកំណើនវិស័យធានារ៉ាប់រង។\n"
        "6. ផ្នែកការអប់រំ ឯកឧត្តមមាន BBA ផ្នែកពាណិជ្ជកម្មពី Faculty of Business "
        "(បច្ចុប្បន្ន NUM), MBA ផ្នែកហិរញ្ញវត្ថុពី University of Technology Sydney (UTS) "
        "ប្រទេសអូស្ត្រាលី និងបរិញ្ញាបត្រអក្សរសាស្ត្រអង់គ្លេសពី Institute of Foreign Languages។"
    )


def extract_gross_premium_answer(question, context_results):
    chunk = find_context_by_doc_id(context_results, "website_homepage_gross_premium_statistics")
    if not chunk:
        return None
    if not re.search(r"(បុព្វលាភ|premium|ស្ថិតិ)", question, re.IGNORECASE):
        return None
    year_match = re.search(r"(20[0-9]{2}|២០២[០-៩]|២០១[៨-៩])", question)
    year = to_khmer_digits(year_match.group(1)) if year_match else "២០២៥"
    text = chunk.get("text") or ""
    for line in text.split("\n"):
        if year in to_khmer_digits(line) and "Total Gross Premium" in line:
            total_match = re.search(r"Total Gross Premium\s+([0-9,]+)\s+USD", line)
            if total_match:
                return (
                    f"ចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ{year} "
                    f"គឺ {total_match.group(1)} ដុល្លារ អាមេរិក។"
                )
        if f"ឆ្នាំ{year}" in to_khmer_digits(line) and "បុព្វលាភ" in line:
            return clean_reference_text(line)
    return None


def resolve_curated_source_answer(question, context_results, best_score):
    """Narrow deterministic answers for clean, structured website facts.

    This is intentionally small and source-specific. Broad "return the raw top
    chunk" behavior made answers look worse, because chunks can include menu
    text, asset links, or adjacent material. These handlers only fire for facts
    we know are structured cleanly in the corpus.
    """
    if answer := extract_bou_chanphirou_profile_answer(question, context_results):
        return answer
    if question_matches_management(question):
        if answer := extract_management_answer(context_results):
            return answer
    if answer := extract_policy_count_answer(question, context_results):
        return answer
    if answer := extract_insurance_company_count_answer(question, context_results):
        return answer
    if answer := extract_gross_premium_answer(question, context_results):
        return answer
    if answer := extract_publication_answer(question, context_results):
        return answer
    return None


def resolve_provision_lookup(question, context_results):
    """Deterministic answer for questions that explicitly name a provision
    (មាត្រា N / ប្រការ N). Bypasses the local model, which both summarizes such
    provisions down to their topic AND hallucinates scope not in the text
    (e.g. inventing "សេវា ឬការអះអាង..." onto មាត្រា ១០), so the provision's own
    words are the most faithful, complete answer to "what does it say".

    Returns:
      - a list of verbatim provision texts, when the chunk(s) for the named
        provision(s) -- from the document the question names -- were retrieved;
      - a one-element list with a "no such provision" message, when the
        question names a document but that document has no such provision in
        context (e.g. asking for មាត្រា ១០០ of a 67-article law). This is what
        stops a same-numbered provision from ANOTHER document being returned;
      - None, when no provision was named -- fall back to the model.
    """
    refs = []
    seen = set()
    for keyword, num in ARTICLE_REF_PATTERN.findall(question):
        # Corpus headings use Khmer digits; normalize so a typed Arabic number
        # ("មាត្រា 67") matches "មាត្រា ៦៧".
        num = to_khmer_digits(num)
        if (keyword, num) not in seen:
            seen.add((keyword, num))
            refs.append((keyword, num))
    if not refs:
        return None

    qn = re.sub(r"\s+", "", question)
    # Document titles the question explicitly names, matched against the known
    # document_name of every retrieved chunk. When the question names a
    # document, a provision is only accepted if it comes from that document.
    named_docs = {
        chunk.get("metadata", {}).get("document_name", "")
        for _score, chunk in context_results
        if re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", ""))
        and re.sub(r"\s+", "", chunk.get("metadata", {}).get("document_name", "")) in qn
    }

    texts = []
    for keyword, num in refs:
        # Corpus heading: "មាត្រា ១០ - ...". Match keyword + number as a whole
        # unit; the lookahead stops "១០" from also matching "១០០"/"១០២".
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
                continue  # right number, wrong document
            match = clean_provision_text(text)
            break
        if match is not None:
            texts.append(match)
        elif named_docs:
            # Question named a document, but it has no such provision: don't
            # fall through to a same-numbered provision from another law.
            doc_title = next(iter(named_docs))
            return [
                f"រកមិនឃើញ{keyword} {num} នៅក្នុង{doc_title}ឡើយ។ {REDIRECT_TEXT}"
            ]
    return texts or None


def context_doc_key(chunk):
    metadata = chunk.get("metadata", {})
    return (
        metadata.get("document_id")
        or document_id_for_chunk(chunk.get("chunk_id", ""))
        or metadata.get("document_name")
        or "unknown"
    )


def select_context_chunks(results):
    """Cap which retrieved chunks actually go into the LLM prompt: at most
    CONTEXT_K total, and no more than MAX_CHUNKS_PER_DOC from any single
    source document. Retrieval still searches/shows the full TOP_K (so
    diagnostics stays informative), but redundant near-duplicate chunks
    from the same document were the single biggest driver of prompt size
    (and therefore prefill/first-token time) without adding real coverage."""
    if results:
        # When one document clearly dominates the fused ranking, keep the
        # prompt centered on that document instead of padding it with weaker
        # neighbours. This makes answers more faithful to the reference and
        # reduces the local model's tendency to merge unrelated sources.
        top_doc = context_doc_key(results[0][1])
        focused = [
            (score, chunk)
            for score, chunk in results[:DOC_FOCUS_LOOKAHEAD]
            if context_doc_key(chunk) == top_doc
        ]
        if len(focused) >= DOC_FOCUS_MIN_HITS:
            return focused[: min(CONTEXT_K, DOC_FOCUS_CONTEXT_K)]

    selected = []
    per_doc_count = {}
    for score, chunk in results:
        doc_key = context_doc_key(chunk)
        if per_doc_count.get(doc_key, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        selected.append((score, chunk))
        per_doc_count[doc_key] = per_doc_count.get(doc_key, 0) + 1
        if len(selected) >= CONTEXT_K:
            break
    return selected


def is_image_derived_source(chunk):
    metadata = chunk.get("metadata", {})
    source_url = (metadata.get("source_url") or "").lower()
    doc_name = (metadata.get("document_name") or "").lower()
    image_exts = (".jpg", ".jpeg", ".png", ".webp")
    return source_url.endswith(image_exts) or doc_name.endswith(image_exts)


def select_used_sources(results):
    """The chunks shown as cited sources under an answer. This receives
    context_results -- already capped to CONTEXT_K with at most
    MAX_CHUNKS_PER_DOC per document, and ranked via hybrid RRF fusion
    rather than raw FAISS cosine score (see reciprocal_rank_fusion) -- so
    it's already a small, curated, genuinely-used set. Previously this
    also filtered by how close each score was to the top result
    (USED_SOURCE_MARGIN), back when it received the full raw TOP_K FAISS
    list and needed a tighter subset. That margin was calibrated for
    cosine similarity's 0-1 scale; RRF fused scores live on a completely
    different, much smaller scale (see reciprocal_rank_fusion's docstring
    for why those scales can't be compared), so reusing that margin here
    would either over- or under-filter depending on the corpus. Simply
    taking the top few of an already-curated list sidesteps needing a new
    threshold calibrated against a fused score scale."""
    non_image_results = [
        (score, chunk)
        for score, chunk in results
        if not is_image_derived_source(chunk)
    ]
    return (non_image_results or results)[:MAX_USED_SOURCES]


def run_pipeline_in_thread(
    question,
    model_name,
    client,
    embedding_model,
    embedding_model_info,
    index,
    bm25_index,
    chunks,
    out_q,
    stop_event,
):
    """Runs entirely in a background thread. Must never call any st.* API."""
    try:
        if not contains_khmer_script(question):
            out_q.put(("phase", "completed"))
            out_q.put(
                (
                    "out_of_domain",
                    {
                        "reason": "not_khmer",
                        "results": [],
                        "best_score": 0.0,
                        "min_score": 0.0,
                        "spread_score": 0.0,
                        "embed_time": 0.0,
                        "search_time": 0.0,
                        "retrieval_time": 0.0,
                    },
                )
            )
            return

        if len(question.strip()) < MIN_QUESTION_LENGTH:
            out_q.put(("phase", "completed"))
            out_q.put(
                (
                    "out_of_domain",
                    {
                        "reason": "too_short",
                        "results": [],
                        "best_score": 0.0,
                        "min_score": 0.0,
                        "spread_score": 0.0,
                        "embed_time": 0.0,
                        "search_time": 0.0,
                        "retrieval_time": 0.0,
                    },
                )
            )
            return

        if is_scope_violation(question):
            # Deterministic, pre-retrieval check -- see SCOPE_VIOLATION_PATTERNS
            # comment block for why this can't wait for the score gate below:
            # these questions often retrieve genuinely relevant chunks (real
            # text about licensed insurers, real fee figures), so best_score
            # stays high and OUT_OF_DOMAIN_THRESHOLD never fires.
            out_q.put(("phase", "completed"))
            out_q.put(
                (
                    "out_of_domain",
                    {
                        "reason": "scope_violation",
                        "results": [],
                        "best_score": 0.0,
                        "min_score": 0.0,
                        "spread_score": 0.0,
                        "embed_time": 0.0,
                        "search_time": 0.0,
                        "retrieval_time": 0.0,
                    },
                )
            )
            return

        out_q.put(("phase", "encoding_query"))

        query_embedding, embed_time = encode_query(
            question,
            embedding_model,
            embedding_model_info,
        )

        out_q.put(("phase", "searching_faiss"))

        results, search_time = search_faiss(
            query_embedding,
            index,
            chunks,
        )

        best_score = float(results[0][0]) if results else 0.0
        min_score = float(results[-1][0]) if results else 0.0
        spread_score = best_score - min_score
        retrieval_time = embed_time + search_time

        if best_score < OUT_OF_DOMAIN_THRESHOLD and not has_curated_source_candidate(
            question, results
        ):
            # Don't waste a full generation cycle on a question this
            # corpus has nothing relevant for -- FAISS always returns k
            # results regardless of actual relevance, so a low top score
            # here is the signal, not the chunk content itself.
            #
            # This gate, and best_score/min_score/spread_score above, are
            # intentionally computed from the RAW FAISS `results` list
            # ONLY, before BM25/fusion ever run -- the OUT_OF_DOMAIN_THRESHOLD
            # and MIN_SCORE calibration log above is calibrated against raw
            # FAISS cosine scores specifically, and RRF fused scores live on
            # a different, incomparable scale (see reciprocal_rank_fusion's
            # docstring). Curated official website facts are the narrow
            # exception above: if their exact source is already in FAISS
            # candidates, they proceed to BM25/fusion so their source row and
            # deterministic answer can be built without lowering the general
            # threshold for unrelated questions.
            out_q.put(("phase", "completed"))
            out_q.put(
                (
                    "out_of_domain",
                    {
                        "reason": "low_score",
                        "results": results,
                        "best_score": best_score,
                        "min_score": min_score,
                        "spread_score": spread_score,
                        "embed_time": embed_time,
                        "search_time": search_time,
                        "retrieval_time": retrieval_time,
                    },
                )
            )
            return

        out_q.put(("phase", "searching_bm25"))
        bm25_results, bm25_time = search_bm25(question, bm25_index, chunks)

        out_q.put(("phase", "fusing_results"))
        fusion_start = time.time()
        fused_results = reciprocal_rank_fusion([results, bm25_results])
        fusion_time = time.time() - fusion_start
        hybrid_time = bm25_time + fusion_time

        # Diagnostics-only: records each chunk's raw FAISS and/or BM25
        # score (a chunk may appear in one list, the other, or both) so
        # the diagnostics page can show provenance -- which ranker(s)
        # actually surfaced this chunk -- alongside the fused RRF score.
        # This is purely a lookup built for display; it has no effect on
        # fused_results, context_results, or anything the LLM sees.
        score_lookup = {}
        for raw_score, chunk in results:
            score_lookup.setdefault(chunk["chunk_id"], {})["faiss"] = float(raw_score)
        for raw_score, chunk in bm25_results:
            score_lookup.setdefault(chunk["chunk_id"], {})["bm25"] = float(raw_score)

        out_q.put(("phase", "building_context"))
        context_results = select_context_chunks(fused_results)
        context_results = augment_curated_context(
            question,
            context_results,
            fused_results + results + bm25_results,
        )
        context, context_length, context_time = build_context(context_results)
        out_q.put(("phase", "constructing_prompt"))
        prompt = build_prompt(
            question, context,
            low_confidence=best_score < MIN_SCORE,
            model_id=model_name,
        )
        context_tokens = count_tokens(context)
        prompt_tokens = count_tokens(prompt)

        out_q.put(("phase", "preparing_response"))

        gen_start = time.time()
        first_token_time = None
        total_chars = 0
        total_tokens_out = 0
        was_stopped = False
        answer_text = ""

        # Deterministic provision lookup. When the question explicitly names a
        # provision (មាត្រា N / ប្រការ N) and we retrieved its exact chunk,
        # answer with the provision verbatim and skip the local model entirely.
        # The model can't be trusted here: it summarizes the provision down to
        # its topic AND hallucinates scope not in the text. The provision's own
        # words are the most faithful, complete answer to "what does it say".
        provision_texts = resolve_provision_lookup(question, context_results)
        if provision_texts:
            answer_text = "\n\n".join(provision_texts)
            first_token_time = time.time() - gen_start
            out_q.put(("phase", "generating_response"))
            out_q.put(("token", answer_text))
            total_chars = len(answer_text)
            total_tokens_out = len(answer_text)
        elif curated_answer := resolve_curated_source_answer(
            question, context_results, best_score
        ):
            answer_text = curated_answer
            first_token_time = time.time() - gen_start
            out_q.put(("phase", "generating_response"))
            out_q.put(("token", answer_text))
            total_chars = len(answer_text)
            total_tokens_out = len(answer_text)
        else:
            # Output cap sized to the largest retrieved chunk (see
            # choose_max_tokens_live), bounded by the context window.
            max_tokens = choose_max_tokens_live(context_results, prompt_tokens)

            # Retry transient generation failures (chiefly the intermittent
            # --jinja PEG-parser error) -- but ONLY while nothing has streamed to
            # the UI yet. Once a token is on screen we can't restart cleanly, so
            # a later failure propagates as before.
            for attempt in range(1, GEN_MAX_RETRIES + 1):
                try:
                    request_kwargs = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                        "stream": True,
                    }
                    if USES_LOCAL_LLAMA_SERVER:
                        request_kwargs["frequency_penalty"] = GEN_FREQUENCY_PENALTY

                    response = client.chat.completions.create(
                        **request_kwargs,
                    )
                    for chunk in response:
                        if stop_event.is_set():
                            was_stopped = True
                            break
                        if not getattr(chunk, "choices", None):
                            continue
                        delta_obj = getattr(chunk.choices[0], "delta", None)
                        delta = getattr(delta_obj, "content", None) if delta_obj else None
                        if delta:
                            answer_text += delta
                            total_chars += len(delta)
                            # llama-server streams one token per content delta,
                            # so this count is a direct token count.
                            total_tokens_out += 1
                            if first_token_time is None:
                                first_token_time = time.time() - gen_start
                                out_q.put(("phase", "generating_response"))
                            out_q.put(("token", delta))

                    try:
                        response.close()
                    except Exception:
                        pass
                    break  # generation succeeded
                except Exception:
                    # Safe to re-roll only if we haven't shown anything yet.
                    if (
                        first_token_time is None
                        and not answer_text
                        and attempt < GEN_MAX_RETRIES
                    ):
                        time.sleep(GEN_RETRY_BACKOFF)
                        continue
                    raise

        generation_time = time.time() - gen_start

        payload = {
            "results": results,
            "fused_results": fused_results,
            "context_results": context_results,
            "best_score": best_score,
            "min_score": min_score,
            "spread_score": spread_score,
            "embed_time": embed_time,
            "search_time": search_time,
            "retrieval_time": retrieval_time,
            "bm25_time": bm25_time,
            "fusion_time": fusion_time,
            "hybrid_time": hybrid_time,
            "score_lookup": score_lookup,
            "context_length": context_length,
            "context_tokens": context_tokens,
            "context_time": context_time,
            "prompt_length": len(prompt),
            "prompt_tokens": prompt_tokens,
            "first_token_time": first_token_time or 0.0,
            "generation_time": generation_time,
            "generated_chars": total_chars,
            "generated_tokens": total_tokens_out,
            "stopped": was_stopped,
        }
        out_q.put(("phase", "completed"))
        out_q.put(("done", payload))
    except Exception as e:
        out_q.put(("error", str(e)))


def render_ladder_html(current_idx, phase_elapsed):
    rows = []
    for i, key in enumerate(PHASES):
        label_html = PHASE_LABELS[key]
        elapsed = phase_elapsed.get(key)
        time_str = f"{elapsed:.4f}s" if elapsed is not None else ""
        if i < current_idx or (key == "completed" and i == current_idx):
            icon = '<span class="kc-step-icon kc-step-done">&#10003;</span>'
            row_cls = "kc-step-row"
        elif i == current_idx:
            icon = '<span class="kc-step-icon kc-step-active"></span>'
            row_cls = "kc-step-row"
        else:
            icon = '<span class="kc-step-icon kc-step-pending"></span>'
            row_cls = "kc-step-row kc-step-row-pending"
        rows.append(
            f'<div class="{row_cls}">'
            f"{icon}"
            f'<span class="kc-step-label">{label_html}</span>'
            f'<span class="kc-step-time">{time_str}</span>'
            f"</div>"
        )
    return f'<div class="kc-ladder">{"".join(rows)}</div>'


# ----------------------------------------------------------------------------
# Chat page rendering
# ----------------------------------------------------------------------------


def open_conversation(conv_id):
    """Swaps the working conversation. Any in-flight generation state belongs
    to the conversation being left, so it's dropped here rather than carried
    across -- the background thread is already told to stop by the caller."""
    st.session_state["conversation_id"] = conv_id
    st.session_state["messages"] = history.load_messages(conv_id) if conv_id else []
    st.session_state["pending_question"] = None
    st.session_state["is_generating"] = False
    st.session_state["active_run"] = None
    st.session_state["stop_event"] = None
    st.session_state["pipeline_out_q"] = None
    st.session_state["pipeline_state"] = None
    st.session_state["renaming_id"] = None


def commit_rename(conv_id, widget_key):
    """Saves an inline row rename and closes the editor.

    Runs as the text input's on_change, so Enter and clicking away both commit
    -- there is no separate save button to hunt for. An empty or whitespace-only
    title is ignored by history.rename_conversation(), which leaves the row
    named as it was rather than blank."""
    history.rename_conversation(conv_id, st.session_state.get(widget_key, ""))
    st.session_state["renaming_id"] = None


def focus_rename_input():
    """Puts the cursor in the inline rename field with its text selected.

    Streamlit has no autofocus, and a rename box you still have to click into
    is a rename box that looks broken. Runs in the usual 0-height iframe and
    reaches into window.parent, same as the other helpers here."""
    components.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          var tries = 0;
          function grab() {
            var el = doc.querySelector(
              '[data-testid="stSidebar"] [class*="st-key-kc_rail_editing_"] input'
            );
            if (el) { el.focus(); el.select(); return; }
            if (tries++ < 20) window.setTimeout(grab, 50);
          }
          grab();
        })();
        </script>
        """,
        height=0,
    )


def render_sidebar():
    """The conversation rail.

    Rows are disabled while an answer is streaming: switching conversations
    mid-generation would orphan the background thread and file its answer
    against whichever conversation happened to be open when it finished. Stop
    the answer first, then switch -- the disabled state says so.
    """
    # A question accepted on the previous run has not reached commit_message()
    # yet -- its conversation is opened here, before the first row is drawn,
    # so the rail shows it selected from the moment generation starts.
    pending = st.session_state.get("pending_question")
    if pending:
        ensure_conversation(pending)

    is_gen = st.session_state["is_generating"]
    active_id = st.session_state["conversation_id"]

    with st.sidebar:
        render_rail_brand()
        if st.button(
            "ការសន្ទនាថ្មី",
            key="kc_rail_new",
            icon=":material/add:",
            type="tertiary",
            width="stretch",
            disabled=is_gen or (not st.session_state["messages"] and active_id is None),
        ):
            open_conversation(None)
            st.rerun()

        st.markdown(
            '<div class="kc-rail-label">ការសន្ទនា</div>', unsafe_allow_html=True
        )

        if st.session_state["renaming_id"] is not None:
            focus_rename_input()

        conversations = history.list_conversations()
        if not conversations:
            st.markdown(
                '<div class="kc-rail-empty">មិនមានការសន្ទនា</div>',
                unsafe_allow_html=True,
            )
            return

        for conv in conversations:
            conv_id = conv["id"]
            is_active = conv_id == active_id
            # The active fill and the hover fill are painted on this wrapper
            # rather than on the title button, so they span the whole row --
            # the overflow menu sits inside the highlight instead of stranded
            # on the rail beside it. The key encodes the state because CSS is
            # the only thing that can style a Streamlit container.
            row_key = (
                f"kc_rail_row_active_{conv_id}" if is_active else f"kc_rail_row_{conv_id}"
            )
            if st.session_state["renaming_id"] == conv_id:
                # The row turns into its own editor: the title is edited where
                # the title is, not in a field tucked inside the menu that
                # opened it. Enter (or clicking away) commits via on_change.
                with st.container(key=f"kc_rail_editing_{conv_id}"):
                    widget_key = f"kc_rail_rename_{conv_id}"
                    st.text_input(
                        "ឈ្មោះការសន្ទនា",
                        value=conv["title"],
                        label_visibility="collapsed",
                        key=widget_key,
                        on_change=commit_rename,
                        args=(conv_id, widget_key),
                    )
                continue

            with st.container(key=row_key):
                col_title, col_menu = st.columns([1, 0.17])
                with col_title:
                    if st.button(
                        conv["title"],
                        key=f"kc_rail_open_{conv_id}",
                        width="stretch",
                        type="tertiary",
                        disabled=is_gen,
                        help=("បញ្ឈប់ចម្លើយមុននឹងប្តូរការសន្ទនា" if is_gen else None),
                    ):
                        if not is_active:
                            open_conversation(conv_id)
                        st.rerun()
                with col_menu:
                    with st.popover(
                        "⋯",
                        width="stretch",
                        disabled=is_gen,
                        key=f"kc_rail_menu_{conv_id}",
                    ):
                        if st.button(
                            "ផ្លាស់ប្តូរឈ្មោះ",
                            key=f"kc_rail_rename_open_{conv_id}",
                            icon=":material/edit:",
                            width="stretch",
                        ):
                            st.session_state["renaming_id"] = conv_id
                            st.rerun()
                        if st.button(
                            "លុបការសន្ទនា",
                            key=f"kc_rail_delete_{conv_id}",
                            icon=":material/delete:",
                            width="stretch",
                        ):
                            history.delete_conversation(conv_id)
                            if is_active:
                                # The open conversation just stopped existing
                                # -- land on a fresh one rather than a thread
                                # whose messages have no home to be saved to.
                                open_conversation(None)
                            st.rerun()


def render_welcome_greeting():
    """The welcome state's centre column. The logo and title do NOT live here
    -- they stay in the fixed top bar in every state, so the identity never
    moves between the welcome screen and the thread. What's left in the middle
    is an invitation to type, which is the only thing this screen asks for."""
    st.markdown(
        f'<div class="kc-welcome-wrap">'
        f'<div class="kc-greeting">{html.escape(WELCOME_GREETING)}</div>'
        f'<div class="kc-greeting-sub">{html.escape(WELCOME_SUBLINE)}</div>'
        f'<div class="kc-greeting-hint">{html.escape(WELCOME_HINT)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_welcome_form():
    """In-flow input under the title -- not pinned. Input and arrow are real
    side-by-side columns (not an absolutely-positioned overlay), so the arrow's
    position comes from normal layout flow rather than a positioning guess."""
    with st.form(key="welcome_form", clear_on_submit=True, border=True):
        col_input, col_btn = st.columns([8, 1])
        with col_input:
            question = st.text_input(
                "Ask a Khmer legal question",
                label_visibility="collapsed",
                placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ...",
            )
        with col_btn:
            submitted = st.form_submit_button("")
    if submitted and question.strip():
        st.session_state["pending_question"] = question.strip()
        st.rerun()


def render_rail_brand():
    """The product identity, at the top of the rail above the conversation
    list. It lives here rather than in a bar over the thread so the chat area
    is nothing but the conversation -- the identity sits with the navigation,
    and collapsing the rail gives the answer the full width."""
    st.markdown(
        f'<div class="kc-brand">{IRC_LOGO_HTML}'
        f"<div>"
        f'<div class="kc-brand-title">{CHATBOT_NAME}</div>'
        f'<div class="kc-brand-sub">{CHATBOT_SUBTITLE}</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_input_bar():
    """Single persistent fixed-position bar at the bottom of the chat
    screen, used for both idle (ask a question) and generating (show stop
    button) states. Both states render through the same call site (outer
    key="kc_chatbar"), so the box itself never moves or resizes -- only the
    inner content (text field vs status text) and the submit button's icon
    (CSS-swapped via :has(), see stylesheet) differ between them.

    The CSS-targetable wrapper is a plain st.container(key=...), NOT
    st.form's own key -- a borderless st.form does not reliably expose the
    same "st-key-<key>" class for CSS targeting, which silently dropped all
    of our positioning/icon-swap CSS once border=False was set. A plain
    keyed container has none of that ambiguity.

    Returns (action, payload):
      ("send", question_text)  -- user submitted a new question
      ("stop", None)           -- user clicked stop while generating
      (None, None)             -- nothing happened this run
    """
    is_gen = st.session_state["is_generating"]
    question_value = ""
    with st.container(key="kc_chatbar"):
        with st.form(key="kc_chatbar_form", clear_on_submit=True, border=False):
            col_input, col_btn = st.columns([8, 1])
            with col_input:
                if is_gen:
                    st.markdown(
                        '<div class="kc-stopbar-text">កំពុងឆ្លើយតប...</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    question_value = st.text_input(
                        "Ask a follow-up question",
                        label_visibility="collapsed",
                        placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ...",
                        key="kc_chatbar_input",
                    )
            with col_btn:
                submitted = st.form_submit_button("")

    if is_gen:
        return ("stop", None) if submitted else (None, None)
    if submitted and question_value.strip():
        return ("send", question_value.strip())
    return (None, None)


# Chunk ids are "<document id>_chunk_<n>" and document ids are
# "<type>_<nn>", where the type names the folder the PDF sits in.
_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")
SOURCE_PDF_DIRS = {
    "law": "laws",
    "prakas": "prakas",
    "sub_decree": "sub_decrees",
    "guidance": "guidance",
}
DOCUMENT_TYPE_KH = {
    "law": "ច្បាប់",
    "prakas": "ប្រកាស",
    "guidance": "គោលការណ៍ណែនាំ",
    "sub_decree": "អនុក្រឹត្យ",
    "website": "ទំព័រគេហទំព័រ",
}
DOCUMENT_TYPE_KH.update(WEBSITE_CATEGORY_LABEL_KH)


def document_id_for_chunk(chunk_id):
    """"law_04_chunk_9" -> "law_04"."""
    return _CHUNK_SUFFIX_RE.sub("", str(chunk_id))


@st.cache_resource(show_spinner=False)
def publish_source_pdfs():
    """Exposes the corpus PDFs to the browser and returns {document id: url}.

    Streamlit's static directory is not configurable -- file_util.py pins it to
    "<script dir>/static" -- so the PDFs have to appear there to be reachable
    by URL at all; data/01_source_documents can't be served directly.

    They are HARD LINKED rather than copied: same inode, second name, no extra
    disk for a 195 MB corpus. Symlinks do not work here -- Streamlit rejects a
    static path that resolves outside the static directory (HTTP 400), which a
    symlink into data/ does. Falls back to a real copy if the link can't be
    made (a different filesystem, or a platform that won't), and skips any
    document whose PDF is missing rather than failing the page: a missing
    source should cost that one link, not the answer.

    cache_resource so the directory is rebuilt once per server start, which is
    also what picks up any newly added source document."""
    static_dir = os.path.join(_APP_DIR, "static", "documents")
    os.makedirs(static_dir, exist_ok=True)
    urls = {}
    for doc_type, folder in SOURCE_PDF_DIRS.items():
        src_dir = os.path.join(_APP_DIR, "data", "01_source_documents", folder)
        if not os.path.isdir(src_dir):
            continue
        for filename in os.listdir(src_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            doc_id = os.path.splitext(filename)[0]
            src = os.path.join(src_dir, filename)
            dst = os.path.join(static_dir, filename)
            try:
                # A link that no longer points at the same inode as the source
                # is stale -- the PDF behind it was replaced -- and would keep
                # serving the old document. Relink rather than skip.
                #
                # The guard is not paranoia: this is the one place in the app
                # that deletes a file, and it sits a couple of path joins away
                # from the irreplaceable source corpus. If `dst` ever resolved
                # outside static/ -- a symlinked static/documents, a doctored
                # filename -- os.remove would take the original PDF instead of
                # the link. Verify where the path really points before unlinking.
                if os.path.exists(dst) and os.stat(dst).st_ino != os.stat(src).st_ino:
                    if os.path.realpath(dst).startswith(
                        os.path.realpath(static_dir) + os.sep
                    ):
                        os.remove(dst)
                    else:
                        continue
                if not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
                urls[doc_id] = f"app/static/documents/{filename}"
            except OSError:
                continue
    return urls


def render_used_sources(used_sources):
    """The documents an answer was drawn from, as plain links to the PDFs.

    One entry per DOCUMENT, not per chunk: which of a law's articles was
    retrieved is a retrieval detail, and it's on the diagnostics page for
    anyone who wants it. What a reader wants here is which laws this came from
    and how to go read them, so each name links to the source PDF."""
    seen = []
    for score, chunk in used_sources:
        metadata = chunk.get("metadata", {})
        name = metadata.get("document_name", "Unknown Document")
        source_url = metadata.get("source_url")
        doc_id = document_id_for_chunk(chunk["chunk_id"])
        if not any(d == doc_id for d, _, _ in seen):
            seen.append((doc_id, name, source_url))

    if not seen:
        return

    pdf_urls = publish_source_pdfs()
    items = []
    for doc_id, name, source_url in seen:
        label = html.escape(name)
        url = source_url or pdf_urls.get(doc_id)
        # A document with no PDF on disk still gets listed -- it is a real
        # source of the answer -- just without a link to follow.
        items.append(
            f'<li><a href="{html.escape(url)}" target="_blank" '
            f'rel="noopener">{label}</a></li>'
            if url
            else f"<li>{label}</li>"
        )

    st.markdown(
        f'<div class="kc-sources-title">មើលប្រភពឯកសារ</div>'
        f'<ul class="kc-sources-list">{"".join(items)}</ul>',
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Renders the mandatory disclaimer as one permanent line pinned under the
    input bar, on every screen including the welcome state.

    It used to repeat under each individual answer. Pinning it once keeps the
    same guarantee that made per-answer rendering worth it -- it is on screen
    whenever an answer is, it covers every answer path (normal, low-confidence,
    out-of-domain), and it is static markup rather than LLM-generated text a
    user's question could influence -- while dropping the repetition that made
    a long thread mostly disclaimer."""
    st.markdown(
        f'<div class="kc-disclaimer">{html.escape(DISCLAIMER_TEXT)}</div>',
        unsafe_allow_html=True,
    )


def install_action_row_script():
    """Wires up the answer action row: copy, and the two reactions.

    The controls are plain elements in the page rather than Streamlit widgets,
    and this installs ONE delegated click listener on the parent document for
    all of them. That matters for copy specifically: the clipboard API needs a
    user gesture in the document it's called from. An earlier version put the
    copy button inside its own iframe to get that gesture, which worked but
    gave it a browser-native tooltip while its three neighbours had Streamlit's
    -- four controls in a row, one of them visibly different. Handling the
    click here keeps the gesture in the page where the click happened, so all
    four can be the same kind of element with the same kind of tooltip.

    The reactions are ordinary links handled by process_feedback_query(), so
    they persist in chat_history.db. The small client-side class toggle below
    only makes the click feel immediate before Streamlit reruns."""
    components.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          if (doc.body.dataset.kcActionsWired) return;
          doc.body.dataset.kcActionsWired = '1';
          doc.addEventListener('click', function (e) {
            var copy = e.target.closest('.kc-act-copy');
            if (copy) {
              var copyText = '';
              try {
                var raw = window.parent.atob(copy.dataset.copyB64 || '');
                var bytes = Uint8Array.from(raw, function (c) { return c.charCodeAt(0); });
                copyText = new TextDecoder('utf-8').decode(bytes);
              } catch (err) {
                copyText = '';
              }
              window.parent.navigator.clipboard
                .writeText(copyText)
                .then(function () {
                  copy.classList.add('is-done');
                  window.setTimeout(function () {
                    copy.classList.remove('is-done');
                  }, 1200);
                });
              return;
            }
            var thumb = e.target.closest('.kc-act-thumb');
            if (thumb) {
              var wasOn = thumb.classList.contains('is-on');
              var row = thumb.closest('.kc-actions');
              if (row) {
                row.querySelectorAll('.kc-act-thumb').forEach(function (t) {
                  t.classList.remove('is-on');
                });
              }
              if (!wasOn) thumb.classList.add('is-on');
            }
          }, false);
        })();
        </script>
        """,
        height=0,
    )


_ICON_COPY = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="11" height="11" rx="2"/>'
    '<path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>'
)
_ICON_THUMB_UP = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 21V10l5-7a2 2 0 0 1 3 2l-1.2 5H19a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 '
    '17.6 21H7z"/><rect x="3" y="10" width="4" height="11" rx="1"/></svg>'
)
_ICON_THUMB_DOWN = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M17 3v11l-5 7a2 2 0 0 1-3-2l1.2-5H5a2 2 0 0 1-2-2.4l1.4-7A2 2 0 0 1 '
    '6.4 3H17z"/><rect x="17" y="3" width="4" height="11" rx="1"/></svg>'
)
_ICON_INFO = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.6v.4"/></svg>'
)


def render_message_actions(msg):
    """Copy, the two reactions, and the way into diagnostics.

    One markdown block, four elements of the same kind, in one flex row -- so
    they line up by construction instead of by reconciling three different
    Streamlit layout behaviours, and every one of them gets the same tooltip.

    The reactions persist to chat_history.db through query params, so demo
    reviewers can mark answers that need later dataset, retrieval, or prompt
    work without leaving the chat screen.
    """
    answer = normalize_answer_text(msg.get("answer") or "")
    answer_b64 = base64.b64encode(answer.encode("utf-8")).decode("ascii")
    msg_id = quote(str(msg["id"]), safe="")
    feedback = msg.get("feedback")
    up_cls = "kc-act kc-act-thumb" + (" is-on" if feedback == "up" else "")
    down_cls = "kc-act kc-act-thumb" + (" is-on" if feedback == "down" else "")
    diag_url = f"?view=diagnostics&amp;msg={html.escape(str(msg['id']))}"
    up_url = html.escape(f"?feedback_msg={msg_id}&feedback=up")
    down_url = html.escape(f"?feedback_msg={msg_id}&feedback=down")
    st.markdown(
        f'<div class="kc-actions">'
        f'<button type="button" class="kc-act kc-act-copy" data-copy-b64="{answer_b64}" '
        f'data-tip="ចម្លង" aria-label="ចម្លង">{_ICON_COPY}</button>'
        f'<a class="{up_cls}" href="{up_url}" data-tip="ពេញចិត្ត" '
        f'aria-label="ពេញចិត្ត" aria-pressed="{str(feedback == "up").lower()}">{_ICON_THUMB_UP}</a>'
        f'<a class="{down_cls}" href="{down_url}" data-tip="មិនពេញចិត្ត" '
        f'aria-label="មិនពេញចិត្ត" aria-pressed="{str(feedback == "down").lower()}">{_ICON_THUMB_DOWN}</a>'
        f'<a class="kc-act" href="{diag_url}" target="_blank" rel="noopener" '
        f'data-tip="មើលព័ត៌មានលម្អិត" aria-label="មើលព័ត៌មានលម្អិត">{_ICON_INFO}</a>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_message(msg):
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(msg["question"])}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)
    if msg.get("low_confidence"):
        st.markdown(
            f'<div class="kc-alert">Low confidence retrieval &mdash; best match score '
            f'{msg["best_score"]:.3f}. Treat this answer with extra caution.</div>',
            unsafe_allow_html=True,
        )
    if msg.get("interrupted"):
        st.markdown(
            '<div class="kc-alert">This question was interrupted before completion.</div>',
            unsafe_allow_html=True,
        )
        if msg.get("ladder_html"):
            st.markdown(msg["ladder_html"], unsafe_allow_html=True)
    if msg.get("errored"):
        st.markdown(
            f'<div class="kc-alert">Something went wrong while generating the answer: '
            f'{html.escape(msg.get("error_text", "unknown error"))}</div>',
            unsafe_allow_html=True,
        )
    if msg.get("out_of_domain"):
        reason = msg.get("out_of_domain_reason")
        if reason == "not_khmer":
            pass
        elif reason == "too_short":
            st.markdown(
                '<div class="kc-alert">Question too short to search &mdash; '
                "generation was skipped.</div>",
                unsafe_allow_html=True,
            )
        elif reason == "scope_violation":
            st.markdown(
                '<div class="kc-alert">Scope boundary &mdash; this assistant '
                "cannot recommend or compare specific companies/products, give "
                "personalized advice, or calculate personal monetary amounts, "
                "so generation was skipped.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="kc-alert">Out of domain &mdash; best match score '
                f'{msg["best_score"]:.3f} (below the relevance floor), so generation '
                f"was skipped.</div>",
                unsafe_allow_html=True,
            )
    if msg["answer"]:
        st.markdown(render_answer_html(msg["answer"]), unsafe_allow_html=True)
    if (
        not msg.get("interrupted")
        and not msg.get("errored")
        and not msg.get("out_of_domain")
    ):
        render_used_sources(msg["used_sources"])
        render_message_actions(msg)

    st.markdown("</div>", unsafe_allow_html=True)


def handle_new_question(
    question,
    model_name,
    client,
    embedding_model,
    embedding_model_info,
    index,
    bm25_index,
    chunks,
):
    """Starts a NEW generation, or RESUMES one already in flight.

    Resuming happens when session_state["pipeline_out_q"] is already set --
    meaning a previous script run created that queue and background thread,
    but got preempted (e.g. the user clicked Details mid-generation) before
    it could finish draining it. The queue and stop_event are real Python
    objects stashed directly in session_state (not just a snapshot of their
    contents), so picking the SAME ones back up here means this run
    continues draining the exact same background thread's output rather
    than it being permanently orphaned with nobody left to read it.
    """
    resuming = st.session_state.get("pipeline_out_q") is not None

    if resuming:
        out_q = st.session_state["pipeline_out_q"]
        stop_event = st.session_state["stop_event"]
        saved = st.session_state["pipeline_state"]
        question = saved["question"]
        overall_start = saved["overall_start"]
        phase_start_times = saved["phase_start_times"]
        phase_elapsed = saved["phase_elapsed"]
        current_idx = saved["current_idx"]
        answer_text = saved["answer_text"]
    else:
        out_q = queue.Queue()
        stop_event = threading.Event()
        st.session_state["stop_event"] = stop_event
        st.session_state["pipeline_out_q"] = out_q

        thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(
                question,
                model_name,
                client,
                embedding_model,
                embedding_model_info,
                index,
                bm25_index,
                chunks,
                out_q,
                stop_event,
            ),
            daemon=True,
        )

        overall_start = time.time()
        phase_start_times = {"encoding_query": overall_start}
        phase_elapsed = {}
        current_idx = 0
        answer_text = ""

        thread.start()

    # The question bubble needs reprinting every time this function runs --
    # including on resume, since THIS is a fresh script execution that
    # hasn't drawn anything yet, regardless of what an earlier, now-dead
    # run already sent to the browser.
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(question)}</div></div>',
        unsafe_allow_html=True,
    )
    scroll_to_bottom()
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)

    ladder_ph = st.empty()
    answer_ph = st.empty()
    if answer_text:
        answer_ph.markdown(render_answer_html(answer_text, final=False), unsafe_allow_html=True)
    ladder_ph.markdown(
        render_ladder_html(current_idx, phase_elapsed), unsafe_allow_html=True
    )

    final_payload = None
    error_text = None
    out_of_domain_payload = None

    def save_progress():
        # Stores BOTH the raw numeric state needed to seamlessly resume
        # this exact polling loop from a future run (pipeline_state), AND
        # a pre-rendered snapshot for the interrupted-message/stop path
        # (active_run, unchanged from before).
        st.session_state["pipeline_state"] = {
            "question": question,
            "overall_start": overall_start,
            "phase_start_times": phase_start_times,
            "phase_elapsed": phase_elapsed,
            "current_idx": current_idx,
            "answer_text": answer_text,
        }
        st.session_state["active_run"] = {
            "question": question,
            "answer_text": answer_text,
            "ladder_html": render_ladder_html(current_idx, phase_elapsed),
        }

    save_progress()

    finished = False
    while not finished:
        # If the user clicked stop on a *previous* run, this whole function
        # call (and its thread) never even started -- Streamlit kills the
        # in-flight script run the instant a new widget interaction comes
        # in. That's handled in render_chat_page before we get here. This
        # loop only watches the queue for the thread tied to it (freshly
        # started, or resumed from a previous run -- same loop either way).
        try:
            kind, payload = out_q.get(timeout=0.12)
            now = time.time()
            if kind == "phase":
                phase_elapsed[PHASES[current_idx]] = (
                    now - phase_start_times[PHASES[current_idx]]
                )
                current_idx = PHASES.index(payload)
                phase_start_times[payload] = now
                # Re-anchor to the bottom as the ladder grows -- a single
                # scroll right when the question bubble appears (before any
                # ladder rows exist) isn't enough, since the page keeps
                # growing taller below that point as each phase completes.
                scroll_to_bottom()
            elif kind == "token":
                answer_text += payload
                answer_ph.markdown(render_answer_html(answer_text, final=False), unsafe_allow_html=True)
                if has_repetition_loop(answer_text):
                    stop_event.set()
            elif kind == "done":
                phase_elapsed[PHASES[current_idx]] = (
                    now - phase_start_times[PHASES[current_idx]]
                )
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                final_payload = payload
                finished = True
                scroll_to_bottom()
            elif kind == "error":
                error_text = payload
                finished = True
            elif kind == "out_of_domain":
                phase_elapsed[PHASES[current_idx]] = (
                    now - phase_start_times[PHASES[current_idx]]
                )
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                out_of_domain_payload = payload
                finished = True
                scroll_to_bottom()
        except queue.Empty:
            pass

        if not finished:
            phase_elapsed[PHASES[current_idx]] = (
                time.time() - phase_start_times[PHASES[current_idx]]
            )
        ladder_html = render_ladder_html(current_idx, phase_elapsed)
        ladder_ph.markdown(ladder_html, unsafe_allow_html=True)

        # Mirror progress so a stop-induced rerun, OR a later run resuming
        # this same queue, can recover it. Cheap dict writes; happens at
        # most ~every 0.12s (queue poll interval) or immediately on each
        # phase/token event.
        save_progress()

    # Generation has concluded one way or another -- this queue/state is no
    # longer adoptable by a future run.
    st.session_state["pipeline_out_q"] = None
    st.session_state["pipeline_state"] = None

    time.sleep(0.4)
    ladder_ph.empty()

    if out_of_domain_payload is not None:
        st.session_state["active_run"] = None
        score = out_of_domain_payload["best_score"]
        reason = out_of_domain_payload.get("reason", "low_score")

        # Distinct wording per reason -- the previous single generic string
        # ("your question doesn't relate to insurance law") was misleading
        # for scope_violation in particular: that case IS about insurance,
        # the assistant just can't recommend/calculate/personally-advise.
        if reason == "not_khmer":
            canned_answer = NOT_KHMER_QUESTION_TEXT
        elif reason == "scope_violation":
            canned_answer = (
                "សំណួររបស់អ្នកស្នើសុំការណែនាំ ឬប្រៀបធៀបក្រុមហ៊ុន/ផលិតផលជាក់លាក់ "
                "ការផ្តល់ប្រឹក្សាផ្ទាល់ខ្លួន ឬការគណនាប្រាក់ផ្ទាល់ខ្លួន ដែលជំនួយការនេះ"
                "មិនអាចផ្តល់បានទេ។ "
                f"{REDIRECT_TEXT}"
            )
        else:
            canned_answer = (
                "សំណួររបស់អ្នកហាក់ដូចជាមិនទាក់ទងនឹងច្បាប់ ឬបទប្បញ្ញត្តិធានារ៉ាប់រងកម្ពុជាទេ។ "
                "ជំនួយការនេះអាចឆ្លើយបានតែសំណួរទាក់ទងនឹងច្បាប់ធានារ៉ាប់រងប៉ុណ្ណោះ។ "
                f"{REDIRECT_TEXT}"
            )
        commit_message(
            {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": canned_answer,
                "used_sources": [],
                "all_results": out_of_domain_payload["results"],
                "best_score": score,
                "low_confidence": False,
                "interrupted": False,
                "out_of_domain": True,
                "out_of_domain_reason": reason,
            }
        )
        return

    if error_text:
        # Previously this only flashed a transient alert via answer_ph and
        # returned -- but render_chat_page unconditionally st.rerun()s right
        # after handle_new_question, which redraws the whole page from
        # session_state["messages"]. Since nothing was appended there, the
        # alert vanished and the question disappeared with no trace. Persist
        # it like an interrupted message instead.
        st.session_state["active_run"] = None
        commit_message(
            {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": "",
                "used_sources": [],
                "all_results": [],
                "best_score": 0.0,
                "low_confidence": False,
                "interrupted": False,
                "errored": True,
                "error_text": error_text,
            }
        )
        return

    # Defensive: anything unexpected here (e.g. a malformed payload from a
    # future change) must never escape as an uncaught exception. If it did,
    # render_chat_page's "is_generating = False" reset right after this call
    # would never be reached, permanently locking the input bar into its
    # generating/stop state with no way back short of restarting the app.
    try:
        results = final_payload["results"]
        fused_results = final_payload["fused_results"]
        best_score = final_payload["best_score"]
        used_sources = select_used_sources(final_payload["context_results"])

        render_used_sources(used_sources)

        answer_text = normalize_answer_text(answer_text)

        msg = {
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": answer_text,
            "used_sources": used_sources,
            # Diagnostics shows the FUSED ranking, not the raw FAISS-only
            # list -- fused_results is what select_context_chunks() drew
            # from inside run_pipeline_in_thread to build context_results
            # (already computed, unaffected by the slice below). The raw
            # FAISS list is still available via diag.best_score/min_score/
            # spread_score (the numbers the safety gates actually used),
            # just not as the chunk cards.
            #
            # Sliced to DIAG_DISPLAY_K here for DISPLAY ONLY -- the fused
            # list can hold more than TOP_K unique chunks (FAISS's TOP_K
            # and BM25's BM25_TOP_K overlap only partially), and this stays
            # purely a presentation choice, not a change to retrieval,
            # fusion, the candidate pool, or context selection.
            "all_results": fused_results[:DIAG_DISPLAY_K],
            "faiss_results": results,
            "score_lookup": final_payload["score_lookup"],
            "best_score": best_score,
            "low_confidence": best_score < MIN_SCORE,
            "interrupted": False,
            "diag": {
                "model": model_name,
                "embedding_model": embedding_model_info["label"],
                "embedding_device": str(embedding_model.device),
                "top_k": TOP_K,
                "embed_time": final_payload["embed_time"],
                "search_time": final_payload["search_time"],
                "retrieval_time": final_payload["retrieval_time"],
                "bm25_time": final_payload["bm25_time"],
                "fusion_time": final_payload["fusion_time"],
                "hybrid_time": final_payload["hybrid_time"],
                "min_score": final_payload["min_score"],
                "spread_score": final_payload["spread_score"],
                "context_length": final_payload["context_length"],
                "context_tokens": final_payload["context_tokens"],
                "context_time": final_payload["context_time"],
                "prompt_length": final_payload["prompt_length"],
                "prompt_tokens": final_payload["prompt_tokens"],
                "first_token_time": final_payload["first_token_time"],
                "generation_time": final_payload["generation_time"],
                "total_time": time.time() - overall_start,
                "generated_chars": final_payload["generated_chars"],
                "generated_tokens": final_payload["generated_tokens"],
            },
        }
        commit_message(msg)
    except Exception as e:
        commit_message(
            {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": answer_text,
                "used_sources": [],
                "all_results": [],
                "best_score": 0.0,
                "low_confidence": False,
                "interrupted": False,
                "errored": True,
                "error_text": f"internal error while finalizing the answer: {e}",
            }
        )
    finally:
        st.session_state["active_run"] = None


def render_chat_page(
    model_name, client, embedding_model, embedding_model_info, index, bm25_index, chunks
):
    messages = st.session_state["messages"]
    pending = st.session_state["pending_question"]

    # Fixed-position and unconditional -- rendered first, in every state, so it
    # doesn't depend on which branch of the page runs below.
    render_disclaimer()

    if not messages and not pending and not st.session_state.get("is_generating"):
        render_welcome_greeting()
        render_welcome_form()
        return

    # Single .kc-thread wrapper spans BOTH the existing history and the
    # currently-processing question below it -- previously this was opened
    # and closed twice (once per section), stacking the wrapper's top/bottom
    # padding on top of itself and leaving a large blank gap between them.
    st.markdown('<div class="kc-thread">', unsafe_allow_html=True)
    for msg in messages:
        render_message(msg)

    # A question is about to start processing this run -- lock the input
    # bar down to its generating/stop state before the (blocking) pipeline
    # call below.
    if pending:
        st.session_state["is_generating"] = True

    # Rendered before the (blocking) pipeline run below so it's already
    # painted on screen the instant a question is sent. Its position on
    # screen is governed purely by CSS (position: fixed), not by where it
    # sits in the markup, so calling it here -- ahead of the new question's
    # own content -- doesn't affect where it visually appears.
    action, payload = render_input_bar()

    if action == "stop":
        # This branch only ever runs on the NEXT script execution after the
        # click: clicking Stop is itself a new widget interaction, so
        # Streamlit immediately tears down whatever run was mid-stream
        # (including the thread-polling loop inside handle_new_question) and
        # starts this fresh run. That old run's local variables are gone --
        # but it was mirroring its progress into session_state["active_run"]
        # on every loop tick, so we can recover the question, the partial
        # answer, and the ladder right up to the moment of the click.
        ev = st.session_state.get("stop_event")
        if ev is not None:
            ev.set()  # tells the orphaned background thread to stop streaming

        active = st.session_state.get("active_run")
        if active is not None:
            commit_message(
                {
                    "id": str(uuid.uuid4()),
                    "question": active["question"],
                    "answer": active["answer_text"],
                    "used_sources": [],
                    "all_results": [],
                    "best_score": 0.0,
                    "low_confidence": False,
                    "interrupted": True,
                    "ladder_html": active["ladder_html"],
                }
            )

        st.session_state["is_generating"] = False
        st.session_state["stop_event"] = None
        st.session_state["active_run"] = None
        st.session_state["pipeline_out_q"] = None
        st.session_state["pipeline_state"] = None
        st.session_state["pending_question"] = None
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()

    # A fresh question was just submitted ("pending"), OR a generation
    # started by some earlier, now-preempted run is still in flight and
    # adoptable (is_generating is True, nothing newly pending, and its
    # queue/stop_event are still sitting in session_state ready to be
    # resumed) -- either way, handle_new_question drives it from here.
    resuming = (
        not pending
        and st.session_state.get("is_generating")
        and st.session_state.get("pipeline_out_q") is not None
    )

    if pending or resuming:
        st.session_state["pending_question"] = None
        handle_new_question(
            pending,
            model_name,
            client,
            embedding_model,
            embedding_model_info,
            index,
            bm25_index,
            chunks,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["is_generating"] = False
        st.session_state["stop_event"] = None
        st.rerun()

    if action == "send":
        st.session_state["pending_question"] = payload
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()

    # Defensive: is_generating shouldn't be True here with nothing to
    # resume (pending and resuming both false) -- but if some future
    # change ever produces that mismatch, fail open rather than leaving
    # the input bar stuck showing the stop button forever.
    if st.session_state.get("is_generating"):
        st.session_state["is_generating"] = False

    st.markdown("</div>", unsafe_allow_html=True)

    # Idle tail: only reached once a question has fully finished (or been
    # stopped) and the page has settled -- safe to snap to the bottom so
    # the newest content (ladder/answer/interrupted notice) is visible
    # without the user needing to scroll.
    scroll_to_bottom()


# ----------------------------------------------------------------------------
# Diagnostics page rendering
# ----------------------------------------------------------------------------


def console_block(title, rows):
    """Renders one pipeline stage block. The title includes a real
    kc-pipeline-dot element (not a CSS-only pseudo-element) so its
    position on the rail is set by a deterministic negative margin (see
    .kc-pipeline-dot CSS) rather than a guessed absolute offset. This
    couples console_block to the pipeline rail in render_diagnostics_page
    -- fine today since this is its only call site, but worth revisiting
    if console_block is ever reused outside that railed container."""
    row_html = "".join(
        f'<div class="kc-row"><span class="kc-row-label">{html.escape(str(k))}</span>'
        f'<span class="kc-row-value">{html.escape(str(v))}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="kc-console-title kc-pipeline-title">'
        f'<span class="kc-pipeline-dot"></span>&gt; {html.escape(title)}'
        f"</div>"
        f'<div class="kc-console-block">{row_html}</div>',
        unsafe_allow_html=True,
    )


def render_provenance_badge(has_faiss, has_bm25):
    """A small circular badge showing which retrieval engine(s) surfaced
    this chunk. The mapping is a nod to physical seals (ត្រា) stamped on
    Cambodian legal documents: a chunk gets "stamped" by FAISS, by BM25,
    or by both. When both rankers agree, the ring and center dot split
    between the IRC gold and green."""
    GOLD, GREEN = "#B19F50", "#02913F"
    OUTER_RING = "#3A4A41"  # quiet neutral ring, independent of provenance color

    if has_faiss and has_bm25:
        inner = (
            f'<path d="M 3 10 A 7 7 0 0 1 17 10" fill="none" stroke="{GOLD}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
            f'<path d="M 3 10 A 7 7 0 0 0 17 10" fill="none" stroke="{GREEN}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
        )
        # Center dot split the same way as the ring so "both" reads as a
        # literal half-and-half blend of the two engines.
        dot = (
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 1 12.8 10 Z" fill="{GOLD}"/>'
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 0 12.8 10 Z" fill="{GREEN}"/>'
        )
    elif has_faiss:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{GOLD}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{GOLD}"/>'
    elif has_bm25:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{GREEN}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{GREEN}"/>'
    else:
        inner, dot = "", ""

    return (
        '<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="10" cy="10" r="9" fill="none" stroke="{OUTER_RING}" stroke-width="1"/>'
        f"{inner}{dot}"
        "</svg>"
    )


def render_diag_source_cards(results, score_lookup):
    # Score bar is normalized relative to the TOP result in this specific
    # list, not assumed to live on a fixed 0-1 range. This list now holds
    # RRF fused scores (see reciprocal_rank_fusion), which are small
    # numbers on a completely different scale than the old raw FAISS
    # cosine scores this card was originally built for -- relative
    # normalization works correctly for either scale without caring which
    # one it's given.
    max_score = max((float(score) for score, _ in results), default=0.0) or 1.0
    cards = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = html.escape(metadata.get("document_name", "Unknown Document"))
        chunk_id = html.escape(chunk["chunk_id"])
        pct = max(0.0, min(1.0, float(score) / max_score)) * 100
        delay = (rank - 1) * 0.05

        # A chunk may have been surfaced by FAISS, by BM25, or by both --
        # an em-dash means that particular ranker didn't return this chunk
        # in its own top-K at all, which is itself useful diagnostic
        # information (e.g. a chunk with FAISS "—" and a real BM25 score
        # means lexical search alone found it; dense search missed it).
        raw_scores = score_lookup.get(chunk["chunk_id"], {})
        faiss_val = raw_scores.get("faiss")
        bm25_val = raw_scores.get("bm25")
        faiss_str = f"{faiss_val:.4f}" if faiss_val is not None else "—"
        bm25_str = f"{bm25_val:.4f}" if bm25_val is not None else "—"
        badge_svg = render_provenance_badge(faiss_val is not None, bm25_val is not None)

        score_rows = "".join(
            f'<div class="kc-diag-src-score-row{extra_class}">'
            f'<span class="kc-diag-src-score-key">{label}</span>'
            f'<span class="kc-diag-src-score-val">{value}</span>'
            f"</div>"
            for label, value, extra_class in [
                ("FAISS", faiss_str, ""),
                ("BM25", bm25_str, ""),
                ("RRF", f"{score:.4f}", " kc-diag-src-score-row--rrf"),
            ]
        )

        card = (
            f'<div class="kc-diag-src-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="kc-diag-src-top">'
            f'<span class="kc-diag-src-rank">{rank:02d}</span>'
            f'<div class="kc-diag-src-main">'
            f'<div class="kc-diag-src-name-row">{badge_svg}'
            f'<div class="kc-diag-src-name">{doc_name}</div></div>'
            f'<div class="kc-diag-src-id">{chunk_id}</div>'
            f"</div>"
            f'<div class="kc-diag-src-scores">{score_rows}</div>'
            f"</div>"
            f'<div class="kc-diag-src-bar"><div class="kc-diag-src-fill" '
            f'style="--w:{pct:.1f}%; animation-delay:{delay:.2f}s"></div></div>'
            f"</div>"
        )
        cards.append(card)
    st.markdown(
        f'<div class="kc-diag-src-list">{"".join(cards)}</div>', unsafe_allow_html=True
    )


def render_source_detail(rank, score, chunk):
    metadata = chunk.get("metadata", {})
    doc_name = metadata.get("document_name", "N/A")
    rows = [
        ("fused score (rrf)", f"{float(score):.4f}"),
        ("document", doc_name),
        (
            "type",
            metadata.get(
                "document_type_kh",
                DOCUMENT_TYPE_KH.get(metadata.get("document_type"), metadata.get("document_type", "N/A")),
            ),
        ),
        ("release date", metadata.get("release_date", "N/A")),
        ("source url", metadata.get("source_url", "N/A")),
        ("chapter", metadata.get("chapter_title", metadata.get("chapter", "N/A"))),
        ("section", metadata.get("section_title", metadata.get("section", "N/A"))),
        ("legal unit", metadata.get("legal_unit", "N/A")),
        ("chunk id", chunk["chunk_id"]),
    ]
    grid = "".join(
        f'<div class="k">{html.escape(str(k))}</div><div class="v">{html.escape(str(v))}</div>'
        for k, v in rows
    )
    text = html.escape(chunk["text"])
    with st.expander(f"Rank {rank}  —  {doc_name}"):
        st.markdown(
            f'<div class="kc-detail-grid">{grid}</div><div class="kc-detail-text">{text}</div>',
            unsafe_allow_html=True,
        )


def render_diagnostics_page():
    # The message is looked for in the open conversation first, then in the
    # database -- so this page renders whether it was reached by clicking the
    # info icon or by opening its URL in a fresh session.
    msg = next(
        (m for m in st.session_state["messages"] if m["id"] == DIAG_MSG_ID),
        None,
    ) or history.load_message(DIAG_MSG_ID)

    if st.button("← Back to chat", key="back_to_chat"):
        open_chat()
        st.rerun()

    if msg is None:
        st.markdown(
            '<div class="kc-diag-title">No diagnostics found for this message.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="kc-diag-header">'
        f'<div class="kc-diag-title">DIAGNOSTICS — <span class="kc-diag-question">{html.escape(msg["question"])}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    d = msg["diag"]

    def tok(value):
        return f"{value:,}" if value is not None else "N/A"

    col1, col2 = st.columns([1.25, 2.15])
    with col1:
        with st.container(key="kc_diag_pipeline"):
            console_block(
                "SYSTEM",
                [
                    ("generator model", d["model"]),
                    ("generator endpoint", LLAMA_SERVER_URL),
                    ("retriever model", d["embedding_model"]),
                    ("retriever device", d["embedding_device"]),
                    ("retrieval mode", "hybrid (FAISS + BM25, RRF)"),
                    ("top k", d["top_k"]),
                ],
            )
            console_block(
                "RETRIEVAL",
                [
                    ("embed time", f'{d["embed_time"]:.4f}s'),
                    ("faiss time", f'{d["search_time"]:.4f}s'),
                    ("bm25 time", f'{d["bm25_time"]:.4f}s'),
                    ("fusion time", f'{d["fusion_time"]:.4f}s'),
                    (
                        "total retrieval time",
                        f'{d["retrieval_time"] + d["hybrid_time"]:.4f}s',
                    ),
                    ("best score (faiss, gate)", f'{msg["best_score"]:.4f}'),
                    ("min score (faiss)", f'{d["min_score"]:.4f}'),
                    ("spread score (faiss)", f'{d["spread_score"]:.4f}'),
                ],
            )
            console_block(
                "CONTEXT",
                [
                    ("context length", f'{d["context_length"]} chars'),
                    ("context token", tok(d["context_tokens"])),
                    ("context build time", f'{d["context_time"]:.4f}s'),
                    ("prompt length", f'{d["prompt_length"]} chars'),
                    ("prompt token", tok(d["prompt_tokens"])),
                ],
            )
            console_block(
                "GENERATION",
                [
                    ("first token", f'{d["first_token_time"]:.4f}s'),
                    ("generation time", f'{d["generation_time"]:.4f}s'),
                    (
                        "tokens/sec (decode)",
                        f'{d["generated_tokens"]/max(d["generation_time"]-d["first_token_time"],0.001):.4f}',
                    ),
                    ("total time", f'{d["total_time"]:.4f}s'),
                    ("output token", tok(d["generated_tokens"])),
                    ("generated chars", d["generated_chars"]),
                ],
            )
    with col2:
        st.markdown(
            f'<div class="kc-console-title">&gt; CHUNKS (TOP {len(msg["all_results"])}, FUSED RANKING)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kc-diag-legend">'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(True, False)}<span>FAISS only</span></div>'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(False, True)}<span>BM25 only</span></div>'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(True, True)}<span>both (confirmed)</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        render_diag_source_cards(msg["all_results"], msg.get("score_lookup", {}))

    st.markdown(
        '<div class="kc-console-title" style="margin-top:18px">&gt; FULL SOURCE DETAILS</div>',
        unsafe_allow_html=True,
    )
    for rank, (score, chunk) in enumerate(msg["all_results"], start=1):
        render_source_detail(rank, score, chunk)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

inject_css()

MODEL_NAME = get_loaded_model()

if MODEL_NAME == "No Model Loaded":
    deployment_hint = (
        f" Deployment commit: {html.escape(DEPLOYMENT_COMMIT[:12])}."
        if DEPLOYMENT_COMMIT
        else ""
    )
    st.markdown(
        '<div class="kc-alert">'
        f"No generator reachable at {html.escape(LLAMA_SERVER_URL)}/models. "
        "For local use, start llama-server. For hosted use, set "
        "GENERATOR_BASE_URL, GENERATOR_MODEL, and GENERATOR_API_KEY."
        f"{deployment_hint}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not USES_LOCAL_LLAMA_SERVER and not GENERATOR_API_KEY:
    st.markdown(
        '<div class="kc-alert">'
        "Hosted generator is configured, but no API key was found. Set "
        "GENERATOR_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEYS."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key=GENERATOR_API_KEY)
embedding_model_info = load_embedding_model_info()
embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()
bm25_index = load_bm25_index(chunks)

if IS_DIAGNOSTICS:
    render_diagnostics_page()
else:
    # Rendered before the page body: the rail can switch or delete the working
    # conversation, and doing that after the thread has already been painted
    # would show one conversation's messages under another's title for a frame.
    render_sidebar()
    # After the rail exists in the DOM, so the first measurement is the real
    # width rather than the 0px fallback.
    sync_rail_width()
    disable_rail_resize()
    install_action_row_script()
    render_chat_page(
        MODEL_NAME, client, embedding_model, embedding_model_info, index, bm25_index, chunks
    )
