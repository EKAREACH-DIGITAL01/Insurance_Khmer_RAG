import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
MODEL = os.getenv("CHATBOT_MODEL", "Generator")

# Set HF_HOME before any HF import so all HF assets stay inside the
# project folder. See scripts/13_install_models.py for the full
# explanation of why this must come before the imports below.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE_DIR = os.path.join(_APP_DIR, "models", "retriever", ".cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR

import faiss

faiss.omp_set_num_threads(1)

import html
import json
import queue
import re
import threading
import time
import uuid

import numpy as np
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

LLAMA_SERVER_URL = "http://localhost:8080/v1"
LLAMA_SERVER_ROOT = "http://localhost:8080"

TOP_K = 10
CONTEXT_K = 5  # how many chunks actually go into the LLM prompt (<= TOP_K)
MAX_CHUNKS_PER_DOC = 1  # cap near-duplicate chunks from the same document
BM25_TOP_K = TOP_K  # lexical candidate pool size, matched to FAISS's TOP_K
                     # so neither ranker dominates the fused pool just by
                     # having a bigger candidate list to draw from
RRF_K = 60  # standard Reciprocal Rank Fusion constant from the IR
            # literature (Cormack et al. 2009) -- not something tuned on
            # this corpus, just the conventional default
NGRAM_SIZE = 3  # character n-gram size for Khmer-script BM25 tokens
DIAG_DISPLAY_K = 10  # how many fused chunks the diagnostics page SHOWS --
                      # presentation-only. The fused candidate pool itself
                      # (up to BM25_TOP_K + TOP_K unique chunks if FAISS and
                      # BM25 barely overlap) and select_context_chunks()'s
                      # CONTEXT_K selection are both untouched by this; it
                      # only trims what's rendered in the "CHUNKS" cards and
                      # "FULL SOURCE DETAILS" section below them.
MIN_SCORE = 0.50
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
    Khmer word segmenter. If 08_evaluate_retrieval.py's hybrid numbers
    don't show a real improvement over dense-only, this tokenizer -- not
    necessarily the hybrid approach itself -- is the first thing to
    re-examine."""
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
    # "how much is the premium/price/cost" -- pricing and premium
    # questions generally, not just personal-payment framing. Per the
    # Internship Proposal's explicit scope boundary (no premium
    # calculation), this blocks "how much is the premium" even when the
    # corpus contains a real statutory rate that would otherwise be
    # retrieved -- reciting a rate as if it answers "how much" risks the
    # exact rate-vs-amount conflation the build_prompt() rules separately
    # warn against, so it's safer to redirect than to answer here.
    re.compile(r"(បុព្វលាភ|ថ្លៃ|តម្លៃ).{0,30}ប៉ុន្មាន"),
]


def is_scope_violation(text):
    """Deterministic, score-independent check for questions asking the
    assistant to do something explicitly out of scope per the Internship
    Proposal's §4/§11 boundaries: recommend or compare a specific company
    or product, give advice framed as a personal choice, or calculate a
    personal monetary amount. See the comment block above for why this is
    a separate gate rather than relying on prompt instructions alone."""
    return any(pattern.search(text) for pattern in SCOPE_VIOLATION_PATTERNS)


MAX_USED_SOURCES = 3

# Which precomputed vector_db/<subfolder>/ the live app loads. Change this
# one line to switch embedding models -- scripts/07_generate_embeddings.py
# builds all installed models into their own subfolders (vector_db/BAAI-BGE-M3/,
# vector_db/jinaai-jina-embeddings-v3/, vector_db/MiniLM-paraphrase-multilingual-L12-v2/),
# so switching here doesn't require rebuilding anything that's already
# been generated -- just restart the app.
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

SEAL_SVG = """<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><circle cx="20" cy="20" r="18" fill="none" stroke="#B68B4C" stroke-width="1.1"/><circle cx="20" cy="20" r="12" fill="none" stroke="#B68B4C" stroke-width="1.1"/><g stroke="#B68B4C" stroke-width="1.1"><line x1="20" y1="2" x2="20" y2="8"/><line x1="20" y1="32" x2="20" y2="38"/><line x1="2" y1="20" x2="8" y2="20"/><line x1="32" y1="20" x2="38" y2="20"/><line x1="7.5" y1="7.5" x2="11.5" y2="11.5"/><line x1="28.5" y1="28.5" x2="32.5" y2="32.5"/><line x1="7.5" y1="32.5" x2="11.5" y2="28.5"/><line x1="28.5" y1="11.5" x2="32.5" y2="7.5"/></g><circle cx="20" cy="20" r="3" fill="#B68B4C"/></svg>"""

favicon = Image.open("assets/favicon.png")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("view", "chat")
st.session_state.setdefault("active_msg_id", None)
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

IS_DIAGNOSTICS = st.session_state["view"] == "diagnostics"

st.set_page_config(
    page_title="Khmer Insurance Law Assistant",
    page_icon=favicon,
    layout="wide" if IS_DIAGNOSTICS else "centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

PAGE_BG = "var(--ink)" if IS_DIAGNOSTICS else "var(--parchment)"
PAGE_FG = "var(--mist)" if IS_DIAGNOSTICS else "var(--stone)"
BUTTON_HOVER_COLOR = "var(--signal)" if IS_DIAGNOSTICS else "var(--lacquer)"
# Streamlit's default block-container top padding is generous, sized for
# chat's own fixed topbar/welcome layout -- the diagnostics page has
# neither, so that default padding just reads as dead space above the
# Back button. Scoped to IS_DIAGNOSTICS only, so chat's layout is
# completely untouched.
DIAG_TOP_PADDING_CSS = (
    '[data-testid="stAppViewContainer"] .block-container { padding-top: 1.2rem !important; }'
    if IS_DIAGNOSTICS
    else ""
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: #15201C;
  --ink-soft: #1D2B25;
  --ink-line: #2C3D34;
  --parchment: #EDE6D6;
  --parchment-soft: #F5F0E4;
  --lacquer: #9C2B1F;
  --lacquer-deep: #7A2118;
  --lacquer-mist: rgba(156,43,31,0.08);
  --brass: #B68B4C;
  --brass-soft: #D8BD8B;
  --stone: #3A352B;
  --mist: #9DB6AB;
  --signal: #6FBF8B;
  --danger: #C0524A;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stToolbar"] {{display: none !important;}}
[data-testid="stDecoration"] {{display: none !important;}}
[data-testid="stStatusWidget"] {{display: none !important;}}
[data-testid="stForm"] div[data-testid="InputInstructions"] {{display: none !important;}}
header {{display: none !important;}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {PAGE_BG} !important;
  color: {PAGE_FG};
  font-family: 'IBM Plex Sans', sans-serif;
}}

{DIAG_TOP_PADDING_CSS}

.kc-welcome-wrap {{ margin-top: 16vh; text-align: center; }}
.kc-welcome-lockup {{
  display: inline-flex;
  align-items: center;
  gap: 20px;
  text-align: left;
}}
.kc-welcome-lockup svg {{
  display: block;
  flex-shrink: 0;
  transform: translateY(4px);
}}
.kc-eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--brass);
  text-align: center;
}}
.kc-title-big {{
  font-family: 'Moul', serif !important;
  font-size: 30px !important;
  color: var(--lacquer) !important;
  margin: 17px 0 0 0 !important;
  line-height: 1.3 !important;
  white-space: nowrap;
}}
.kc-subtitle {{
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 13px;
  color: var(--stone);
  opacity: 0.7;
  margin-top: 16px;
}}
.kc-model-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--ink);
  color: var(--parchment-soft);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  padding: 6px 14px;
  border-radius: 999px;
  margin-top: 10px;
}}
.kc-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.kc-dot-on {{ background: var(--signal); box-shadow: 0 0 0 3px rgba(111,191,139,0.25); }}
.kc-dot-off {{ background: var(--danger); box-shadow: 0 0 0 3px rgba(192,82,74,0.25); }}

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

/* ---------- Compact fixed top bar (chat state) ---------- */
.kc-topbar {{
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 998;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 20px;
  background: var(--parchment);
  border-bottom: 1px solid var(--brass-soft);
}}
.kc-topbar-title {{ font-family: 'Moul', serif; font-size: 15px; color: var(--lacquer); }}
.kc-topbar-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--brass); opacity: 0.85; }}

.kc-thread {{ padding-top: 64px; padding-bottom: 96px; }}

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
  line-height: 1.9;
  color: var(--stone);
  white-space: pre-wrap;
  min-height: 1.4em;
}}

.kc-sources-block {{ margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }}
.kc-source-chip {{ border-left: 2px solid var(--brass); padding: 4px 0 4px 12px; }}
.kc-source-chip .name {{ font-family: 'Hanuman', sans-serif; font-size: 13.5px; color: var(--stone); }}
.kc-source-chip .id {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--brass); margin-top: 2px; }}

.kc-msg-actions {{ margin-top: 10px; }}

.kc-disclaimer {{
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 11.5px;
  color: var(--stone);
  opacity: 0.65;
  border-top: 1px dashed var(--brass-soft);
  padding-top: 8px;
  margin-top: 14px;
  line-height: 1.6;
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
  left: 50% !important;
  bottom: 24px !important;
  transform: translateX(-50%) !important;
  contain: none !important;
  filter: none !important;
  will-change: auto !important;
  z-index: 999;
  max-width: 560px;
  width: calc(100% - 48px);
  margin: 0 !important;
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
  box-shadow: 0 0 0 4px var(--ink), 0 0 8px rgba(182, 139, 76, 0.45);
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

.kc-chunk-row {{
  display: grid; grid-template-columns: 28px 60px 1fr 60px; gap: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
  padding: 6px 0; border-bottom: 1px dashed rgba(157,182,171,0.12); align-items: center;
}}
.kc-chunk-row:last-child {{ border-bottom: none; }}
.kc-chunk-rank {{ color: var(--brass); }}
.kc-chunk-score {{ color: var(--signal); }}
.kc-chunk-id {{ color: var(--parchment-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.kc-chunk-len {{ color: var(--mist); opacity: 0.7; text-align: right; }}

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
  background-color: rgba(182,139,76,0.1) !important;
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
    try:
        resp = requests.post(
            f"{LLAMA_SERVER_ROOT}/tokenize", json={"content": text}, timeout=10
        )
        resp.raise_for_status()
        tokens = resp.json().get("tokens", [])
        return len(tokens)
    except Exception:
        return None


def get_loaded_model():
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["id"]
    except Exception:
        return "No Model Loaded"


@st.cache_resource(show_spinner=False)
def resolve_vector_db():
    """Figures out which vector_db/ layout is actually on disk and
    resolves the FAISS index path, metadata path, and embedding model
    info accordingly. Tries, in order:

      1. New per-model subfolder layout -- vector_db/<ACTIVE_EMBEDDING_MODEL_DIR>/
         -- what the current scripts/07_generate_embeddings.py builds.
      2. Old flat layout with a top-level embedding_model_info.json --
         what an earlier version of scripts/07 built (one model,
         dynamically chosen, no subfolders).
      3. Original flat layout with no info file at all -- assumes
         BAAI/bge-m3, since that was the only option before multiple
         embedding models existed.

    Returns (faiss_path, metadata_path, embedding_model_info, found,
    layout_note). `found` is True only for case 1 (the current, fully
    correct layout) -- cases 2 and 3 still work, but return a
    human-readable note explaining the situation so the UI can show it
    rather than silently guessing forever."""
    new_dir = f"{VECTOR_DB_ROOT}/{ACTIVE_EMBEDDING_MODEL_DIR}"
    new_faiss = f"{new_dir}/khmer_insurance.index"
    new_metadata = f"{new_dir}/khmer_insurance_metadata.json"
    new_info = f"{new_dir}/embedding_model_info.json"

    if os.path.exists(new_faiss) and os.path.exists(new_metadata):
        if os.path.exists(new_info):
            with open(new_info, "r", encoding="utf-8") as f:
                info = json.load(f)
            return new_faiss, new_metadata, info, True, None

        # Subfolder exists but somehow no info file -- shouldn't happen
        # from a normal scripts/07 run, but don't crash over it.
        fallback_info = {
            "label": ACTIVE_EMBEDDING_MODEL_DIR,
            "hf_id": None,
            "trust_remote_code": False,
            "task_aware": False,
        }
        note = (
            f"Found {new_dir}/ but no embedding_model_info.json inside it -- "
            "assuming a non-task-aware model. Re-run "
            "scripts/07_generate_embeddings.py to fix."
        )
        return new_faiss, new_metadata, fallback_info, False, note

    # Fall back to the old flat layout (pre-dates per-model subfolders).
    old_faiss = f"{VECTOR_DB_ROOT}/khmer_insurance.index"
    old_metadata = f"{VECTOR_DB_ROOT}/khmer_insurance_metadata.json"
    old_info = f"{VECTOR_DB_ROOT}/embedding_model_info.json"

    if os.path.exists(old_faiss) and os.path.exists(old_metadata):
        if os.path.exists(old_info):
            with open(old_info, "r", encoding="utf-8") as f:
                info = json.load(f)
            note = (
                f"Using a legacy flat vector_db/ layout built before "
                f"per-model subfolders existed. Re-run "
                f"scripts/07_generate_embeddings.py to migrate to "
                f"vector_db/{ACTIVE_EMBEDDING_MODEL_DIR}/."
            )
            return old_faiss, old_metadata, info, False, note

        fallback_info = {
            "label": "BAAI/bge-m3",
            "hf_id": "BAAI/bge-m3",
            "trust_remote_code": False,
            "task_aware": False,
        }
        note = (
            "No embedding_model_info.json found anywhere -- assuming "
            "BAAI/bge-m3 (the only option before multiple embedding "
            "models were supported). Re-run "
            f"scripts/07_generate_embeddings.py to build "
            f"vector_db/{ACTIVE_EMBEDDING_MODEL_DIR}/ properly."
        )
        return old_faiss, old_metadata, fallback_info, False, note

    raise FileNotFoundError(
        f"No vector_db/ found at either {new_dir}/ or the legacy flat "
        f"location. Run: python scripts/07_generate_embeddings.py"
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model_info():
    _, _, info, found, note = resolve_vector_db()
    return info, found, note


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    info, _, _ = load_embedding_model_info()
    kwargs = {"trust_remote_code": True} if info.get("trust_remote_code") else {}
    return SentenceTransformer(info["hf_id"], device="cpu", **kwargs)


@st.cache_resource
def load_index():
    faiss_path, _, _, _, _ = resolve_vector_db()
    return faiss.read_index(faiss_path)


@st.cache_resource
def load_metadata():
    _, metadata_path, _, _, _ = resolve_vector_db()
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_bm25_index(_chunks):
    """Builds the lexical BM25 index in memory at startup, directly from
    the same chunk metadata FAISS already loads. Unlike the FAISS index
    (built once offline by scripts/07_generate_embeddings.py and shipped
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
        header = (
            f"[Source {rank}] {doc_name} ({doc_type}, {release_date}) | "
            f"{chapter} > {section} | {legal_unit} | {chunk['chunk_id']}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    context_length = len(context)
    context_time = time.time() - context_start
    return context, context_length, context_time


def build_prompt(question, context, low_confidence=False):
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


def select_context_chunks(results):
    """Cap which retrieved chunks actually go into the LLM prompt: at most
    CONTEXT_K total, and no more than MAX_CHUNKS_PER_DOC from any single
    source document. Retrieval still searches/shows the full TOP_K (so
    diagnostics stays informative), but redundant near-duplicate chunks
    from the same document were the single biggest driver of prompt size
    (and therefore prefill/first-token time) without adding real coverage."""
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
    return results[:MAX_USED_SOURCES]


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

        if best_score < OUT_OF_DOMAIN_THRESHOLD:
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
            # docstring). Rejecting before BM25 search also means an
            # already-rejected question never pays for the extra lexical
            # search + fusion work at all.
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
        context, context_length, context_time = build_context(context_results)
        out_q.put(("phase", "constructing_prompt"))
        prompt = build_prompt(question, context, low_confidence=best_score < MIN_SCORE)
        context_tokens = count_tokens(context)
        prompt_tokens = count_tokens(prompt)

        out_q.put(("phase", "preparing_response"))

        gen_start = time.time()
        first_token_time = None
        total_chars = 0
        total_tokens_out = 0
        was_stopped = False

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            if stop_event.is_set():
                was_stopped = True
                break
            delta = chunk.choices[0].delta.content
            if delta:
                total_chars += len(delta)
                # llama-server streams one token per content delta, so this
                # count is a direct token count, not a character estimate.
                total_tokens_out += 1
                if first_token_time is None:
                    first_token_time = time.time() - gen_start
                    out_q.put(("phase", "generating_response"))
                out_q.put(("token", delta))

        try:
            response.close()
        except Exception:
            pass

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


def render_full_header(model_name):
    st.markdown(
        f'<div class="kc-welcome-wrap">'
        f'<div class="kc-eyebrow">RAG &middot; Insurance Law Corpus</div>'
        f'<div class="kc-welcome-lockup">'
        f"{SEAL_SVG}"
        f'<div class="kc-title-big">ជំនួយការច្បាប់ធានារ៉ាប់រង</div>'
        f"</div>"
        f'<div class="kc-subtitle">Khmer Insurance Law Assistant</div>'
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


def render_compact_topbar(model_name):
    st.markdown(
        f'<div class="kc-topbar">{SEAL_SVG}'
        f"<div>"
        f'<div class="kc-topbar-title">ជំនួយការច្បាប់ធានារ៉ាប់រង</div>'
        f'<div class="kc-topbar-sub">Khmer Insurance Law Assistant</div>'
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


def render_used_sources(used_sources):
    chips = []
    for score, chunk in used_sources:
        metadata = chunk.get("metadata", {})
        name = html.escape(metadata.get("document_name", "Unknown Document"))
        cid = html.escape(chunk["chunk_id"])
        chips.append(
            f'<div class="kc-source-chip"><div class="name">{name}</div>'
            f'<div class="id">{cid}</div></div>'
        )
    st.markdown(
        f"""
        <div class="kc-sources-title" style="color: var(--brass); padding-top: 20px;">
            ប្រភព
        </div>
        <div class="kc-sources-block">
            {"".join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Renders the mandatory disclaimer. Called from the single shared
    render_message() call site so every answer path -- normal,
    low-confidence, out-of-domain -- carries it structurally, and it's
    never part of LLM-generated text a user's question could influence."""
    st.markdown(
        f'<div class="kc-disclaimer">{html.escape(DISCLAIMER_TEXT)}</div>',
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
            st.markdown(
                '<div class="kc-alert">No Khmer text detected &mdash; '
                "this assistant only answers Khmer-language questions, so "
                "generation was skipped.</div>",
                unsafe_allow_html=True,
            )
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
        st.markdown(
            f'<div class="kc-msg-assistant">{html.escape(msg["answer"])}</div>',
            unsafe_allow_html=True,
        )
    if (
        not msg.get("interrupted")
        and not msg.get("errored")
        and not msg.get("out_of_domain")
    ):
        render_used_sources(msg["used_sources"])
        st.markdown('<div class="kc-msg-actions">', unsafe_allow_html=True)
        if st.button("មើលព័ត៌មានលម្អិត", key=f"diag_{msg['id']}"):
            st.session_state["view"] = "diagnostics"
            st.session_state["active_msg_id"] = msg["id"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer is shown on every real answer path -- normal, low-confidence,
    # and out-of-domain -- but not on interrupted/errored states, since those
    # aren't actual answers. This is the one render_message() call site every
    # message passes through, so there's no path that can skip it.
    if not msg.get("interrupted") and not msg.get("errored"):
        render_disclaimer()

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
        answer_ph.markdown(
            f'<div class="kc-msg-assistant">{html.escape(answer_text)}</div>',
            unsafe_allow_html=True,
        )
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
                answer_ph.markdown(
                    f'<div class="kc-msg-assistant">{html.escape(answer_text)}</div>',
                    unsafe_allow_html=True,
                )
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
        if reason == "scope_violation":
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
        st.session_state["messages"].append(
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
        st.session_state["messages"].append(
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
        st.session_state["messages"].append(msg)
    except Exception as e:
        st.session_state["messages"].append(
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

    if not messages and not pending and not st.session_state.get("is_generating"):
        render_full_header(model_name)
        render_welcome_form()
        return

    render_compact_topbar(model_name)
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
            st.session_state["messages"].append(
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
    """A small circular badge, deliberately styled like the app's own
    SEAL_SVG header mark, showing which retrieval engine(s) actually
    surfaced this chunk. The mapping is a literal nod to the physical
    seals (ត្រា) stamped on real Cambodian legal documents -- here, a
    chunk gets "stamped" by FAISS (semantic), by BM25 (lexical), or by
    both, and when both rankers agree the ring AND the center dot split
    half brass / half lacquer -- a literal half-and-half blend of the
    two engines' colors, not a third unrelated color standing in for
    "agreement".

    Built as a faint outer ring plus a bolder inner colored ring (rather
    than one thin ring), echoing SEAL_SVG's own concentric-ring
    construction and giving the badge enough visual weight to still read
    clearly at the ~13-18px size it's shown at in the legend.

    Colors are hardcoded hex (not CSS var()) to match the existing
    SEAL_SVG constant's own pattern, since SVG presentation attributes
    don't reliably resolve custom properties across browsers."""
    BRASS, LACQUER = "#B68B4C", "#9C2B1F"
    OUTER_RING = "#3A4A41"  # quiet neutral ring, independent of provenance color

    if has_faiss and has_bm25:
        inner = (
            f'<path d="M 3 10 A 7 7 0 0 1 17 10" fill="none" stroke="{BRASS}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
            f'<path d="M 3 10 A 7 7 0 0 0 17 10" fill="none" stroke="{LACQUER}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
        )
        # Center dot split the same way as the ring (top brass, bottom
        # lacquer) -- two filled half-discs sharing a flat diameter edge,
        # not a single solid color, so "both" reads as a literal half-and-
        # half blend of the two engines rather than a third, unrelated
        # color.
        dot = (
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 1 12.8 10 Z" fill="{BRASS}"/>'
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 0 12.8 10 Z" fill="{LACQUER}"/>'
        )
    elif has_faiss:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{BRASS}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{BRASS}"/>'
    elif has_bm25:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{LACQUER}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{LACQUER}"/>'
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
            metadata.get("document_type_kh", metadata.get("document_type", "N/A")),
        ),
        ("release date", metadata.get("release_date", "N/A")),
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
    msg = next(
        (
            m
            for m in st.session_state["messages"]
            if m["id"] == st.session_state["active_msg_id"]
        ),
        None,
    )

    if st.button("← Back to chat", key="back_to_chat"):
        st.session_state["view"] = "chat"
        st.session_state["active_msg_id"] = None
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
    st.markdown(
        '<div class="kc-alert">llama-server is not running. Start it before using the assistant.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
embedding_model_info, embedding_model_info_found, embedding_layout_note = (
    load_embedding_model_info()
)
embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()
bm25_index = load_bm25_index(chunks)

if embedding_layout_note:
    st.markdown(
        f'<div class="kc-alert">{html.escape(embedding_layout_note)}</div>',
        unsafe_allow_html=True,
    )

if st.session_state["view"] == "diagnostics":
    render_diagnostics_page()
else:
    render_chat_page(
        MODEL_NAME, client, embedding_model, embedding_model_info, index, bm25_index, chunks
    )