"""Runtime settings, retrieval limits, and project paths."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def is_local_generator_url(url):
    normalized = normalize_base_url(url).lower()
    return normalized.startswith(
        (
            "http://localhost:",
            "http://127.0.0.1:",
            "http://0.0.0.0:",
        )
    )


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
    HOSTED_DEFAULT_GENERATOR_BASE_URL if IS_HOSTED_RUNTIME else f"{DEFAULT_GENERATOR_ROOT}/v1"
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

USES_LOCAL_LLAMA_SERVER = is_local_generator_url(LLAMA_SERVER_URL)

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

DOC_FOCUS_LOOKAHEAD = 8

DOC_FOCUS_MIN_HITS = 3

DOC_FOCUS_CONTEXT_K = 4

BM25_TOP_K = TOP_K  # lexical candidate pool size, matched to FAISS's TOP_K

RRF_K = 60  # Reciprocal rank fusion smoothing constant.

NGRAM_SIZE = 3  # character n-gram size for Khmer-script BM25 tokens

DIAG_DISPLAY_K = 10  # Display limit; does not change retrieval.

MIN_SCORE = 0.49

OUT_OF_DOMAIN_THRESHOLD = 0.49

MIN_QUESTION_LENGTH = 8

MAX_USED_SOURCES = 3

GEN_FREQUENCY_PENALTY = 0.6

MAX_TOKENS_FLOOR = 512  # enough for legal lists, low enough to avoid demo-time runaways

GEN_TOKEN_HEADROOM = 1.25  # source-backed answers need room for Khmer list wording

SERVER_CTX_SIZE = 16384  # Shared with the local server launcher.

PROMPT_TOKEN_MARGIN = 256  # safety gap so prompt + max_tokens < ctx size

SERVER_OUTPUT_HARD_CAP = SERVER_CTX_SIZE - 9000  # ~7384 tokens

DEMO_OUTPUT_HARD_CAP = 4096

GEN_MAX_RETRIES = 3

GEN_RETRY_BACKOFF = 2.0

ACTIVE_EMBEDDING_MODEL_DIR = "BAAI-BGE-M3"

VECTOR_DB_ROOT = str(PROJECT_ROOT / "vector_db")
