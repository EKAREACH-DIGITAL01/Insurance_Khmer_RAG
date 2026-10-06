"""Generator discovery, token counting, and output limits."""

import requests

from khmer_rag.config import (
    DEMO_OUTPUT_HARD_CAP,
    GEN_TOKEN_HEADROOM,
    GENERATOR_MODEL,
    LLAMA_SERVER_ROOT,
    LLAMA_SERVER_URL,
    MAX_TOKENS_FLOOR,
    PROMPT_TOKEN_MARGIN,
    SERVER_CTX_SIZE,
    SERVER_OUTPUT_HARD_CAP,
    USES_LOCAL_LLAMA_SERVER,
    env_value,
)


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
