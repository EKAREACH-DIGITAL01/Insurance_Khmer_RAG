"""Generator requests, cancellation, and retries before the first output."""

import logging
from collections.abc import Iterator
from threading import Event

from openai import OpenAI

from khmer_rag.config import (
    GEN_FREQUENCY_PENALTY,
    GEN_MAX_RETRIES,
    GEN_RETRY_BACKOFF,
    USES_LOCAL_LLAMA_SERVER,
)

logger = logging.getLogger(__name__)


def stream_answer(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    stop_event: Event,
) -> Iterator[str]:
    """Yield text deltas; never restart an answer the user has already seen."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    # Gemini's compatibility endpoint rejects this llama-server option.
    if USES_LOCAL_LLAMA_SERVER:
        request["frequency_penalty"] = GEN_FREQUENCY_PENALTY

    emitted = False
    for attempt in range(1, GEN_MAX_RETRIES + 1):
        if stop_event.is_set():
            return
        response = None
        try:
            response = client.chat.completions.create(**request)
            for chunk in response:
                if stop_event.is_set():
                    return
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    emitted = True
                    yield content
            return
        except Exception:
            if emitted or attempt == GEN_MAX_RETRIES:
                raise
            logger.warning(
                "Generator request failed before output; retry %s/%s", attempt, GEN_MAX_RETRIES
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    logger.debug("Failed to close generator stream", exc_info=True)
        if stop_event.wait(GEN_RETRY_BACKOFF):
            return
