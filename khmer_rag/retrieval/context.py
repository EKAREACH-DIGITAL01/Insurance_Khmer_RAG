"""Context selection and source limits."""

from khmer_rag.config import (
    CONTEXT_K,
    DOC_FOCUS_CONTEXT_K,
    DOC_FOCUS_LOOKAHEAD,
    DOC_FOCUS_MIN_HITS,
    MAX_CHUNKS_PER_DOC,
    MAX_USED_SOURCES,
)
from khmer_rag.sources import document_id_for_chunk


def context_doc_key(chunk):
    metadata = chunk.get("metadata", {})
    return (
        metadata.get("document_id")
        or document_id_for_chunk(chunk.get("chunk_id", ""))
        or metadata.get("document_name")
        or "unknown"
    )


def select_context_chunks(results):
    """Select bounded context, retaining adjacent chunks of a dominant document."""
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
    """Select source references from the context actually supplied to generation."""
    non_image_results = [
        (score, chunk) for score, chunk in results if not is_image_derived_source(chunk)
    ]
    return (non_image_results or results)[:MAX_USED_SOURCES]
