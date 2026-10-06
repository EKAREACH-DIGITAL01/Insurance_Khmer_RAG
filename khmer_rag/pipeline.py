"""Background retrieval and generation; communicates through queue events."""

import logging
import time

from khmer_rag.answers import (
    augment_curated_context,
    has_curated_source_candidate,
    resolve_curated_source_answer,
)
from khmer_rag.config import (
    MIN_QUESTION_LENGTH,
    MIN_SCORE,
    OUT_OF_DOMAIN_THRESHOLD,
)
from khmer_rag.generation.client import (
    choose_max_tokens_live,
    count_tokens,
)
from khmer_rag.generation.prompts import (
    build_context,
    build_prompt,
)
from khmer_rag.generation.stream import stream_answer
from khmer_rag.language import (
    GREETING_RESPONSE_TEXT,
    contains_khmer_script,
    is_khmer_greeting,
)
from khmer_rag.policy import is_scope_violation
from khmer_rag.provisions import resolve_provision_lookup
from khmer_rag.retrieval.context import select_context_chunks
from khmer_rag.retrieval.search import (
    encode_query,
    reciprocal_rank_fusion,
    search_bm25,
    search_faiss,
)

logger = logging.getLogger(__name__)


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

        if is_khmer_greeting(question):
            out_q.put(("phase", "completed"))
            out_q.put(
                (
                    "canned_response",
                    {
                        "answer": GREETING_RESPONSE_TEXT,
                        "reason": "khmer_greeting",
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
            question, results, chunks
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
            corpus=chunks,
        )
        context, context_length, context_time = build_context(context_results)
        out_q.put(("phase", "constructing_prompt"))
        prompt = build_prompt(
            question,
            context,
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
        elif curated_answer := resolve_curated_source_answer(question, context_results, best_score):
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

            for delta in stream_answer(client, model_name, prompt, max_tokens, stop_event):
                answer_text += delta
                total_chars += len(delta)
                # A delta is one token for llama-server, an estimate for hosted APIs.
                total_tokens_out += 1
                if first_token_time is None:
                    first_token_time = time.time() - gen_start
                    out_q.put(("phase", "generating_response"))
                out_q.put(("token", delta))
            was_stopped = stop_event.is_set()

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
        logger.exception("Answer pipeline failed")
        out_q.put(("error", str(e)))
