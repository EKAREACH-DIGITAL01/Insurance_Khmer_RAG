"""Queue consumption and resumable generation state for Streamlit."""

import html
import queue
import threading
import time
import uuid

import streamlit as st

from khmer_rag.config import (
    DIAG_DISPLAY_K,
    MIN_SCORE,
    TOP_K,
)
from khmer_rag.language import (
    NOT_KHMER_QUESTION_TEXT,
    REDIRECT_TEXT,
)
from khmer_rag.pipeline import run_pipeline_in_thread
from khmer_rag.retrieval.context import select_used_sources
from khmer_rag.text import (
    has_repetition_loop,
    normalize_answer_text,
)
from khmer_rag.ui.layout import scroll_to_bottom
from khmer_rag.ui.messages import render_answer_html
from khmer_rag.ui.progress import (
    PHASES,
    render_ladder_html,
)
from khmer_rag.ui.session import commit_message
from khmer_rag.ui.sources import render_used_sources


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
    ladder_ph.markdown(render_ladder_html(current_idx, phase_elapsed), unsafe_allow_html=True)

    final_payload = None
    error_text = None
    out_of_domain_payload = None
    canned_response_payload = None

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
                phase_elapsed[PHASES[current_idx]] = now - phase_start_times[PHASES[current_idx]]
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
                    render_answer_html(answer_text, final=False), unsafe_allow_html=True
                )
                if has_repetition_loop(answer_text):
                    stop_event.set()
            elif kind == "done":
                phase_elapsed[PHASES[current_idx]] = now - phase_start_times[PHASES[current_idx]]
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                final_payload = payload
                finished = True
                scroll_to_bottom()
            elif kind == "error":
                error_text = payload
                finished = True
            elif kind == "out_of_domain":
                phase_elapsed[PHASES[current_idx]] = now - phase_start_times[PHASES[current_idx]]
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                out_of_domain_payload = payload
                finished = True
                scroll_to_bottom()
            elif kind == "canned_response":
                phase_elapsed[PHASES[current_idx]] = now - phase_start_times[PHASES[current_idx]]
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                canned_response_payload = payload
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

    if canned_response_payload is not None:
        st.session_state["active_run"] = None
        answer = canned_response_payload["answer"]
        answer_ph.markdown(render_answer_html(answer), unsafe_allow_html=True)
        commit_message(
            {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "used_sources": [],
                "all_results": canned_response_payload["results"],
                "best_score": canned_response_payload["best_score"],
                "low_confidence": False,
                "interrupted": False,
                "no_sources": True,
                "canned_response_reason": canned_response_payload["reason"],
            }
        )
        return

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
