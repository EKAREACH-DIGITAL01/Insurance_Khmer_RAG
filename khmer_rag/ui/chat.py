"""Chat UI components."""

import uuid

import streamlit as st

from khmer_rag.language import (
    GREETING_RESPONSE_TEXT,
    is_khmer_greeting,
)
from khmer_rag.ui.composer import (
    render_disclaimer,
    render_input_bar,
    render_welcome_form,
    render_welcome_greeting,
)
from khmer_rag.ui.layout import scroll_to_bottom
from khmer_rag.ui.messages import render_message
from khmer_rag.ui.session import commit_message
from khmer_rag.ui.streaming import handle_new_question


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

    if pending and is_khmer_greeting(pending):
        commit_message(
            {
                "id": str(uuid.uuid4()),
                "question": pending,
                "answer": GREETING_RESPONSE_TEXT,
                "used_sources": [],
                "all_results": [],
                "best_score": 0.0,
                "low_confidence": False,
                "interrupted": False,
                "no_sources": True,
                "canned_response_reason": "khmer_greeting",
            }
        )
        st.session_state["pending_question"] = None
        st.session_state["is_generating"] = False
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()

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
