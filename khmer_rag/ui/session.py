"""Session initialization and conversation persistence."""

import time

import streamlit as st

from khmer_rag import storage as history


def open_chat():
    st.query_params.clear()


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


def ensure_conversation(question):
    """Create a conversation on the first question and return its ID."""
    if st.session_state["conversation_id"] is None:
        st.session_state["conversation_id"] = history.create_conversation(
            history.derive_title(question)
        )
    return st.session_state["conversation_id"]


def commit_message(msg):
    """Append the finished message to session state and persist it to SQLite."""
    st.session_state["messages"].append(msg)
    history.append_message(ensure_conversation(msg["question"]), msg)


def initialize_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("stop_event", None)
    st.session_state.setdefault("active_run", None)
    st.session_state.setdefault("pipeline_out_q", None)
    st.session_state.setdefault("pipeline_state", None)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("renaming_id", None)
    history.init_db()
