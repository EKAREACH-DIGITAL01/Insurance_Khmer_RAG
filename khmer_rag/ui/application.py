"""Explicit Streamlit startup and page routing."""

import html

import streamlit as st
from openai import OpenAI

from khmer_rag.config import (
    DEPLOYMENT_COMMIT,
    GENERATOR_API_KEY,
    LLAMA_SERVER_URL,
    USES_LOCAL_LLAMA_SERVER,
)
from khmer_rag.generation.client import get_loaded_model
from khmer_rag.ui.actions import install_action_row_script
from khmer_rag.ui.branding import (
    CHATBOT_NAME,
    FAVICON_PATH,
)
from khmer_rag.ui.chat import render_chat_page
from khmer_rag.ui.diagnostics import render_diagnostics_page
from khmer_rag.ui.layout import (
    disable_rail_resize,
    inject_css,
    install_keyboard_handler,
    sync_rail_width,
)
from khmer_rag.ui.resources import (
    load_bm25_index,
    load_embedding_model,
    load_embedding_model_info,
    load_index,
    load_metadata,
)
from khmer_rag.ui.session import (
    initialize_session,
    process_feedback_query,
)
from khmer_rag.ui.sidebar import render_sidebar


def main():
    message_id = st.query_params.get("msg")
    is_diagnostics = st.query_params.get("view") == "diagnostics" and bool(message_id)
    st.set_page_config(
        page_title=CHATBOT_NAME,
        page_icon=FAVICON_PATH,
        layout="wide" if is_diagnostics else "centered",
        # Diagnostics is a full-width single-purpose readout with its own "Back
        # to chat" control -- the conversation rail is noise there.
        initial_sidebar_state="collapsed" if is_diagnostics else "expanded",
    )
    initialize_session()
    process_feedback_query()

    inject_css(is_diagnostics)
    install_keyboard_handler()

    if is_diagnostics:
        render_diagnostics_page(message_id)
        return

    model_name = get_loaded_model()

    if model_name == "No Model Loaded":
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

    # Sidebar actions can change the active conversation before the thread renders.
    render_sidebar()
    sync_rail_width()
    disable_rail_resize()
    install_action_row_script()
    render_chat_page(
        model_name, client, embedding_model, embedding_model_info, index, bm25_index, chunks
    )
