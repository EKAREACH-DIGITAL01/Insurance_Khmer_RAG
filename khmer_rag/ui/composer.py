"""Composer UI components."""

import html

import streamlit as st

from khmer_rag.language import DISCLAIMER_TEXT
from khmer_rag.ui.branding import (
    CHATBOT_NAME,
    CHATBOT_SUBTITLE,
    IRC_LOGO_HTML,
    WELCOME_GREETING,
    WELCOME_HINT,
    WELCOME_SUBLINE,
)


def render_welcome_greeting():
    """The welcome state's centre column. The logo and title do NOT live here
    -- they stay in the fixed top bar in every state, so the identity never
    moves between the welcome screen and the thread. What's left in the middle
    is an invitation to type, which is the only thing this screen asks for."""
    st.markdown(
        f'<div class="kc-welcome-wrap">'
        f'<div class="kc-greeting">{html.escape(WELCOME_GREETING)}</div>'
        f'<div class="kc-greeting-sub">{html.escape(WELCOME_SUBLINE)}</div>'
        f'<div class="kc-greeting-hint">{html.escape(WELCOME_HINT)}</div>'
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


def render_rail_brand():
    """The product identity, at the top of the rail above the conversation
    list. It lives here rather than in a bar over the thread so the chat area
    is nothing but the conversation -- the identity sits with the navigation,
    and collapsing the rail gives the answer the full width."""
    st.markdown(
        f'<div class="kc-brand">{IRC_LOGO_HTML}'
        f"<div>"
        f'<div class="kc-brand-title">{CHATBOT_NAME}</div>'
        f'<div class="kc-brand-sub">{CHATBOT_SUBTITLE}</div>'
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


def render_disclaimer():
    """Render the disclaimer below the fixed input bar."""
    st.markdown(
        f'<div class="kc-disclaimer">{html.escape(DISCLAIMER_TEXT)}</div>',
        unsafe_allow_html=True,
    )
