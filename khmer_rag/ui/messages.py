"""Messages UI components."""

import html

import streamlit as st

from khmer_rag.text import normalize_answer_text
from khmer_rag.ui.actions import render_message_actions
from khmer_rag.ui.sources import render_used_sources


def render_answer_html(answer, final=True):
    return f'<div class="kc-msg-assistant">{html.escape(normalize_answer_text(answer, final=final))}</div>'


def render_message(msg):
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(msg["question"])}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)
    if msg.get("low_confidence"):
        st.markdown(
            f'<div class="kc-alert">Low confidence retrieval &mdash; best match score '
            f"{msg['best_score']:.3f}. Treat this answer with extra caution.</div>",
            unsafe_allow_html=True,
        )
    if msg.get("interrupted"):
        st.markdown(
            '<div class="kc-alert">This question was interrupted before completion.</div>',
            unsafe_allow_html=True,
        )
        if msg.get("ladder_html"):
            st.markdown(msg["ladder_html"], unsafe_allow_html=True)
    if msg.get("errored"):
        st.markdown(
            f'<div class="kc-alert">Something went wrong while generating the answer: '
            f"{html.escape(msg.get('error_text', 'unknown error'))}</div>",
            unsafe_allow_html=True,
        )
    if msg.get("out_of_domain"):
        reason = msg.get("out_of_domain_reason")
        if reason == "not_khmer":
            pass
        elif reason == "too_short":
            st.markdown(
                '<div class="kc-alert">Question too short to search &mdash; '
                "generation was skipped.</div>",
                unsafe_allow_html=True,
            )
        elif reason == "scope_violation":
            st.markdown(
                '<div class="kc-alert">Scope boundary &mdash; this assistant '
                "cannot recommend or compare specific companies/products, give "
                "personalized advice, or calculate personal monetary amounts, "
                "so generation was skipped.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="kc-alert">Out of domain &mdash; best match score '
                f"{msg['best_score']:.3f} (below the relevance floor), so generation "
                f"was skipped.</div>",
                unsafe_allow_html=True,
            )
    if msg["answer"]:
        st.markdown(render_answer_html(msg["answer"]), unsafe_allow_html=True)
    if (
        not msg.get("interrupted")
        and not msg.get("errored")
        and not msg.get("out_of_domain")
        and not msg.get("no_sources")
    ):
        render_used_sources(msg["used_sources"])
        render_message_actions(msg)

    st.markdown("</div>", unsafe_allow_html=True)
