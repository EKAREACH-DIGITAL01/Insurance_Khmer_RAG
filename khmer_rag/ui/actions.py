"""Actions UI components."""

import base64
import html
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

from khmer_rag.text import normalize_answer_text
from khmer_rag.ui.assets import script_html


def install_action_row_script():
    """Install delegated copy and reaction handlers once per browser document."""
    components.html(
        script_html("install_action_row_script.js"),
        height=0,
    )


_ICON_COPY = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="11" height="11" rx="2"/>'
    '<path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>'
)

_ICON_THUMB_UP = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 21V10l5-7a2 2 0 0 1 3 2l-1.2 5H19a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 '
    '17.6 21H7z"/><rect x="3" y="10" width="4" height="11" rx="1"/></svg>'
)

_ICON_THUMB_DOWN = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M17 3v11l-5 7a2 2 0 0 1-3-2l1.2-5H5a2 2 0 0 1-2-2.4l1.4-7A2 2 0 0 1 '
    '6.4 3H17z"/><rect x="17" y="3" width="4" height="11" rx="1"/></svg>'
)

_ICON_INFO = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.6v.4"/></svg>'
)


def render_message_actions(msg):
    """Render copy, thumbs up, thumbs down, and diagnostics as one action row."""
    answer = normalize_answer_text(msg.get("answer") or "")
    answer_b64 = base64.b64encode(answer.encode("utf-8")).decode("ascii")
    msg_id = quote(str(msg["id"]), safe="")
    feedback = msg.get("feedback")
    up_cls = "kc-act kc-act-thumb" + (" is-on" if feedback == "up" else "")
    down_cls = "kc-act kc-act-thumb" + (" is-on" if feedback == "down" else "")
    diag_url = f"?view=diagnostics&amp;msg={html.escape(str(msg['id']))}"
    up_url = html.escape(f"?feedback_msg={msg_id}&feedback=up")
    down_url = html.escape(f"?feedback_msg={msg_id}&feedback=down")
    st.markdown(
        f'<div class="kc-actions">'
        f'<button type="button" class="kc-act kc-act-copy" data-copy-b64="{answer_b64}" '
        f'data-tip="ចម្លង" aria-label="ចម្លង">{_ICON_COPY}</button>'
        f'<a class="{up_cls}" href="{up_url}" data-tip="ពេញចិត្ត" '
        f'aria-label="ពេញចិត្ត" aria-pressed="{str(feedback == "up").lower()}">{_ICON_THUMB_UP}</a>'
        f'<a class="{down_cls}" href="{down_url}" data-tip="មិនពេញចិត្ត" '
        f'aria-label="មិនពេញចិត្ត" aria-pressed="{str(feedback == "down").lower()}">{_ICON_THUMB_DOWN}</a>'
        f'<a class="kc-act" href="{diag_url}" target="_blank" rel="noopener" '
        f'data-tip="មើលព័ត៌មានលម្អិត" aria-label="មើលព័ត៌មានលម្អិត">{_ICON_INFO}</a>'
        f"</div>",
        unsafe_allow_html=True,
    )
