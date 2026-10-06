"""Layout UI components."""

from string import Template

import streamlit as st
import streamlit.components.v1 as components

from khmer_rag.ui.assets import (
    UI_ASSET_DIR,
    script_html,
)


def sync_rail_width():
    """Keep the fixed composer aligned with Streamlit's sidebar width."""
    components.html(
        script_html("sync_rail_width.js"),
        height=0,
    )


def disable_rail_resize():
    """Hide Streamlit's resize handle for the fixed-width sidebar."""
    components.html(
        script_html("disable_rail_resize.js"),
        height=0,
    )


def scroll_to_bottom():
    """Scroll the active Streamlit viewport to the latest message."""
    components.html(
        script_html("scroll_to_bottom.js"),
        height=0,
    )


def inject_css(is_diagnostics=False):
    PAGE_BG = "var(--ink)" if is_diagnostics else "var(--parchment)"
    PAGE_FG = "var(--mist)" if is_diagnostics else "var(--stone)"
    BUTTON_HOVER_COLOR = "var(--signal)" if is_diagnostics else "var(--lacquer)"
    TOP_PADDING_CSS = (
        '[data-testid="stAppViewContainer"] .block-container { padding-top: 1.2rem !important; }'
        if is_diagnostics
        # Chat owns both ends: 9px above (which, plus the first message row's own
        # 18px margin, lands the opening question level with the rail's logo) and
        # enough below to clear the fixed chat bar and disclaimer.
        else '[data-testid="stAppViewContainer"] .block-container '
        "{ padding-top: 9px !important; padding-bottom: 150px !important; }"
    )
    template = Template((UI_ASSET_DIR / "chat.css").read_text(encoding="utf-8"))
    css = template.substitute(
        BUTTON_HOVER_COLOR=BUTTON_HOVER_COLOR,
        PAGE_BG=PAGE_BG,
        PAGE_FG=PAGE_FG,
        TOP_PADDING_CSS=TOP_PADDING_CSS,
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def install_keyboard_handler():
    components.html(
        script_html("install_keyboard_handler.js"),
        height=0,
    )
