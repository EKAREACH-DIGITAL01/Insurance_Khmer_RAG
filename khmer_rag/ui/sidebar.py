"""Sidebar UI components."""

import streamlit as st
import streamlit.components.v1 as components

from khmer_rag import storage as history
from khmer_rag.ui.assets import script_html
from khmer_rag.ui.composer import render_rail_brand
from khmer_rag.ui.session import ensure_conversation


def open_conversation(conv_id):
    """Swaps the working conversation. Any in-flight generation state belongs
    to the conversation being left, so it's dropped here rather than carried
    across -- the background thread is already told to stop by the caller."""
    st.session_state["conversation_id"] = conv_id
    st.session_state["messages"] = history.load_messages(conv_id) if conv_id else []
    st.session_state["pending_question"] = None
    st.session_state["is_generating"] = False
    st.session_state["active_run"] = None
    st.session_state["stop_event"] = None
    st.session_state["pipeline_out_q"] = None
    st.session_state["pipeline_state"] = None
    st.session_state["renaming_id"] = None


def commit_rename(conv_id, widget_key):
    """Saves an inline row rename and closes the editor.

    Runs as the text input's on_change, so Enter and clicking away both commit
    -- there is no separate save button to hunt for. An empty or whitespace-only
    title is ignored by history.rename_conversation(), which leaves the row
    named as it was rather than blank."""
    history.rename_conversation(conv_id, st.session_state.get(widget_key, ""))
    st.session_state["renaming_id"] = None


def focus_rename_input():
    """Puts the cursor in the inline rename field with its text selected.

    Streamlit has no autofocus, and a rename box you still have to click into
    is a rename box that looks broken. Runs in the usual 0-height iframe and
    reaches into window.parent, same as the other helpers here."""
    components.html(
        script_html("focus_rename_input.js"),
        height=0,
    )


def render_sidebar():
    """The conversation rail.

    Rows are disabled while an answer is streaming: switching conversations
    mid-generation would orphan the background thread and file its answer
    against whichever conversation happened to be open when it finished. Stop
    the answer first, then switch -- the disabled state says so.
    """
    # A question accepted on the previous run has not reached commit_message()
    # yet -- its conversation is opened here, before the first row is drawn,
    # so the rail shows it selected from the moment generation starts.
    pending = st.session_state.get("pending_question")
    if pending:
        ensure_conversation(pending)

    is_gen = st.session_state["is_generating"]
    active_id = st.session_state["conversation_id"]

    with st.sidebar:
        render_rail_brand()
        if st.button(
            "ការសន្ទនាថ្មី",
            key="kc_rail_new",
            icon=":material/add:",
            type="tertiary",
            width="stretch",
            disabled=is_gen or (not st.session_state["messages"] and active_id is None),
        ):
            open_conversation(None)
            st.rerun()

        st.markdown('<div class="kc-rail-label">ការសន្ទនា</div>', unsafe_allow_html=True)

        if st.session_state["renaming_id"] is not None:
            focus_rename_input()

        conversations = history.list_conversations()
        if not conversations:
            st.markdown(
                '<div class="kc-rail-empty">មិនមានការសន្ទនា</div>',
                unsafe_allow_html=True,
            )
            return

        for conv in conversations:
            conv_id = conv["id"]
            is_active = conv_id == active_id
            # The active fill and the hover fill are painted on this wrapper
            # rather than on the title button, so they span the whole row --
            # the overflow menu sits inside the highlight instead of stranded
            # on the rail beside it. The key encodes the state because CSS is
            # the only thing that can style a Streamlit container.
            row_key = f"kc_rail_row_active_{conv_id}" if is_active else f"kc_rail_row_{conv_id}"
            if st.session_state["renaming_id"] == conv_id:
                # The row turns into its own editor: the title is edited where
                # the title is, not in a field tucked inside the menu that
                # opened it. Enter (or clicking away) commits via on_change.
                with st.container(key=f"kc_rail_editing_{conv_id}"):
                    widget_key = f"kc_rail_rename_{conv_id}"
                    st.text_input(
                        "ឈ្មោះការសន្ទនា",
                        value=conv["title"],
                        label_visibility="collapsed",
                        key=widget_key,
                        on_change=commit_rename,
                        args=(conv_id, widget_key),
                    )
                continue

            with st.container(key=row_key):
                col_title, col_menu = st.columns([1, 0.17])
                with col_title:
                    if st.button(
                        conv["title"],
                        key=f"kc_rail_open_{conv_id}",
                        width="stretch",
                        type="tertiary",
                        disabled=is_gen,
                        help=("បញ្ឈប់ចម្លើយមុននឹងប្តូរការសន្ទនា" if is_gen else None),
                    ):
                        if not is_active:
                            open_conversation(conv_id)
                        st.rerun()
                with col_menu:
                    with st.popover(
                        "⋯",
                        width="stretch",
                        disabled=is_gen,
                        key=f"kc_rail_menu_{conv_id}",
                    ):
                        if st.button(
                            "ផ្លាស់ប្តូរឈ្មោះ",
                            key=f"kc_rail_rename_open_{conv_id}",
                            icon=":material/edit:",
                            width="stretch",
                        ):
                            st.session_state["renaming_id"] = conv_id
                            st.rerun()
                        if st.button(
                            "លុបការសន្ទនា",
                            key=f"kc_rail_delete_{conv_id}",
                            icon=":material/delete:",
                            width="stretch",
                        ):
                            history.delete_conversation(conv_id)
                            if is_active:
                                # The open conversation just stopped existing
                                # -- land on a fresh one rather than a thread
                                # whose messages have no home to be saved to.
                                open_conversation(None)
                            st.rerun()
