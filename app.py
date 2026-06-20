import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss

faiss.omp_set_num_threads(1)

import html
import json
import queue
import threading
import time
import uuid

import numpy as np
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LLAMA_SERVER_URL = "http://localhost:8080/v1"
LLAMA_SERVER_ROOT = "http://localhost:8080"

TOP_K = 10
MIN_SCORE = 0.50
USED_SOURCE_MARGIN = (
    0.03  # how close a score must be to the top score to also count as "used"
)
MAX_USED_SOURCES = 3

FAISS_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EMBEDDING_MODEL_PATH = "models/retriever/BAAI-BGE-M3"

PHASES = [
    "encoding_query",
    "searching_faiss",
    "building_context",
    "constructing_prompt",
    "generating_response",
    "completed",
]
PHASE_LABELS = {
    "encoding_query": (
        "<b>Encoding Query</b> - "
        "<i>BGE-M3 is converting the question into an embedding vector</i>"
    ),
    "searching_faiss": (
        "<b>Searching FAISS Index</b> - "
        "<i>FAISS is searching the vector database for relevant chunks</i>"
    ),
    "building_context": (
        "<b>Building Retrieval Context</b> - "
        "<i>Retrieved chunks are being assembled into context</i>"
    ),
    "constructing_prompt": (
        "<b>Constructing Prompt</b> - "
        "<i>Context, instructions, and question are being merged</i>"
    ),
    "generating_response": (
        "<b>Generating Response</b> - "
        "<i>SeaLLMs is generating the answer from retrieved context</i>"
    ),
    "completed": ("<b>Completed</b>"),
}

SEAL_SVG = """<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><circle cx="20" cy="20" r="18" fill="none" stroke="#B68B4C" stroke-width="1.1"/><circle cx="20" cy="20" r="12" fill="none" stroke="#B68B4C" stroke-width="1.1"/><g stroke="#B68B4C" stroke-width="1.1"><line x1="20" y1="2" x2="20" y2="8"/><line x1="20" y1="32" x2="20" y2="38"/><line x1="2" y1="20" x2="8" y2="20"/><line x1="32" y1="20" x2="38" y2="20"/><line x1="7.5" y1="7.5" x2="11.5" y2="11.5"/><line x1="28.5" y1="28.5" x2="32.5" y2="32.5"/><line x1="7.5" y1="32.5" x2="11.5" y2="28.5"/><line x1="28.5" y1="11.5" x2="32.5" y2="7.5"/></g><circle cx="20" cy="20" r="3" fill="#B68B4C"/></svg>"""

favicon = Image.open("assets/favicon.png")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("view", "chat")
st.session_state.setdefault("active_msg_id", None)
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("is_generating", False)
st.session_state.setdefault("stop_event", None)
st.session_state.setdefault("active_run", None)

IS_DIAGNOSTICS = st.session_state["view"] == "diagnostics"

st.set_page_config(
    page_title="Khmer Insurance Law Assistant",
    page_icon=favicon,
    layout="wide" if IS_DIAGNOSTICS else "centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

PAGE_BG = "var(--ink)" if IS_DIAGNOSTICS else "var(--parchment)"
PAGE_FG = "var(--mist)" if IS_DIAGNOSTICS else "var(--stone)"
BUTTON_HOVER_COLOR = "var(--signal)" if IS_DIAGNOSTICS else "var(--lacquer)"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: #15201C;
  --ink-soft: #1D2B25;
  --ink-line: #2C3D34;
  --parchment: #EDE6D6;
  --parchment-soft: #F5F0E4;
  --lacquer: #9C2B1F;
  --lacquer-deep: #7A2118;
  --lacquer-mist: rgba(156,43,31,0.08);
  --brass: #B68B4C;
  --brass-soft: #D8BD8B;
  --stone: #3A352B;
  --mist: #9DB6AB;
  --signal: #6FBF8B;
  --danger: #C0524A;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stToolbar"] {{display: none !important;}}
[data-testid="stDecoration"] {{display: none !important;}}
[data-testid="stStatusWidget"] {{display: none !important;}}
[data-testid="stForm"] div[data-testid="InputInstructions"] {{display: none !important;}}
header {{display: none !important;}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {PAGE_BG} !important;
  color: {PAGE_FG};
  font-family: 'IBM Plex Sans', sans-serif;
}}

.kc-welcome-wrap {{ margin-top: 16vh; text-align: center; }}
.kc-welcome-lockup {{
  display: inline-flex;
  align-items: center;
  gap: 20px;
  text-align: left;
}}
.kc-welcome-lockup svg {{
  display: block;
  flex-shrink: 0;
  transform: translateY(4px);
}}
.kc-eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--brass);
  text-align: center;
}}
.kc-title-big {{
  font-family: 'Moul', serif !important;
  font-size: 30px !important;
  color: var(--lacquer) !important;
  margin: 17px 0 0 0 !important;
  line-height: 1.3 !important;
  white-space: nowrap;
}}
.kc-subtitle {{
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 13px;
  color: var(--stone);
  opacity: 0.7;
  margin-top: 16px;
}}
.kc-model-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--ink);
  color: var(--parchment-soft);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  padding: 6px 14px;
  border-radius: 999px;
  margin-top: 10px;
}}
.kc-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.kc-dot-on {{ background: var(--signal); box-shadow: 0 0 0 3px rgba(111,191,139,0.25); }}
.kc-dot-off {{ background: var(--danger); box-shadow: 0 0 0 3px rgba(192,82,74,0.25); }}

/* welcome-only input: in-flow under the title, NOT pinned. Arrow is laid out
   as a real adjacent column rather than absolutely positioned, since the
   latter kept landing in the wrong place. */
[data-testid="stForm"] {{
  background: var(--parchment-soft) !important;
  border: 1px solid var(--brass-soft) !important;
  border-radius: 26px !important;
  padding: 6px 16px !important;
  box-shadow: 0 4px 18px rgba(58,53,43,0.12);
  max-width: 560px;
  width: 100%;
  margin: 32px auto 0 auto !important;
}}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
  align-items: center !important;
  gap: 0.5rem !important;
}}
[data-testid="stForm"] [data-testid="column"] {{ padding: 0 !important; }}
[data-testid="stForm"] [data-testid="stWidgetLabel"] {{ display: none; }}
[data-testid="stForm"] [data-testid="stTextInput"],
[data-testid="stForm"] [data-testid="stTextInput"] > div,
[data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-testid="stBaseButton-secondary"],
[data-testid="stForm"] [data-testid="stTextInput"] div[class*="BaseInput"] {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input {{
  background: transparent !important;
  border: none !important;
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  color: var(--stone);
  padding: 10px 8px;
  box-shadow: none !important;
}}
[data-testid="stForm"] [data-baseweb="input"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

[data-testid="stForm"] [data-baseweb="input"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {{ box-shadow: none !important; }}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] {{
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
  background: var(--lacquer) !important;
  color: var(--parchment-soft) !important;
  border: none !important;

  width: 42px !important;
  height: 42px !important;
  min-width: 42px !important;

  border-radius: 50% !important;

  margin-right: 0px !important;
  margin-left: 10px !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  font-size: 28px !important;
  font-weight: 800 !important;
  line-height: 1 !important;

  box-shadow: none !important;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button::before {{
    content: "↑";
    font-size: 20px;
    font-weight: 500;
    color: white;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
  background: var(--lacquer-deep) !important;
  transform: translateY(-1px);
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:focus-visible {{
  outline: 2px solid var(--brass);
  outline-offset: 2px;
}}

/* ---------- Compact fixed top bar (chat state) ---------- */
.kc-topbar {{
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 998;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 20px;
  background: var(--parchment);
  border-bottom: 1px solid var(--brass-soft);
}}
.kc-topbar-title {{ font-family: 'Moul', serif; font-size: 15px; color: var(--lacquer); }}
.kc-topbar-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--brass); opacity: 0.85; }}

.kc-thread {{ padding-top: 64px; }}

.kc-msg-user-row {{ display: flex; justify-content: flex-end; margin: 18px 0 6px 0; }}
.kc-msg-user {{
  background: var(--lacquer);
  color: var(--parchment-soft);
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  line-height: 1.7;
  padding: 10px 16px;
  border-radius: 14px 14px 2px 14px;
  max-width: 80%;
  white-space: pre-wrap;
}}

.kc-msg-assistant-row {{ margin: 4px 0 22px 0; }}
.kc-msg-assistant {{
  font-family: 'Hanuman', sans-serif;
  font-size: 16.5px;
  line-height: 1.9;
  color: var(--stone);
  white-space: pre-wrap;
  min-height: 1.4em;
}}

.kc-sources-block {{ margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }}
.kc-source-chip {{ border-left: 2px solid var(--brass); padding: 4px 0 4px 12px; }}
.kc-source-chip .name {{ font-family: 'Hanuman', sans-serif; font-size: 13.5px; color: var(--stone); }}
.kc-source-chip .id {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--brass); margin-top: 2px; }}

.kc-msg-actions {{ margin-top: 10px; }}

[data-testid="stButton"] button {{
  background: transparent !important;
  border: none !important;
  color: var(--brass) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 4px 0 !important;
  box-shadow: none !important;
}}
[data-testid="stButton"] button:hover {{ color: {BUTTON_HOVER_COLOR} !important; text-decoration: underline; }}

/* ---------- Native chat input (bottom-pinned by Streamlit itself) ---------- */
[data-testid="stChatInput"] {{
  background: var(--parchment-soft) !important;
  border: 1px solid var(--brass-soft) !important;
  border-radius: 26px !important;
  box-shadow: 0 4px 18px rgba(58,53,43,0.12);
}}
[data-testid="stChatInput"] textarea {{
  font-family: 'Hanuman', sans-serif !important;
  font-size: 15.5px !important;
  color: var(--stone) !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--stone); opacity: 0.5; }}
[data-testid="stChatInputSubmitButton"] button,
[data-testid="stChatInputSubmitButton"] {{
  background: var(--lacquer) !important;
  border-radius: 50% !important;
}}
[data-testid="stChatInputSubmitButton"] svg {{ fill: var(--parchment-soft) !important; }}
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stApp"] [data-testid="stBottom"] {{
  background: var(--parchment) !important;
}}

.kc-alert {{
  background: var(--lacquer-mist);
  border: 1px solid var(--lacquer);
  color: var(--lacquer-deep);
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 12.5px;
  padding: 8px 12px;
  border-radius: 4px;
  margin: 4px 0 10px 0;
  display: inline-block;
}}

/* ---------- Progress ladder ---------- */
.kc-ladder {{
  border-left: 2px solid var(--brass-soft);
  padding-left: 14px;
  display: flex;
  flex-direction: column;
  gap: 9px;
  margin: 6px 0 18px 0;
}}
.kc-step-row {{ display: flex; align-items: center; }}
.kc-step-row-pending {{ opacity: 0.4; }}
.kc-step-icon {{
  width: 14px; height: 14px;
  border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 9px; color: var(--ink);
  margin-right: 9px; flex-shrink: 0;
}}
.kc-step-done {{ background: var(--signal); }}
.kc-step-pending {{ background: var(--brass-soft); }}
.kc-step-active {{ background: var(--lacquer); animation: kc-pulse 1s ease-in-out infinite; }}
@keyframes kc-pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.5; transform: scale(0.75); }} }}
.kc-step-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--stone); }}
.kc-step-time {{ margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--brass); }}

/* ---------- Diagnostics page (ink register) ---------- */
.kc-diag-header {{
  display: flex; align-items: baseline; gap: 14px;
  border-bottom: 1px solid var(--ink-line);
  padding-bottom: 14px; margin-bottom: 18px;
}}
.kc-diag-title {{
  font-family: 'IBM Plex Mono', monospace; font-size: 13px;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--brass);
}}
.kc-diag-question {{ font-family: 'Hanuman', sans-serif; font-size: 14px; color: var(--parchment-soft); opacity: 0.85; }}
.kc-console-block {{
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 14px;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.kc-console-block:hover {{
  transform: translateY(-2px);
  border-color: var(--brass);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
.kc-console-title {{
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
  letter-spacing: 0.08em; color: var(--brass); text-transform: uppercase; margin-bottom: 8px;
}}
.kc-row {{
  display: flex; justify-content: space-between; gap: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  padding: 3px 0; border-bottom: 1px dashed rgba(157,182,171,0.15);
}}
.kc-row:last-child {{ border-bottom: none; }}
.kc-row-label {{ color: var(--mist); opacity: 0.85; }}
.kc-row-value {{ color: var(--parchment-soft); text-align: right; }}

.kc-chunk-row {{
  display: grid; grid-template-columns: 28px 60px 1fr 60px; gap: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
  padding: 6px 0; border-bottom: 1px dashed rgba(157,182,171,0.12); align-items: center;
}}
.kc-chunk-row:last-child {{ border-bottom: none; }}
.kc-chunk-rank {{ color: var(--brass); }}
.kc-chunk-score {{ color: var(--signal); }}
.kc-chunk-id {{ color: var(--parchment-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.kc-chunk-len {{ color: var(--mist); opacity: 0.7; text-align: right; }}

/* ---------- Diagnostics source cards (right column, ink register) ---------- */
.kc-diag-src-list {{ display: flex; flex-direction: column; gap: 10px; }}
.kc-diag-src-card {{
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 6px;
  padding: 14px 16px;
  opacity: 0;
  animation: kc-rise 0.5s ease forwards;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.kc-diag-src-card:hover {{
  transform: translateY(-2px);
  border-color: var(--brass);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
.kc-diag-src-top {{ display: flex; align-items: flex-start; gap: 12px; }}
.kc-diag-src-rank {{
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--brass);
  width: 20px; flex-shrink: 0; padding-top: 2px;
}}
.kc-diag-src-main {{ flex: 1; min-width: 0; }}
.kc-diag-src-name {{ font-family: 'Hanuman', sans-serif; font-size: 14px; color: var(--parchment-soft); line-height: 1.5; }}
.kc-diag-src-id {{ font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--brass); margin-top: 4px; }}
.kc-diag-src-score {{
  font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: var(--signal);
  font-weight: 600; flex-shrink: 0;
}}
.kc-diag-src-bar {{ width: 100%; height: 5px; background: var(--ink-line); border-radius: 3px; overflow: hidden; margin-top: 10px; }}
.kc-diag-src-fill {{
  height: 100%; width: 0; border-radius: 3px;
  background: linear-gradient(90deg, var(--brass), var(--lacquer));
  animation: kc-fill 0.9s cubic-bezier(.2,.8,.2,1) forwards;
}}
@keyframes kc-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes kc-fill {{ from {{ width: 0; }} to {{ width: var(--w, 0%); }} }}

.kc-detail-grid {{
  display: grid; grid-template-columns: auto 1fr; gap: 4px 14px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; margin-bottom: 12px;
}}
.kc-detail-grid .k {{ color: var(--brass); }}
.kc-detail-grid .v {{ color: var(--parchment-soft); }}
.kc-detail-text {{
  font-family: 'Hanuman', sans-serif; font-size: 14px; line-height: 1.85; color: var(--mist);
  border-top: 1px dashed var(--ink-line); padding-top: 12px; white-space: pre-wrap;
}}

/* Full Source Details -- back to native st.expander: simple, reliable,
   guaranteed clickable. Styled to match the ink card system. The
   .kc-diag-page wrapping div used to scope these never actually contained
   the expanders in the real DOM (opening/closing a div across separate
   st.markdown calls doesn't nest the content in between), so every rule
   below was previously matching nothing. st.expander is only used on this
   one page in the whole app, so these are safe to leave unscoped. */
[data-testid="stExpander"] {{
  background: var(--ink-soft) !important;
  border: 1px solid var(--ink-line) !important;
  border-radius: 6px !important;
  margin-bottom: 10px !important;
  overflow: hidden;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
[data-testid="stExpander"]:hover {{
  transform: translateY(-2px);
  border-color: var(--brass) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}}
/* Force every background inside the expander to transparent first -- we
   don't know exactly which element/attribute Streamlit uses to paint the
   "open" state in this version, so rather than guess a single selector,
   blank everything out and re-apply our own color via more specific rules
   below (which win on specificity regardless of source order). */
[data-testid="stExpander"] * {{
  background-color: transparent !important;
}}
[data-testid="stExpander"] summary {{
  font-family: 'IBM Plex Mono', monospace !important;
  color: var(--parchment-soft) !important;
  font-size: 12px !important;
  padding: 12px 16px !important;
  transition: background-color 0.15s ease, color 0.15s ease;
}}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details[open] summary,
[data-testid="stExpander"] summary[aria-expanded="true"],
[data-testid="stExpander"] [aria-expanded="true"] summary {{
  background-color: rgba(182,139,76,0.1) !important;
  color: var(--brass) !important;
}}
[data-testid="stExpander"] summary svg {{ fill: var(--brass) !important; }}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
  padding: 4px 16px 16px 16px !important;
  border-top: 1px solid var(--ink-line);
}}

/* ---------- Stop / interrupt bar (replaces chat input while generating) ---------- */
.st-key-kc_stopbar {{
  position: fixed;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  z-index: 999;
  max-width: 560px;
  width: calc(100% - 48px);
}}
.st-key-kc_stopbar [data-testid="stHorizontalBlock"] {{
  background: var(--parchment-soft);
  border: 1px solid var(--brass-soft);
  border-radius: 26px;
  padding: 6px 16px;
  box-shadow: 0 4px 18px rgba(58,53,43,0.12);
  align-items: center !important;
}}
.kc-stopbar-text {{
  font-family: 'Hanuman', sans-serif;
  font-size: 15.5px;
  color: var(--stone);
  opacity: 0.55;
  padding: 10px 8px;
}}
.st-key-kc_stopbar [data-testid="stButton"] {{
  display: flex !important;
  justify-content: center !important;
}}
.st-key-kc_stopbar [data-testid="stButton"] button {{
  background: var(--lacquer) !important;
  border: none !important;
  width: 36px !important;
  height: 36px !important;
  min-width: 36px !important;
  border-radius: 9px !important;
  margin-left: auto !important;
  padding: 0 !important;
  position: relative !important;
  box-shadow: none !important;
}}
.st-key-kc_stopbar [data-testid="stButton"] button p {{ display: none !important; }}
.st-key-kc_stopbar [data-testid="stButton"] button::before {{
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 13px;
  height: 13px;
  background: var(--parchment-soft);
  border-radius: 3px;
}}
.st-key-kc_stopbar [data-testid="stButton"] button:hover {{
  background: var(--lacquer-deep) !important;
}}

@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


import streamlit.components.v1 as components

components.html(
    """
    <script>
    window.parent.document.addEventListener(
        'keydown',
        function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
                e.stopImmediatePropagation();
            }
        },
        true
    );
    </script>
    """,
    height=0,
)

# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------


def count_tokens(text):
    """Real token count via llama-server's native /tokenize endpoint (not an
    estimate -- character-based heuristics are especially wrong for Khmer).
    Returns None if the server doesn't support it / is unreachable."""
    try:
        resp = requests.post(
            f"{LLAMA_SERVER_ROOT}/tokenize", json={"content": text}, timeout=10
        )
        resp.raise_for_status()
        tokens = resp.json().get("tokens", [])
        return len(tokens)
    except Exception:
        return None


def get_loaded_model():
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["id"]
    except Exception:
        return "No Model Loaded"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_PATH, device="cpu")


@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)


@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_query(query, embedding_model):
    embed_start = time.time()
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )
    embed_time = time.time() - embed_start
    return query_embedding, embed_time


def search_faiss(query_embedding, index, chunks):
    search_start = time.time()
    scores, indices = index.search(
        query_embedding,
        k=TOP_K,
    )
    search_time = time.time() - search_start
    results = [(score, chunks[idx]) for score, idx in zip(scores[0], indices[0])]
    return results, search_time


def build_context(results):
    context_start = time.time()
    context_parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        context_parts.append(f"""
[Source {rank}]

Document Name:
{metadata.get("document_name", "N/A")}

Document Type:
{metadata.get("document_type_kh", metadata.get("document_type", "N/A"))}

Release Date:
{metadata.get("release_date", "N/A")}

Chapter:
{metadata.get("chapter_title", metadata.get("chapter", "N/A"))}

Section:
{metadata.get("section_title", metadata.get("section", "N/A"))}

Legal Unit:
{metadata.get("legal_unit", "N/A")}

Chunk ID:
{chunk["chunk_id"]}

Content:
{chunk["text"]}
""")
    context = "\n\n".join(context_parts)
    context_length = len(context)
    context_time = time.time() - context_start
    return context, context_length, context_time


def build_prompt(question, context):
    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.

Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Provide a complete and detailed answer. If the context lists multiple conditions, sub-points, exceptions, or requirements, enumerate ALL of them.
- If relevant information spans multiple sources, synthesize all of them into one coherent answer.
- If answer not found, reply:

រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។

Context:
{context}

Question:
{question}

Answer:
"""


def select_used_sources(results):
    if not results:
        return []
    top_score = results[0][0]
    selected = [r for r in results if (top_score - r[0]) <= USED_SOURCE_MARGIN]
    return selected[:MAX_USED_SOURCES]


def run_pipeline_in_thread(
    question, model_name, client, embedding_model, index, chunks, out_q, stop_event
):
    """Runs entirely in a background thread. Must never call any st.* API."""
    try:
        out_q.put(("phase", "encoding_query"))

        query_embedding, embed_time = encode_query(
            question,
            embedding_model,
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

        out_q.put(("phase", "building_context"))
        context, context_length, context_time = build_context(results)
        out_q.put(("phase", "constructing_prompt"))
        prompt = build_prompt(question, context)
        context_tokens = count_tokens(context)
        prompt_tokens = count_tokens(prompt)

        out_q.put(("phase", "generating_response"))

        gen_start = time.time()
        first_token_time = None
        total_chars = 0
        total_tokens_out = 0
        was_stopped = False

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            if stop_event.is_set():
                was_stopped = True
                break
            delta = chunk.choices[0].delta.content
            if delta:
                total_chars += len(delta)
                # llama-server streams one token per content delta, so this
                # count is a direct token count, not a character estimate.
                total_tokens_out += 1
                if first_token_time is None:
                    first_token_time = time.time() - gen_start
                out_q.put(("token", delta))

        try:
            response.close()
        except Exception:
            pass

        generation_time = time.time() - gen_start

        payload = {
            "results": results,
            "best_score": best_score,
            "min_score": min_score,
            "spread_score": spread_score,
            "embed_time": embed_time,
            "search_time": search_time,
            "retrieval_time": retrieval_time,
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
        out_q.put(("error", str(e)))


def render_ladder_html(current_idx, phase_elapsed):
    rows = []
    for i, key in enumerate(PHASES):
        label_html = PHASE_LABELS[key]
        elapsed = phase_elapsed.get(key)
        time_str = f"{elapsed:.1f}s" if elapsed is not None else ""
        if i < current_idx or (key == "completed" and i == current_idx):
            icon = '<span class="kc-step-icon kc-step-done">&#10003;</span>'
            row_cls = "kc-step-row"
        elif i == current_idx:
            icon = '<span class="kc-step-icon kc-step-active"></span>'
            row_cls = "kc-step-row"
        else:
            icon = '<span class="kc-step-icon kc-step-pending"></span>'
            row_cls = "kc-step-row kc-step-row-pending"
        rows.append(
            f'<div class="{row_cls}">'
            f"{icon}"
            f'<span class="kc-step-label">{label_html}</span>'
            f'<span class="kc-step-time">{time_str}</span>'
            f"</div>"
        )
    return f'<div class="kc-ladder">{"".join(rows)}</div>'


# ----------------------------------------------------------------------------
# Chat page rendering
# ----------------------------------------------------------------------------


def render_full_header(model_name):
    st.markdown(
        f'<div class="kc-welcome-wrap">'
        f'<div class="kc-eyebrow">RAG &middot; Insurance Law Corpus</div>'
        f'<div class="kc-welcome-lockup">'
        f"{SEAL_SVG}"
        f'<div class="kc-title-big">ជំនួយការច្បាប់ធានារ៉ាប់រង</div>'
        f"</div>"
        f'<div class="kc-subtitle">Khmer Insurance Law Assistant</div>'
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


def render_compact_topbar(model_name):
    st.markdown(
        f'<div class="kc-topbar">{SEAL_SVG}'
        f"<div>"
        f'<div class="kc-topbar-title">ជំនួយការច្បាប់ធានារ៉ាប់រង</div>'
        f'<div class="kc-topbar-sub">Khmer Insurance Law Assistant</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_stop_bar():
    """Replaces the native chat input while a question is being processed.
    Visually a square 'stop' button instead of the arrow. Returns True if
    the user clicked it on this run."""
    with st.container(key="kc_stopbar"):
        col_text, col_btn = st.columns([8, 1])
        with col_text:
            st.markdown(
                '<div class="kc-stopbar-text">កំពុងបង្កើតចម្លើយ...</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            clicked = st.button("stop", key="kc_stop_btn")
    return clicked


def render_used_sources(used_sources):
    chips = []
    for score, chunk in used_sources:
        metadata = chunk.get("metadata", {})
        name = html.escape(metadata.get("document_name", "Unknown Document"))
        cid = html.escape(chunk["chunk_id"])
        chips.append(
            f'<div class="kc-source-chip"><div class="name">{name}</div>'
            f'<div class="id">{cid}</div></div>'
        )
    st.markdown(
        f'<div class="kc-sources-block">{"".join(chips)}</div>', unsafe_allow_html=True
    )


def render_message(msg):
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(msg["question"])}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)
    if msg.get("low_confidence"):
        st.markdown(
            f'<div class="kc-alert">Low confidence retrieval &mdash; best match score '
            f'{msg["best_score"]:.3f}. Treat this answer with extra caution.</div>',
            unsafe_allow_html=True,
        )
    if msg.get("interrupted"):
        st.markdown(
            '<div class="kc-alert">This question was interrupted before completion.</div>',
            unsafe_allow_html=True,
        )
        if msg.get("ladder_html"):
            st.markdown(msg["ladder_html"], unsafe_allow_html=True)
    if msg["answer"]:
        st.markdown(
            f'<div class="kc-msg-assistant">{html.escape(msg["answer"])}</div>',
            unsafe_allow_html=True,
        )
    if not msg.get("interrupted"):
        render_used_sources(msg["used_sources"])
        st.markdown('<div class="kc-msg-actions">', unsafe_allow_html=True)
        if st.button("Details", key=f"diag_{msg['id']}"):
            st.session_state["view"] = "diagnostics"
            st.session_state["active_msg_id"] = msg["id"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def handle_new_question(question, model_name, client, embedding_model, index, chunks):
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(question)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)

    ladder_ph = st.empty()
    answer_ph = st.empty()

    out_q = queue.Queue()
    stop_event = threading.Event()
    st.session_state["stop_event"] = stop_event

    thread = threading.Thread(
        target=run_pipeline_in_thread,
        args=(
            question,
            model_name,
            client,
            embedding_model,
            index,
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
    final_payload = None
    error_text = None

    # Seeded immediately, then kept in sync below on every loop tick. If the
    # user clicks Stop, this whole function call (and its local variables)
    # gets torn down mid-flight by Streamlit before it ever sees the click --
    # so the *next* run has nothing to go on except whatever was last written
    # here to session_state.
    st.session_state["active_run"] = {
        "question": question,
        "answer_text": answer_text,
        "ladder_html": render_ladder_html(current_idx, phase_elapsed),
    }

    thread.start()
    finished = False
    while not finished:
        # If the user clicked stop on a *previous* run, this whole function
        # call (and its thread) never even started -- Streamlit kills the
        # in-flight script run the instant a new widget interaction comes
        # in. That's handled in render_chat_page before we get here. This
        # loop only watches the queue for the thread we just launched.
        try:
            kind, payload = out_q.get(timeout=0.12)
            now = time.time()
            if kind == "phase":
                phase_elapsed[PHASES[current_idx]] = (
                    now - phase_start_times[PHASES[current_idx]]
                )
                current_idx = PHASES.index(payload)
                phase_start_times[payload] = now
            elif kind == "token":
                answer_text += payload
                answer_ph.markdown(
                    f'<div class="kc-msg-assistant">{html.escape(answer_text)}</div>',
                    unsafe_allow_html=True,
                )
            elif kind == "done":
                phase_elapsed[PHASES[current_idx]] = (
                    now - phase_start_times[PHASES[current_idx]]
                )
                current_idx = PHASES.index("completed")
                phase_elapsed["completed"] = now - overall_start
                final_payload = payload
                finished = True
            elif kind == "error":
                error_text = payload
                finished = True
        except queue.Empty:
            pass

        if not finished:
            phase_elapsed[PHASES[current_idx]] = (
                time.time() - phase_start_times[PHASES[current_idx]]
            )
        ladder_html = render_ladder_html(current_idx, phase_elapsed)
        ladder_ph.markdown(ladder_html, unsafe_allow_html=True)

        # Mirror progress so a stop-induced rerun can recover it. Cheap
        # dict writes; happens at most ~every 0.12s (queue poll interval)
        # or immediately on each phase/token event.
        st.session_state["active_run"] = {
            "question": question,
            "answer_text": answer_text,
            "ladder_html": ladder_html,
        }

    time.sleep(0.4)
    ladder_ph.empty()

    if error_text:
        answer_ph.markdown(
            f'<div class="kc-alert">Something went wrong while generating the answer: '
            f"{html.escape(error_text)}</div>",
            unsafe_allow_html=True,
        )
        return

    results = final_payload["results"]
    best_score = final_payload["best_score"]
    used_sources = select_used_sources(results)

    render_used_sources(used_sources)

    st.session_state["active_run"] = None

    msg = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer_text,
        "used_sources": used_sources,
        "all_results": results,
        "best_score": best_score,
        "low_confidence": best_score < MIN_SCORE,
        "interrupted": False,
        "diag": {
            "model": model_name,
            "embedding_model": EMBEDDING_MODEL_PATH,
            "embedding_device": str(embedding_model.device),
            "top_k": TOP_K,
            "embed_time": final_payload["embed_time"],
            "search_time": final_payload["search_time"],
            "retrieval_time": final_payload["retrieval_time"],
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
    st.session_state["messages"].append(msg)


def render_chat_page(model_name, client, embedding_model, index, chunks):
    messages = st.session_state["messages"]
    pending = st.session_state["pending_question"]

    if not messages and not pending and not st.session_state.get("is_generating"):
        render_full_header(model_name)
        render_welcome_form()
        return

    render_compact_topbar(model_name)
    st.markdown('<div class="kc-thread">', unsafe_allow_html=True)
    for msg in messages:
        render_message(msg)
    st.markdown("</div>", unsafe_allow_html=True)

    # A question is about to start processing this run -- lock the input
    # bar down to the stop button before the (blocking) pipeline call below.
    if pending:
        st.session_state["is_generating"] = True

    if st.session_state["is_generating"]:
        stop_clicked = render_stop_bar()
        prompt = None
    else:
        stop_clicked = False
        # Called before the (blocking) pipeline run below so it's already
        # painted on screen -- and pinned at the bottom -- the instant a
        # question is sent.
        prompt = st.chat_input(placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ...")

    if stop_clicked:
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
            st.session_state["messages"].append(
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
        st.session_state["pending_question"] = None
        st.rerun()

    if pending:
        st.session_state["pending_question"] = None
        st.markdown('<div class="kc-thread">', unsafe_allow_html=True)
        handle_new_question(pending, model_name, client, embedding_model, index, chunks)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state["is_generating"] = False
        st.session_state["stop_event"] = None
        st.rerun()

    if prompt and prompt.strip():
        st.session_state["pending_question"] = prompt.strip()
        st.rerun()


# ----------------------------------------------------------------------------
# Diagnostics page rendering
# ----------------------------------------------------------------------------


def console_block(title, rows):
    row_html = "".join(
        f'<div class="kc-row"><span class="kc-row-label">{html.escape(str(k))}</span>'
        f'<span class="kc-row-value">{html.escape(str(v))}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="kc-console-block"><div class="kc-console-title">&gt; {html.escape(title)}</div>'
        f"{row_html}</div>",
        unsafe_allow_html=True,
    )


def render_diag_source_cards(results):
    cards = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = html.escape(metadata.get("document_name", "Unknown Document"))
        chunk_id = html.escape(chunk["chunk_id"])
        pct = max(0.0, min(1.0, float(score))) * 100
        delay = (rank - 1) * 0.05
        card = (
            f'<div class="kc-diag-src-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="kc-diag-src-top">'
            f'<span class="kc-diag-src-rank">{rank:02d}</span>'
            f'<div class="kc-diag-src-main">'
            f'<div class="kc-diag-src-name">{doc_name}</div>'
            f'<div class="kc-diag-src-id">{chunk_id}</div>'
            f"</div>"
            f'<span class="kc-diag-src-score">{score:.3f}</span>'
            f"</div>"
            f'<div class="kc-diag-src-bar"><div class="kc-diag-src-fill" '
            f'style="--w:{pct:.1f}%; animation-delay:{delay:.2f}s"></div></div>'
            f"</div>"
        )
        cards.append(card)
    st.markdown(
        f'<div class="kc-diag-src-list">{"".join(cards)}</div>', unsafe_allow_html=True
    )


def render_source_detail(rank, score, chunk):
    metadata = chunk.get("metadata", {})
    doc_name = metadata.get("document_name", "N/A")
    rows = [
        ("similarity", f"{float(score):.4f}"),
        ("document", doc_name),
        (
            "type",
            metadata.get("document_type_kh", metadata.get("document_type", "N/A")),
        ),
        ("release date", metadata.get("release_date", "N/A")),
        ("chapter", metadata.get("chapter_title", metadata.get("chapter", "N/A"))),
        ("section", metadata.get("section_title", metadata.get("section", "N/A"))),
        ("legal unit", metadata.get("legal_unit", "N/A")),
        ("chunk id", chunk["chunk_id"]),
    ]
    grid = "".join(
        f'<div class="k">{html.escape(str(k))}</div><div class="v">{html.escape(str(v))}</div>'
        for k, v in rows
    )
    text = html.escape(chunk["text"])
    with st.expander(f"Rank {rank} | {doc_name} | Score {score:.4f}"):
        st.markdown(
            f'<div class="kc-detail-grid">{grid}</div><div class="kc-detail-text">{text}</div>',
            unsafe_allow_html=True,
        )


def render_diagnostics_page():
    msg = next(
        (
            m
            for m in st.session_state["messages"]
            if m["id"] == st.session_state["active_msg_id"]
        ),
        None,
    )

    if st.button("← Back to chat", key="back_to_chat"):
        st.session_state["view"] = "chat"
        st.session_state["active_msg_id"] = None
        st.rerun()

    if msg is None:
        st.markdown(
            '<div class="kc-diag-title">No diagnostics found for this message.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="kc-diag-header"><div class="kc-diag-title">DIAGNOSTICS</div>'
        f'<div class="kc-diag-question">{html.escape(msg["question"])}</div></div>',
        unsafe_allow_html=True,
    )

    d = msg["diag"]

    def tok(value):
        return f"{value:,}" if value is not None else "N/A"

    col1, col2 = st.columns([1, 2.4])
    with col1:
        console_block(
            "SYSTEM",
            [
                ("generator model", d["model"]),
                ("generator endpoint", LLAMA_SERVER_URL),
                ("retriever model", d["embedding_model"]),
                ("retriever device", d["embedding_device"]),
                ("top k", d["top_k"]),
            ],
        )
        console_block(
            "RETRIEVAL",
            [
                ("embed time", f'{d["embed_time"]:.3f}s'),
                ("faiss time", f'{d["search_time"]:.3f}s'),
                ("retrieval time", f'{d["retrieval_time"]:.3f}s'),
                ("best score", f'{msg["best_score"]:.4f}'),
                ("min score", f'{d["min_score"]:.4f}'),
                ("spread score", f'{d["spread_score"]:.4f}'),
            ],
        )
        console_block(
            "CONTEXT",
            [
                ("context length", f'{d["context_length"]} chars'),
                ("context token", tok(d["context_tokens"])),
                ("context build time", f'{d["context_time"]:.3f}s'),
                ("prompt length", f'{d["prompt_length"]} chars'),
                ("prompt token", tok(d["prompt_tokens"])),
            ],
        )
        console_block(
            "GENERATION",
            [
                ("first token", f'{d["first_token_time"]:.3f}s'),
                ("generation time", f'{d["generation_time"]:.3f}s'),
                ("total time", f'{d["total_time"]:.3f}s'),
                ("output token", tok(d["generated_tokens"])),
                ("generated chars", d["generated_chars"]),
            ],
        )
    with col2:
        st.markdown(
            f'<div class="kc-console-title">&gt; CHUNKS (TOP {len(msg["all_results"])})</div>',
            unsafe_allow_html=True,
        )
        render_diag_source_cards(msg["all_results"])

    st.markdown(
        '<div class="kc-console-title" style="margin-top:18px">&gt; FULL SOURCE DETAILS</div>',
        unsafe_allow_html=True,
    )
    for rank, (score, chunk) in enumerate(msg["all_results"], start=1):
        render_source_detail(rank, score, chunk)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

inject_css()

MODEL_NAME = get_loaded_model()

if MODEL_NAME == "No Model Loaded":
    st.markdown(
        '<div class="kc-alert">llama-server is not running. Start it before using the assistant.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()

if st.session_state["view"] == "diagnostics":
    render_diagnostics_page()
else:
    render_chat_page(MODEL_NAME, client, embedding_model, index, chunks)
