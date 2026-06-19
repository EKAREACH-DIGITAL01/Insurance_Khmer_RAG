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

TOP_K = 10
MIN_SCORE = 0.50
USED_SOURCE_MARGIN = (
    0.03  # how close a score must be to the top score to also count as "used"
)
MAX_USED_SOURCES = 3

FAISS_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EMBEDDING_MODEL_PATH = "models/retriever/BAAI-BGE-M3"

PHASES = ["processing", "searching", "generating", "done"]
PHASE_LABELS = {
    "processing": "Processing",
    "searching": "Searching",
    "generating": "Generating",
    "done": "Done",
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
[data-testid="stButton"] button:hover {{ color: var(--lacquer) !important; text-decoration: underline; }}
.kc-diag-page [data-testid="stButton"] button {{ color: var(--brass) !important; }}
.kc-diag-page [data-testid="stButton"] button:hover {{ color: var(--signal) !important; }}

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
  background: var(--ink-soft); border: 1px solid var(--ink-line);
  border-radius: 4px; padding: 12px 14px; margin-bottom: 14px;
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

.kc-diag-page [data-testid="stExpander"] {{
  background: var(--ink-soft); border: 1px solid var(--ink-line) !important; border-radius: 4px !important;
}}
.kc-diag-page [data-testid="stExpander"] summary {{
  font-family: 'IBM Plex Mono', monospace; color: var(--parchment-soft); font-size: 12px;
}}
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


def retrieve(query, embedding_model, index, chunks):
    embed_start = time.time()
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    embed_time = time.time() - embed_start

    search_start = time.time()
    scores, indices = index.search(query_embedding, k=TOP_K)
    search_time = time.time() - search_start

    results = [(score, chunks[idx]) for score, idx in zip(scores[0], indices[0])]
    return results, embed_time, search_time


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
    question, model_name, client, embedding_model, index, chunks, out_q
):
    """Runs entirely in a background thread. Must never call any st.* API."""
    try:
        results, embed_time, search_time = retrieve(
            question, embedding_model, index, chunks
        )
        best_score = float(results[0][0]) if results else 0.0
        retrieval_time = embed_time + search_time

        out_q.put(("phase", "searching"))
        context, context_length, context_time = build_context(results)
        prompt = build_prompt(question, context)

        out_q.put(("phase", "generating"))
        gen_start = time.time()
        first_token_time = None
        total_chars = 0

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                total_chars += len(delta)
                if first_token_time is None:
                    first_token_time = time.time() - gen_start
                out_q.put(("token", delta))

        generation_time = time.time() - gen_start

        payload = {
            "results": results,
            "best_score": best_score,
            "embed_time": embed_time,
            "search_time": search_time,
            "retrieval_time": retrieval_time,
            "context_length": context_length,
            "context_time": context_time,
            "prompt_length": len(prompt),
            "first_token_time": first_token_time or 0.0,
            "generation_time": generation_time,
            "generated_chars": total_chars,
        }
        out_q.put(("done", payload))
    except Exception as e:
        out_q.put(("error", str(e)))


def render_ladder_html(current_idx, phase_elapsed):
    rows = []
    for i, key in enumerate(PHASES):
        label = PHASE_LABELS[key]
        elapsed = phase_elapsed.get(key)
        time_str = f"{elapsed:.1f}s" if elapsed is not None else ""
        if i < current_idx:
            icon = '<span class="kc-step-icon kc-step-done">&#10003;</span>'
            row_cls = "kc-step-row"
        elif i == current_idx:
            icon = '<span class="kc-step-icon kc-step-active"></span>'
            row_cls = "kc-step-row"
        else:
            icon = '<span class="kc-step-icon kc-step-pending"></span>'
            row_cls = "kc-step-row kc-step-row-pending"
        rows.append(
            f'<div class="{row_cls}">{icon}<span class="kc-step-label">{label}</span>'
            f'<span class="kc-step-time">{time_str}</span></div>'
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
    st.markdown(
        f'<div class="kc-msg-assistant">{html.escape(msg["answer"])}</div>',
        unsafe_allow_html=True,
    )
    render_used_sources(msg["used_sources"])
    st.markdown('<div class="kc-msg-actions">', unsafe_allow_html=True)
    if st.button("Details", key=f"diag_{msg['id']}"):
        st.session_state["view"] = "diagnostics"
        st.session_state["active_msg_id"] = msg["id"]
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def handle_new_question(question, model_name, client, embedding_model, index, chunks):
    st.markdown(
        f'<div class="kc-msg-user-row"><div class="kc-msg-user">{html.escape(question)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kc-msg-assistant-row">', unsafe_allow_html=True)

    ladder_ph = st.empty()
    answer_ph = st.empty()

    out_q = queue.Queue()
    thread = threading.Thread(
        target=run_pipeline_in_thread,
        args=(question, model_name, client, embedding_model, index, chunks, out_q),
        daemon=True,
    )

    overall_start = time.time()
    phase_start_times = {"processing": overall_start}
    phase_elapsed = {}
    current_idx = 0
    answer_text = ""
    final_payload = None
    error_text = None

    thread.start()
    finished = False
    while not finished:
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
                current_idx = PHASES.index("done")
                phase_elapsed["done"] = now - overall_start
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
        ladder_ph.markdown(
            render_ladder_html(current_idx, phase_elapsed), unsafe_allow_html=True
        )

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

    msg = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer_text,
        "used_sources": used_sources,
        "all_results": results,
        "best_score": best_score,
        "low_confidence": best_score < MIN_SCORE,
        "diag": {
            "model": model_name,
            "embedding_model": EMBEDDING_MODEL_PATH,
            "embedding_device": str(embedding_model.device),
            "top_k": TOP_K,
            "embed_time": final_payload["embed_time"],
            "search_time": final_payload["search_time"],
            "retrieval_time": final_payload["retrieval_time"],
            "context_length": final_payload["context_length"],
            "context_time": final_payload["context_time"],
            "prompt_length": final_payload["prompt_length"],
            "first_token_time": final_payload["first_token_time"],
            "generation_time": final_payload["generation_time"],
            "total_time": time.time() - overall_start,
            "generated_chars": final_payload["generated_chars"],
        },
    }
    st.session_state["messages"].append(msg)


def render_chat_page(model_name, client, embedding_model, index, chunks):
    messages = st.session_state["messages"]
    pending = st.session_state["pending_question"]

    if not messages and not pending:
        render_full_header(model_name)
        render_welcome_form()
        return

    render_compact_topbar(model_name)
    st.markdown('<div class="kc-thread">', unsafe_allow_html=True)
    for msg in messages:
        render_message(msg)
    st.markdown("</div>", unsafe_allow_html=True)

    # Called before the (blocking) pipeline run below so it's already painted
    # on screen -- and pinned at the bottom -- the instant a question is sent.
    prompt = st.chat_input(placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ...")

    if pending:
        st.session_state["pending_question"] = None
        st.markdown('<div class="kc-thread">', unsafe_allow_html=True)
        handle_new_question(pending, model_name, client, embedding_model, index, chunks)
        st.markdown("</div>", unsafe_allow_html=True)
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


def render_chunks_console(results):
    rows = []
    for rank, (score, chunk) in enumerate(results, start=1):
        chunk_id = html.escape(chunk["chunk_id"])
        rows.append(
            f'<div class="kc-chunk-row">'
            f'<span class="kc-chunk-rank">{rank:02d}</span>'
            f'<span class="kc-chunk-score">{float(score):.3f}</span>'
            f'<span class="kc-chunk-id">{chunk_id}</span>'
            f'<span class="kc-chunk-len">{len(chunk["text"])}</span>'
            f"</div>"
        )
    st.markdown(
        f'<div class="kc-console-block"><div class="kc-console-title">&gt; CHUNKS (TOP {len(results)})</div>'
        f'{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def render_source_detail(rank, score, chunk):
    metadata = chunk.get("metadata", {})
    rows = [
        ("similarity", f"{float(score):.4f}"),
        ("document", metadata.get("document_name", "N/A")),
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
    with st.expander(f"Rank {rank} | Score {score:.4f}"):
        st.markdown(
            f'<div class="kc-detail-grid">{grid}</div><div class="kc-detail-text">{text}</div>',
            unsafe_allow_html=True,
        )


def render_diagnostics_page():
    st.markdown('<div class="kc-diag-page">', unsafe_allow_html=True)

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
            '<div class="kc-diag-title">No diagnostics found for this message.</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="kc-diag-header"><div class="kc-diag-title">DIAGNOSTICS</div>'
        f'<div class="kc-diag-question">{html.escape(msg["question"])}</div></div>',
        unsafe_allow_html=True,
    )

    d = msg["diag"]
    col1, col2 = st.columns(2)
    with col1:
        console_block(
            "SYSTEM",
            [
                ("model", d["model"]),
                ("embedding model", d["embedding_model"]),
                ("embedding device", d["embedding_device"]),
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
            ],
        )
    with col2:
        console_block(
            "CONTEXT",
            [
                ("context length", f'{d["context_length"]} chars'),
                ("build time", f'{d["context_time"]:.3f}s'),
                ("prompt length", f'{d["prompt_length"]} chars'),
            ],
        )
        console_block(
            "GENERATION",
            [
                ("first token", f'{d["first_token_time"]:.3f}s'),
                ("generation time", f'{d["generation_time"]:.3f}s'),
                ("total time", f'{d["total_time"]:.3f}s'),
                ("generated chars", d["generated_chars"]),
            ],
        )

    render_chunks_console(msg["all_results"])

    st.markdown(
        '<div class="kc-console-title" style="margin-top:18px">&gt; FULL SOURCE DETAILS</div>',
        unsafe_allow_html=True,
    )
    for rank, (score, chunk) in enumerate(msg["all_results"], start=1):
        render_source_detail(rank, score, chunk)

    st.markdown("</div>", unsafe_allow_html=True)


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
