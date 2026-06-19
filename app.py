import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss

faiss.omp_set_num_threads(1)

import html
import json
import time

import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LLAMA_SERVER_URL = "http://localhost:8080/v1"

TOP_K = 10
MIN_SCORE = 0.50

FAISS_PATH = "vector_db/khmer_insurance.index"
METADATA_PATH = "vector_db/khmer_insurance_metadata.json"
EMBEDDING_MODEL_PATH = "models/retriever/BAAI-BGE-M3"

SEAL_SVG = """
<svg width="42" height="42" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
<circle cx="20" cy="20" r="18" fill="none" stroke="#B68B4C" stroke-width="1.1"/>
<circle cx="20" cy="20" r="12" fill="none" stroke="#B68B4C" stroke-width="1.1"/>
<g stroke="#B68B4C" stroke-width="1.1">
<line x1="20" y1="2" x2="20" y2="8"/>
<line x1="20" y1="32" x2="20" y2="38"/>
<line x1="2" y1="20" x2="8" y2="20"/>
<line x1="32" y1="20" x2="38" y2="20"/>
<line x1="7.5" y1="7.5" x2="11.5" y2="11.5"/>
<line x1="28.5" y1="28.5" x2="32.5" y2="32.5"/>
<line x1="7.5" y1="32.5" x2="11.5" y2="28.5"/>
<line x1="28.5" y1="11.5" x2="32.5" y2="7.5"/>
</g>
<circle cx="20" cy="20" r="3" fill="#B68B4C"/>
</svg>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink: #15201C;
  --ink-soft: #1D2B25;
  --ink-line: #2C3D34;
  --parchment: #EDE6D6;
  --parchment-soft: #F5F0E4;
  --lacquer: #9C2B1F;
  --lacquer-deep: #7A2118;
  --brass: #B68B4C;
  --brass-soft: #D8BD8B;
  --stone: #3A352B;
  --mist: #9DB6AB;
  --signal: #6FBF8B;
  --danger: #C0524A;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--parchment) !important;
  color: var(--stone);
  font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 8px; }

/* ---------- Header ---------- */
.kc-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 4px 22px 4px;
  border-bottom: 1px solid var(--brass-soft);
  margin-bottom: 8px;
}
.kc-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--brass);
}
.kc-title {
  font-family: 'Moul', serif;
  font-size: 28px;
  color: var(--lacquer);
  margin: 3px 0 2px 0;
  line-height: 1.35;
}
.kc-subtitle {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 12.5px;
  color: var(--stone);
  opacity: 0.7;
}
.kc-status-pill {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--ink);
  color: var(--parchment-soft);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  padding: 7px 16px;
  border-radius: 999px;
  white-space: nowrap;
}
.kc-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.kc-dot-on { background: var(--signal); box-shadow: 0 0 0 3px rgba(111,191,139,0.25); }
.kc-dot-off { background: var(--danger); box-shadow: 0 0 0 3px rgba(192,82,74,0.25); }

/* ---------- Section labels ---------- */
.kc-section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--brass);
  margin: 22px 0 8px 0;
}

/* ---------- Left column: paper cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--parchment-soft) !important;
  border: 1px solid var(--brass-soft) !important;
  border-radius: 4px !important;
  animation: kc-rise 0.45s ease;
}
[data-testid="stVerticalBlockBorderWrapper"] p {
  font-family: 'Hanuman', sans-serif;
  font-size: 16.5px;
  line-height: 1.9;
  color: var(--stone);
}

[data-testid="stTextArea"] textarea {
  background: var(--parchment-soft);
  border: 1px solid var(--brass-soft);
  border-radius: 4px;
  font-family: 'Hanuman', sans-serif;
  font-size: 16px;
  color: var(--stone);
  padding: 14px;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--lacquer);
  box-shadow: 0 0 0 1px var(--lacquer);
}
[data-testid="stTextArea"] label p {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13px !important;
  color: var(--stone) !important;
}

[data-testid="stButton"] button {
  background: var(--lacquer);
  color: var(--parchment-soft);
  border: none;
  border-radius: 3px;
  padding: 10px 30px;
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  letter-spacing: 0.02em;
  transition: background 0.15s ease, transform 0.15s ease;
}
[data-testid="stButton"] button:hover {
  background: var(--lacquer-deep);
  transform: translateY(-1px);
}
[data-testid="stButton"] button:focus-visible {
  outline: 2px solid var(--brass);
  outline-offset: 2px;
}

[data-testid="stExpander"] {
  background: var(--parchment-soft);
  border: 1px solid var(--brass-soft) !important;
  border-radius: 4px !important;
  margin-bottom: 8px;
}
[data-testid="stExpander"] summary {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--stone);
  font-weight: 600;
  font-size: 13.5px;
}
[data-testid="stStatusWidget"] {
  background: var(--parchment-soft);
  border: 1px solid var(--brass-soft);
  border-radius: 4px;
}

[data-testid="stAlert"] { border-radius: 4px; font-family: 'IBM Plex Sans', sans-serif; }

.kc-alert {
  background: rgba(156,43,31,0.08);
  border: 1px solid var(--lacquer);
  color: var(--lacquer-deep);
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 13px;
  padding: 10px 14px;
  border-radius: 4px;
  margin: 6px 0 16px 0;
}

/* ---------- Source cards (the signature element) ---------- */
.kc-source-list { display: flex; flex-direction: column; gap: 10px; margin: 4px 0 16px 0; }
.kc-source-card {
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--parchment-soft);
  border: 1px solid var(--brass-soft);
  border-radius: 4px;
  padding: 12px 16px;
  opacity: 0;
  animation: kc-rise 0.5s ease forwards;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.kc-source-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(58,53,43,0.14);
  border-color: var(--lacquer);
}
.kc-source-rank {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  color: var(--brass);
  width: 22px;
  flex-shrink: 0;
}
.kc-source-main { flex: 1; min-width: 0; }
.kc-source-name {
  font-family: 'Hanuman', sans-serif;
  font-size: 14.5px;
  color: var(--stone);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kc-source-meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  color: var(--brass);
  margin-top: 3px;
}
.kc-bar {
  width: 100%;
  height: 6px;
  background: rgba(182,139,76,0.18);
  border-radius: 3px;
  overflow: hidden;
  margin-top: 7px;
}
.kc-bar-fill {
  height: 100%;
  width: 0;
  background: linear-gradient(90deg, var(--brass), var(--lacquer));
  border-radius: 3px;
  animation: kc-fill 0.9s cubic-bezier(.2,.8,.2,1) forwards;
}
.kc-source-score {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  color: var(--lacquer);
  font-weight: 600;
  width: 50px;
  text-align: right;
  flex-shrink: 0;
}

.kc-detail-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 14px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  margin-bottom: 12px;
}
.kc-detail-grid .k { color: var(--brass); }
.kc-detail-grid .v { color: var(--stone); }
.kc-detail-text {
  font-family: 'Hanuman', sans-serif;
  font-size: 15px;
  line-height: 1.85;
  color: var(--stone);
  border-top: 1px dashed var(--brass-soft);
  padding-top: 12px;
}

.kc-footnote {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 11.5px;
  color: var(--stone);
  opacity: 0.65;
  margin-top: 26px;
  padding-top: 14px;
  border-top: 1px dashed var(--brass-soft);
  line-height: 1.6;
}

/* ---------- Right column: engine console ---------- */
[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {
  background: var(--ink);
  border-radius: 6px;
  padding: 18px 18px 24px 18px;
}
[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) * {
  color: var(--mist);
}

.kc-console-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--brass);
  border-bottom: 1px solid var(--ink-line);
  padding-bottom: 10px;
  margin-bottom: 14px;
}
.kc-console-block {
  background: var(--ink-soft);
  border: 1px solid var(--ink-line);
  border-radius: 4px;
  padding: 12px 14px;
  margin-bottom: 12px;
}
.kc-console-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.08em;
  color: var(--brass);
  text-transform: uppercase;
  margin-bottom: 8px;
}
.kc-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  padding: 3px 0;
  border-bottom: 1px dashed rgba(157,182,171,0.15);
}
.kc-row:last-child { border-bottom: none; }
.kc-row-label { color: var(--mist); opacity: 0.85; }
.kc-row-value { color: var(--parchment-soft); text-align: right; }

.kc-chunk-row {
  display: grid;
  grid-template-columns: 24px 50px 1fr 40px;
  gap: 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  padding: 5px 0;
  border-bottom: 1px dashed rgba(157,182,171,0.12);
  align-items: center;
}
.kc-chunk-row:last-child { border-bottom: none; }
.kc-chunk-rank { color: var(--brass); }
.kc-chunk-score { color: var(--signal); }
.kc-chunk-id { color: var(--parchment-soft); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kc-chunk-len { color: var(--mist); opacity: 0.7; text-align: right; }

@keyframes kc-rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes kc-fill {
  from { width: 0; }
  to { width: var(--w, 0%); }
}
@media (prefers-reduced-motion: reduce) {
  .kc-source-card { animation: none !important; opacity: 1 !important; }
  .kc-bar-fill { animation: none !important; width: var(--w, 0%) !important; }
  [data-testid="stVerticalBlockBorderWrapper"] { animation: none !important; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def render_header(model_name: str):
    status_ok = model_name != "No Model Loaded"
    dot_class = "kc-dot-on" if status_ok else "kc-dot-off"
    st.markdown(
        f"""
        <div class="kc-header">
          <div>{SEAL_SVG}</div>
          <div>
            <div class="kc-eyebrow">RAG &middot; Insurance Law Corpus</div>
            <h1 class="kc-title">ជំនួយការច្បាប់ធានារ៉ាប់រង</h1>
            <div class="kc-subtitle">Khmer Legal Research Assistant</div>
          </div>
          <div class="kc-status-pill">
            <span class="kc-dot {dot_class}"></span>{html.escape(model_name)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="kc-section-label">{text}</div>', unsafe_allow_html=True)


def render_source_cards(results):
    cards = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = html.escape(metadata.get("document_name", "Unknown Document"))
        chunk_id = html.escape(chunk["chunk_id"])
        pct = max(0.0, min(1.0, float(score))) * 100
        delay = (rank - 1) * 0.05
        card = (
            f'<div class="kc-source-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="kc-source-rank">{rank:02d}</div>'
            f'<div class="kc-source-main">'
            f'<div class="kc-source-name">{doc_name}</div>'
            f'<div class="kc-source-meta">{chunk_id}</div>'
            f'<div class="kc-bar"><div class="kc-bar-fill" '
            f'style="--w:{pct:.1f}%; animation-delay:{delay:.2f}s"></div></div>'
            f"</div>"
            f'<div class="kc-source-score">{score:.3f}</div>'
            f"</div>"
        )
        cards.append(card)
    st.markdown(
        f'<div class="kc-source-list">{"".join(cards)}</div>', unsafe_allow_html=True
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
            f'<div class="kc-detail-grid">{grid}</div>'
            f'<div class="kc-detail-text">{text}</div>',
            unsafe_allow_html=True,
        )


def console_block(title: str, rows):
    row_html = "".join(
        f'<div class="kc-row"><span class="kc-row-label">{html.escape(str(k))}</span>'
        f'<span class="kc-row-value">{html.escape(str(v))}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f"""
        <div class="kc-console-block">
          <div class="kc-console-title">&gt; {html.escape(title)}</div>
          {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chunks_console(results):
    rows = []
    for rank, (score, chunk) in enumerate(results, start=1):
        chunk_id = html.escape(chunk["chunk_id"])
        row = (
            f'<div class="kc-chunk-row">'
            f'<span class="kc-chunk-rank">{rank:02d}</span>'
            f'<span class="kc-chunk-score">{float(score):.3f}</span>'
            f'<span class="kc-chunk-id">{chunk_id}</span>'
            f'<span class="kc-chunk-len">{len(chunk["text"])}</span>'
            f"</div>"
        )
        rows.append(row)
    st.markdown(
        f'<div class="kc-console-block"><div class="kc-console-title">&gt; CHUNKS</div>'
        f'{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


# Get the model that user input
def get_loaded_model():
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["id"]
    except Exception:
        return "No Model Loaded"


st.set_page_config(page_title="Khmer Legal Chatbot", layout="wide")
inject_css()

MODEL_NAME = get_loaded_model()

if MODEL_NAME == "No Model Loaded":
    render_header(MODEL_NAME)
    st.markdown(
        '<div class="kc-alert">llama-server is not running. Start it before using the assistant.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

render_header(MODEL_NAME)

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_PATH, device="cpu")


@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)


@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


embedding_model = load_embedding_model()
index = load_index()
chunks = load_metadata()

if "has_result" not in st.session_state:
    st.session_state["has_result"] = False


def retrieve(query):
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
    return context, context_time, context_length


def stream_answer(question, context, diagnostics):
    prompt = f"""
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
    diagnostics["prompt_length"] = len(prompt)
    start = time.time()
    first_token = True

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )

    total_chars = 0
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            total_chars += len(delta)
            if first_token:
                diagnostics["first_token_time"] = time.time() - start
                first_token = False
            yield delta

    diagnostics["generated_chars"] = total_chars
    diagnostics["generation_time"] = time.time() - start


left_col, right_col = st.columns([2, 1])

with left_col:
    section_label("សំណួររបស់អ្នក &middot; YOUR QUESTION")
    with st.container(border=True):
        question = st.text_area(
            "Ask a Khmer legal question:",
            height=120,
            label_visibility="collapsed",
            placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ...",
        )
        ask_clicked = st.button("Ask")

    if ask_clicked and question.strip():
        status = st.status("Processing...", expanded=True)
        total_start = time.time()

        with st.spinner("Searching..."):
            retrieval_start = time.time()
            status.write("Searching vector database...")
            results, embed_time, search_time = retrieve(question)

            if not results:
                st.error("រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។")
                st.stop()

            best_score = results[0][0]
            if best_score < MIN_SCORE:
                st.markdown(
                    f'<div class="kc-alert">Low confidence retrieval &mdash; best match score {best_score:.3f}. '
                    f"Treat this answer with extra caution.</div>",
                    unsafe_allow_html=True,
                )

            retrieval_time = time.time() - retrieval_start

            status.write("Building context...")
            context, context_time, context_length = build_context(results)

            generation_start = time.time()
            status.write("Generating answer...")

            section_label("ចម្លើយ &middot; ANSWER")
            diagnostics = {}
            with st.container(border=True):
                st.write_stream(stream_answer(question, context, diagnostics))

            status.update(label="Completed", state="complete")

            generation_time = time.time() - generation_start
            total_time = time.time() - total_start
            st.session_state["diagnostics"] = {
                "embed_time": embed_time,
                "search_time": search_time,
                "retrieval_time": retrieval_time,
                "best_score": float(best_score),
                "context_length": context_length,
                "context_time": context_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "results": results,
                "diagnostics": diagnostics,
            }
            st.session_state["has_result"] = True

        section_label("ប្រភព &middot; SOURCES")
        render_source_cards(results)

        with st.expander("Show full source details"):
            for rank, (score, chunk) in enumerate(results, start=1):
                render_source_detail(rank, score, chunk)

    st.markdown(
        '<div class="kc-footnote">Answers are generated by AI from the retrieved legal text and may be '
        "incomplete. Always verify against the original law, sub-decree, or prakas before relying on this "
        "for formal decisions.</div>",
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        '<div class="kc-console-label">DIAGNOSTICS</div>', unsafe_allow_html=True
    )

    console_block(
        "SYSTEM",
        [
            ("model", MODEL_NAME),
            ("embedding model", EMBEDDING_MODEL_PATH),
            ("embedding device", str(embedding_model.device)),
            ("top k", TOP_K),
        ],
    )

    if st.session_state.get("has_result", False):
        diag = st.session_state["diagnostics"]

        console_block(
            "RETRIEVAL",
            [
                ("embed time", f'{diag["embed_time"]:.3f}s'),
                ("faiss time", f'{diag["search_time"]:.3f}s'),
                ("retrieval time", f'{diag["retrieval_time"]:.3f}s'),
                ("best score", f'{diag["best_score"]:.4f}'),
            ],
        )

        console_block(
            "CONTEXT",
            [
                ("context length", f'{diag["context_length"]} chars'),
                ("build time", f'{diag["context_time"]:.3f}s'),
                (
                    "prompt length",
                    f'{diag["diagnostics"].get("prompt_length", 0)} chars',
                ),
            ],
        )

        console_block(
            "GENERATION",
            [
                (
                    "first token",
                    f'{diag["diagnostics"].get("first_token_time", 0):.3f}s',
                ),
                (
                    "generation time",
                    f'{diag["diagnostics"].get("generation_time", 0):.3f}s',
                ),
                ("total time", f'{diag["total_time"]:.3f}s'),
                ("generated chars", diag["diagnostics"].get("generated_chars", 0)),
            ],
        )

        render_chunks_console(diag["results"])
    else:
        console_block("STATUS", [("state", "awaiting question")])
