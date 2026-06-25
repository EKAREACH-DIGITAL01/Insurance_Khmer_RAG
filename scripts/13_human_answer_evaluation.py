"""
Khmer Insurance RAG — Human Answer Evaluation UI
-------------------------------------------------
Launch:  python scripts/13_human_answer_evaluation.py
         (runs Streamlit on port 8502, separate from the chatbot on 8501)
"""

import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Launch behaviour
# ----------------------------------------------------------------------------
_LAUNCHED_KEY = "_EVAL_UI_LAUNCHED"

if os.environ.get(_LAUNCHED_KEY) != "1":
    env = {**os.environ, _LAUNCHED_KEY: "1"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).resolve()),
            "--server.port", "8502",
            "--server.headless", "true",
            "--client.toolbarMode", "minimal",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
    )
    import webbrowser
    time.sleep(2.5)
    webbrowser.open("http://localhost:8502")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    sys.exit(0)

import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DB_PATH = Path("reports/human_answer_evaluation/human_answer_evaluation.db")
JSONL_PATH = Path(
    "reports/generator_model_comparison/"
    "SeaLLMs-v3-7B-Chat-Q4_K_M/"
    "generated_answer_results.jsonl"
)
QUESTIONS_PER_EVALUATOR = None  # all questions, in order

METRICS = [
    {
        "key": "context_relevance",
        "label": "Context Relevance",
        "desc": "Did the retrieved context contain information relevant to the question?",
    },
    {
        "key": "faithfulness",
        "label": "Faithfulness",
        "desc": (
            "Is the answer fully supported by the retrieved sources, "
            "without introducing unsupported information?"
        ),
    },
    {
        "key": "answer_relevance",
        "label": "Answer Relevance",
        "desc": "Does the answer actually address what was asked?",
    },
    {
        "key": "answer_correctness",
        "label": "Answer Correctness",
        "desc": "How accurately does the answer match the expected answer?",
    },
    {
        "key": "completeness",
        "label": "Completeness",
        "desc": "Is the answer complete and covering all required points without being cut off?",
    },
]

HALLUCINATION_KEY = "hallucination"
CHIP_OPTS  = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
CHIP_REVERSE = {label: i + 1 for i, label in enumerate(CHIP_OPTS)}
HALLUC_OPTS  = ["Yes", "No"]
HALLUC_MAP   = {"Yes": 1, "No": 0}
MAX_TOTAL    = len(METRICS) * 5   # 25 — hallucination stored separately

NAME_COL_WIDTH = 1.6   # shared ratio for name cell vs radio cell

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluators (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assignments (
            evaluator_id INTEGER NOT NULL,
            question_idx INTEGER NOT NULL,
            position     INTEGER NOT NULL,
            PRIMARY KEY (evaluator_id, question_idx),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
        CREATE TABLE IF NOT EXISTS responses (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id       INTEGER NOT NULL,
            question_idx       INTEGER NOT NULL,
            context_relevance  INTEGER,
            faithfulness       INTEGER,
            answer_relevance   INTEGER,
            answer_correctness INTEGER,
            completeness       INTEGER,
            hallucination      INTEGER,
            total_score        INTEGER,
            submitted_at       INTEGER NOT NULL,
            UNIQUE (evaluator_id, question_idx),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
    """)
    resp_cols = {row[1] for row in conn.execute("PRAGMA table_info(responses)").fetchall()}
    for col in ("completeness", "hallucination"):
        if col not in resp_cols:
            conn.execute(f"ALTER TABLE responses ADD COLUMN {col} INTEGER")
    eval_cols = {row[1] for row in conn.execute("PRAGMA table_info(evaluators)").fetchall()}
    if "session_id" not in eval_cols:
        conn.execute("ALTER TABLE evaluators ADD COLUMN session_id TEXT")
    conn.commit()
    conn.close()


def create_new_session(name):
    conn = get_db()
    session_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    cur = conn.execute(
        "INSERT INTO evaluators (name, session_id, created_at) VALUES (?, ?, ?)",
        (name, session_id, int(time.time())),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid, session_id


def get_or_create_assignment(evaluator_id, all_questions):
    conn = get_db()
    rows = conn.execute(
        "SELECT question_idx, position FROM assignments WHERE evaluator_id = ? ORDER BY position",
        (evaluator_id,),
    ).fetchall()
    if rows:
        assignment = [r["question_idx"] for r in rows]
    else:
        assignment = list(range(len(all_questions)))
        for pos, qidx in enumerate(assignment):
            conn.execute(
                "INSERT INTO assignments (evaluator_id, question_idx, position) VALUES (?, ?, ?)",
                (evaluator_id, qidx, pos),
            )
        conn.commit()
    conn.close()
    return assignment


def get_responses(evaluator_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT question_idx, context_relevance, faithfulness, answer_relevance, "
        "answer_correctness, completeness, hallucination, total_score "
        "FROM responses WHERE evaluator_id = ?", (evaluator_id,),
    ).fetchall()
    conn.close()
    return {r["question_idx"]: dict(r) for r in rows}


def save_response(evaluator_id, question_idx, scores, hallucination_flag):
    total = sum(scores.values())
    conn  = get_db()
    conn.execute(
        """INSERT INTO responses
           (evaluator_id, question_idx, context_relevance, faithfulness, answer_relevance,
            answer_correctness, completeness, hallucination, total_score, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(evaluator_id, question_idx) DO UPDATE SET
               context_relevance=excluded.context_relevance,
               faithfulness=excluded.faithfulness,
               answer_relevance=excluded.answer_relevance,
               answer_correctness=excluded.answer_correctness,
               completeness=excluded.completeness,
               hallucination=excluded.hallucination,
               total_score=excluded.total_score,
               submitted_at=excluded.submitted_at""",
        (
            evaluator_id, question_idx,
            scores["context_relevance"], scores["faithfulness"],
            scores["answer_relevance"], scores["answer_correctness"],
            scores["completeness"], hallucination_flag,
            total, int(time.time()),
        ),
    )
    conn.commit()
    conn.close()


@st.cache_data
def load_questions():
    if not JSONL_PATH.exists():
        return []
    questions = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return questions


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink:            #15201C;
    --parchment:      #EDE6D6;
    --parchment-soft: #F5F0E4;
    --lacquer:        #9C2B1F;
    --lacquer-deep:   #7A2118;
    --lacquer-mist:   rgba(156,43,31,0.10);
    --brass:          #B68B4C;
    --brass-soft:     #D8BD8B;
    --brass-mist:     rgba(182,139,76,0.12);
    --stone:          #3A352B;
    --stone-light:    #6B6358;
    --signal:         #2E7D52;
    --signal-soft:    rgba(46,125,82,0.10);
    --danger:         #C0524A;
}

#MainMenu, footer, header,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--parchment) !important;
    color: var(--stone);
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
}

/* ── Rating cards: 6 equal cards in a horizontal row ─────────────── */
.st-key-metric_cards_row [data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    gap: 8px !important;
}
.st-key-metric_cards_row [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    min-width: 0 !important;
    padding: 0 !important;
}
.st-key-metric_cards_row [data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(var(--parchment-soft), var(--parchment-soft)) padding-box,
        linear-gradient(160deg, var(--brass), var(--lacquer)) border-box !important;
    border: 1.5px solid transparent !important;
    border-radius: 10px !important;
    padding: 12px 8px 10px !important;
    margin: 0 !important;
    flex: 1 1 auto !important;
    height: 100% !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    transition: box-shadow 0.15s ease !important;
}
.st-key-metric_cards_row [data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 14px rgba(156, 43, 31, 0.15) !important;
}
.st-key-metric_cards_row [data-testid="stVerticalBlockBorderWrapper"] > div,
.st-key-metric_cards_row [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    height: 100% !important;
    overflow: visible !important;
}
.st-key-metric_cards_row [data-testid="column"] {
    overflow: visible !important;
}
.st-key-metric_cards_row [data-testid="element-container"],
.st-key-metric_cards_row [data-testid="stMarkdown"] {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    line-height: 1 !important;
    overflow: visible !important;
}

/* Card title with hover tooltip */
.metric-card-title {
    position: relative;
    cursor: help;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 7px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--stone-light);
    text-align: center;
    line-height: 1.1;
    margin: 0 0 6px 0 !important;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--brass-soft);
    min-height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.metric-tooltip {
    position: absolute;
    top: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%) translateY(-4px);
    width: 190px;
    background: var(--ink);
    color: var(--parchment-soft);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11.5px;
    font-weight: 400;
    line-height: 1.55;
    border-radius: 7px;
    padding: 9px 11px;
    text-transform: none;
    letter-spacing: 0;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.15s ease, transform 0.15s ease;
    z-index: 999;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    white-space: normal;
}
.metric-card-title:hover .metric-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0);
}

/* Unselected option button — higher specificity overrides global button rule */
.st-key-metric_cards_row [data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--stone-light) !important;
    border: 1px solid var(--brass-soft) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 5px 2px !important;
    box-shadow: none !important;
    transition: background 0.1s ease, border-color 0.1s ease, color 0.1s ease !important;
}
.st-key-metric_cards_row [data-testid="stBaseButton-secondary"]:hover {
    background: var(--brass-mist) !important;
    border-color: var(--brass) !important;
    color: var(--stone) !important;
}

/* Selected option button */
.st-key-metric_cards_row [data-testid="stBaseButton-primary"] {
    background: var(--lacquer) !important;
    color: var(--parchment-soft) !important;
    border: 1px solid var(--lacquer) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 5px 2px !important;
    box-shadow: none !important;
}
.st-key-metric_cards_row [data-testid="stBaseButton-primary"]:hover {
    background: var(--lacquer-deep) !important;
    border-color: var(--lacquer-deep) !important;
}

/* ── Progress ──────────────────────────────────────────────────────── */
.prog-wrap   { margin-bottom: 14px; }
.prog-label  { display: flex; justify-content: space-between; font-size: 12px; color: var(--stone-light); margin-bottom: 5px; font-family: 'IBM Plex Mono', monospace; }
.prog-track  { height: 4px; background: var(--brass-soft); border-radius: 99px; overflow: hidden; }
.prog-fill   { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--lacquer), var(--brass)); transition: width 0.4s; }
.nav-dots    { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 14px 0; }
.nav-dot     { width: 9px; height: 9px; border-radius: 50%; background: var(--brass-soft); }
.nav-dot.done    { background: var(--signal); }
.nav-dot.current { background: var(--lacquer); }

/* ── Question block ──────────────────────────────────────────────── */
.ev-category-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--brass);
    background: var(--brass-mist);
    border: 1px solid var(--brass-soft);
    border-radius: 99px;
    padding: 3px 10px;
    margin-bottom: 10px;
}
.ev-question-text {
    font-family: 'Hanuman', sans-serif;
    font-size: 21px;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.85;
    margin: 0 0 16px 0;
    padding-bottom: 14px;
    border-bottom: 1px dashed var(--brass-soft);
}

/* ── Answer panes ─────────────────────────────────────────────────── */
.ev-answer-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
.ev-answer-label.expected  { color: var(--signal); }
.ev-answer-label.generated { color: var(--lacquer); }
.ev-answer-label .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.ev-answer-label.expected  .dot { background: var(--signal); }
.ev-answer-label.generated .dot { background: var(--lacquer); }
.ev-answer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; margin-bottom: 16px; }
.ev-answer-block { border-radius: 8px; padding: 12px 14px; font-family: 'Hanuman', sans-serif; font-size: 15px; line-height: 1.85; color: var(--stone); white-space: pre-wrap; }
.ev-answer-block.expected  { background: var(--signal-soft); border: 1px solid rgba(46,125,82,0.20); }
.ev-answer-block.generated { background: var(--brass-mist);; border: 1px solid var(--brass-soft); }
.ev-answer-blank { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--danger); font-style: italic; }
@media (max-width: 900px) { .ev-answer-grid { grid-template-columns: 1fr; } }

/* ── Score bar ────────────────────────────────────────────────────── */
.score-row { display: flex; align-items: center; gap: 16px; }
.score-label { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--stone-light); white-space: nowrap; flex-shrink: 0; }
.score-label strong { font-size: 20px; font-weight: 700; color: var(--lacquer); }
.score-track { flex: 1; height: 7px; background: var(--brass-soft); border-radius: 99px; overflow: hidden; min-width: 0; }
.score-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--danger), var(--brass) 50%, var(--signal)); transition: width 0.3s ease; }
.ev-incomplete-note { font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: var(--stone-light); margin-top: 8px; text-align: center; font-weight: 500; }

/* ── Buttons ──────────────────────────────────────────────────────── */
[data-testid="stButton"] button { background: var(--lacquer) !important; color: var(--parchment-soft) !important; border: none !important; border-radius: 8px !important; font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; font-size: 13px !important; padding: 9px 22px !important; white-space: nowrap; transition: background 0.15s ease !important; }
[data-testid="stButton"] button:hover    { background: var(--lacquer-deep) !important; }
[data-testid="stButton"] button:disabled { background: var(--brass-soft) !important; color: var(--stone-light) !important; opacity: 0.75 !important; }

/* ── Text input ───────────────────────────────────────────────────── */
[data-testid="stTextInput"] input { background: var(--parchment-soft) !important; border: 1px solid var(--brass-soft) !important; border-radius: 8px !important; color: var(--stone) !important; font-family: 'IBM Plex Sans', sans-serif !important; font-size: 14px !important; padding: 8px 12px !important; box-shadow: none !important; text-align: center !important; }
[data-testid="stTextInput"] input:focus { border-color: var(--brass) !important; box-shadow: 0 0 0 2px var(--brass-mist) !important; }
[data-testid="stTextInput"] div[data-baseweb="input"] input::placeholder { text-align: center !important; }
[data-testid="InputInstructions"] { display: none !important; }

/* ── Completion screen ────────────────────────────────────────────── */
.done-wrap  { text-align: center; padding: 60px 20px; }
.done-icon  { font-size: 52px; margin-bottom: 16px; }
.done-title { font-size: 24px; font-weight: 700; color: var(--stone); margin-bottom: 8px; }
.done-sub   { font-size: 14px; color: var(--stone-light); }
.done-stat  { display: inline-block; background: var(--parchment-soft); border: 1px solid var(--brass-soft); border-radius: 10px; padding: 16px 32px; margin-top: 24px; font-size: 32px; font-weight: 700; color: var(--lacquer); }
.done-stat span { display: block; font-size: 12px; font-weight: 400; color: var(--stone-light); margin-top: 2px; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase; }
</style>
"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _name_cell_html(label, desc):
    return (
        f'<div class="met-name-cell">'
        f'<span class="met-name">{label}'
        f'<span class="met-tooltip">{desc}</span>'
        f'</span></div>'
    )

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def page_login():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; padding-top:22vh; padding-bottom:18px;">'
        '<div style="font-size:28px;font-weight:700;color:var(--stone);margin-bottom:8px;letter-spacing:-0.5px;">Khmer Rag Insurance Chatbot Evaluation</div>'
        '<div style="font-size:15px;color:var(--stone-light);">Enter your name to begin your evaluation. You will evaluate all questions in order.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _, c_input, _ = st.columns([1, 2, 1])
    with c_input:
        name = st.text_input("name", placeholder="Enter your name", label_visibility="collapsed")
    _, c_btn, _ = st.columns([2.2, 0.6, 2.2])
    with c_btn:
        if st.button("Begin Evaluation", use_container_width=True):
            if name.strip():
                st.session_state.pop("evaluator_id", None)
                st.session_state.pop("session_id",   None)
                st.session_state.pop("q_pos",         None)
                st.session_state["evaluator_name"] = name.strip()
                st.rerun()
            else:
                st.error("Please enter your name.")


def page_evaluate(evaluator_id, evaluator_name, all_questions):
    st.markdown(CSS, unsafe_allow_html=True)

    assignment = get_or_create_assignment(evaluator_id, all_questions)
    responses  = get_responses(evaluator_id)
    total_qs   = len(assignment)
    done_count = len(responses)

    if "q_pos" not in st.session_state:
        st.session_state["q_pos"] = next(
            (i for i, qidx in enumerate(assignment) if qidx not in responses), total_qs,
        )

    q_pos = st.session_state["q_pos"]

    # ── Completion screen ────────────────────────────────────────────
    if q_pos >= total_qs:
        avg = (
            sum(r["total_score"] for r in responses.values()) / len(responses)
            if responses else 0
        )
        halluc_count = sum(1 for r in responses.values() if r.get("hallucination") == 1)
        st.markdown(
            f'<div class="done-wrap"><div class="done-icon">✓</div>'
            f'<div class="done-title">Evaluation Complete</div>'
            f'<div class="done-sub">Thank you, {evaluator_name}. All {total_qs} questions have been rated.</div>'
            f'<div class="done-stat">{avg:.1f}<span>avg score / {MAX_TOTAL}</span></div>'
            f'<p style="margin-top:16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:var(--stone-light);">'
            f'Hallucination flagged: {halluc_count} / {total_qs} answers</p></div>',
            unsafe_allow_html=True,
        )
        return

    q_idx    = assignment[q_pos]
    question = all_questions[q_idx]
    existing = responses.get(q_idx, {})

    # ── Progress ─────────────────────────────────────────────────────
    pct = int(done_count / total_qs * 100)
    dots_html = "".join(
        f'<div class="nav-dot {"done" if assignment[i] in responses else ("current" if i == q_pos else "")}"></div>'
        for i in range(total_qs)
    )
    st.markdown(
        f'<div class="prog-wrap">'
        f'<div class="prog-label">'
        f'<span>Question {q_pos + 1} of {total_qs}</span>'
        f'<span>{done_count} done · {evaluator_name}</span>'
        f'</div>'
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
        f'</div>'
        f'<div class="nav-dots">{dots_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Question + answers ───────────────────────────────────────────
    category  = question.get("category", "").replace("_", " ").title()
    expected  = question.get("expected_answer") or ""
    generated = question.get("generated_answer") or ""
    gen_body  = (
        '<span class="ev-answer-blank">[ No answer was generated ]</span>'
        if not generated.strip() else generated
    )
    st.markdown(f'<div class="ev-category-pill">{category}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ev-question-text">{question["question"]}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ev-answer-grid">'
        '<div>'
        '<div class="ev-answer-label expected"><span class="dot"></span>Expected Answer</div>'
        f'<div class="ev-answer-block expected">{expected if expected else "—"}</div>'
        '</div>'
        '<div>'
        '<div class="ev-answer-label generated"><span class="dot"></span>Chatbot Answer</div>'
        f'<div class="ev-answer-block generated">{gen_body}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Rating cards (6 in a row) ─────────────────────────────────────
    current_scores: dict = {}
    all_scored = True

    # Seed session state from DB data only on first visit to this question
    for metric in METRICS:
        sk = f"eval_{metric['key']}_{q_pos}"
        if sk not in st.session_state:
            st.session_state[sk] = existing.get(metric["key"])

    halluc_sk = f"eval_hallucination_{q_pos}"
    if halluc_sk not in st.session_state:
        st.session_state[halluc_sk] = existing.get(HALLUCINATION_KEY)

    card_defs = [
        {"key": m["key"], "label": m["label"], "desc": m["desc"], "options": CHIP_OPTS, "is_halluc": False}
        for m in METRICS
    ] + [{"key": HALLUCINATION_KEY, "label": "Hallucination",
          "desc": "Does the answer include unsupported information — fabricated facts, incorrect references, or invented details?",
          "options": HALLUC_OPTS, "is_halluc": True}]

    with st.container(key="metric_cards_row"):
        cols = st.columns(6, gap="small")
        for col, card in zip(cols, card_defs):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<p class="metric-card-title">{card["label"]}'
                        f'<span class="metric-tooltip">{card["desc"]}</span>'
                        f'</p>',
                        unsafe_allow_html=True,
                    )
                    sk = halluc_sk if card["is_halluc"] else f"eval_{card['key']}_{q_pos}"
                    for i, opt in enumerate(card["options"]):
                        val = HALLUC_MAP[opt] if card["is_halluc"] else (i + 1)
                        is_sel = st.session_state.get(sk) == val
                        if st.button(
                            opt,
                            key=f"eval_btn_{card['key']}_{i}_{q_pos}",
                            type="primary" if is_sel else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state[sk] = val
                            st.rerun()

    # Collect scores from session state
    for metric in METRICS:
        val = st.session_state.get(f"eval_{metric['key']}_{q_pos}")
        current_scores[metric["key"]] = val
        if val is None:
            all_scored = False

    hallucination_flag = st.session_state.get(halluc_sk)
    if hallucination_flag is None:
        all_scored = False

    # ── Footer: score bar + submit ────────────────────────────────────
    metric_sum = sum(v for v in current_scores.values() if v is not None)
    bar_pct    = int(metric_sum / MAX_TOTAL * 100)
    btn_label  = "Submit & Next →" if q_pos < total_qs - 1 else "Submit & Finish"

    st.markdown(
        '<hr style="border:none;border-top:1px dashed var(--brass-soft);margin:16px 0;">',
        unsafe_allow_html=True,
    )
    bar_col, btn_col = st.columns([5, 1], gap="large")
    with bar_col:
        st.markdown(
            f'<div class="score-row">'
            f'<div class="score-label">Score <strong>{metric_sum}</strong> / {MAX_TOTAL}</div>'
            f'<div class="score-track"><div class="score-fill" style="width:{bar_pct}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button(btn_label, disabled=not all_scored):
            final_scores = {k: v for k, v in current_scores.items() if v is not None}
            save_response(evaluator_id, q_idx, final_scores, hallucination_flag)
            st.session_state["q_pos"] = q_pos + 1
            st.rerun()

    if not all_scored:
        st.markdown(
            '<p class="ev-incomplete-note">Complete all ratings before submitting.</p>',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="RAG Evaluation · Khmer Insurance",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_db()
    all_questions = load_questions()
    if not all_questions:
        st.markdown(CSS, unsafe_allow_html=True)
        st.error(f"No questions found at `{JSONL_PATH}`. Run scripts/10 first.")
        return
    if "evaluator_name" not in st.session_state:
        page_login()
        return
    name = st.session_state["evaluator_name"]
    if "evaluator_id" not in st.session_state:
        eid, sid = create_new_session(name)
        st.session_state["evaluator_id"] = eid
        st.session_state["session_id"]   = sid
    evaluator_id = st.session_state["evaluator_id"]
    page_evaluate(evaluator_id, name, all_questions)


if __name__ == "__main__":
    main()