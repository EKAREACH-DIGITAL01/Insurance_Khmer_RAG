"""
Khmer Insurance RAG — Human Answer Evaluation UI
-------------------------------------------------
Launch:  python scripts/13_human_answer_evaluation.py
         (runs Streamlit on port 8502, separate from the chatbot on 8501)

Or directly:
         streamlit run scripts/13_human_answer_evaluation.py --server.port 8502
"""

import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ----------------------------------------------------------------------------
# Self-launching on port 8502
# When executed as a plain Python script rather than via `streamlit run`,
# re-launches itself through Streamlit so the user never has to remember
# the port flag.
# ----------------------------------------------------------------------------
if "__streamlit__" not in dir(st) or not st.runtime.exists():
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).resolve()),
            "--server.port", "8502",
            "--client.toolbarMode=minimal",
            "--browser.gatherUsageStats=false",
        ]
    )
    sys.exit(0)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DB_PATH   = Path("reports/human_answer_evaluation/human_answer_evaluation.db")
JSONL_PATH = Path(
    "reports/generator_model_comparison/"
    "SeaLLMs-v3-7B-Chat-Q4_K_M/"
    "generated_answer_results.jsonl"
)
QUESTIONS_PER_EVALUATOR = 25

METRICS = [
    {
        "key":   "context_relevance",
        "label": "Context Relevance",
        "desc":  "Did the retrieved sources actually contain information relevant to the question?",
    },
    {
        "key":   "faithfulness",
        "label": "Faithfulness",
        "desc":  "Does the answer only contain facts from the sources, with no invented information?",
    },
    {
        "key":   "answer_relevance",
        "label": "Answer Relevance",
        "desc":  "Does the answer actually address what was asked?",
    },
    {
        "key":   "answer_correctness",
        "label": "Answer Correctness",
        "desc":  "How well does the answer match the expected answer?",
    },
]

SCORE_LABELS = {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"}

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
            name       TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assignments (
            evaluator_id  INTEGER NOT NULL,
            question_idx  INTEGER NOT NULL,
            position      INTEGER NOT NULL,
            PRIMARY KEY (evaluator_id, question_idx),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
        CREATE TABLE IF NOT EXISTS responses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id        INTEGER NOT NULL,
            question_idx        INTEGER NOT NULL,
            context_relevance   INTEGER,
            faithfulness        INTEGER,
            answer_relevance    INTEGER,
            answer_correctness  INTEGER,
            total_score         INTEGER,
            submitted_at        INTEGER NOT NULL,
            UNIQUE (evaluator_id, question_idx),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
    """)
    conn.commit()
    conn.close()


def get_or_create_evaluator(name):
    conn = get_db()
    row = conn.execute("SELECT id FROM evaluators WHERE name = ?", (name,)).fetchone()
    if row:
        eid = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO evaluators (name, created_at) VALUES (?, ?)",
            (name, int(time.time())),
        )
        eid = cur.lastrowid
        conn.commit()
    conn.close()
    return eid


def get_or_create_assignment(evaluator_id, all_questions):
    conn = get_db()
    rows = conn.execute(
        "SELECT question_idx, position FROM assignments WHERE evaluator_id = ? ORDER BY position",
        (evaluator_id,),
    ).fetchall()
    if rows:
        assignment = [r["question_idx"] for r in rows]
    else:
        indices    = list(range(len(all_questions)))
        assignment = random.sample(indices, min(QUESTIONS_PER_EVALUATOR, len(indices)))
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
        "SELECT question_idx, context_relevance, faithfulness, "
        "answer_relevance, answer_correctness, total_score "
        "FROM responses WHERE evaluator_id = ?",
        (evaluator_id,),
    ).fetchall()
    conn.close()
    return {r["question_idx"]: dict(r) for r in rows}


def save_response(evaluator_id, question_idx, scores):
    total = sum(scores.values())
    conn  = get_db()
    conn.execute(
        """INSERT INTO responses
           (evaluator_id, question_idx, context_relevance, faithfulness,
            answer_relevance, answer_correctness, total_score, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(evaluator_id, question_idx) DO UPDATE SET
               context_relevance  = excluded.context_relevance,
               faithfulness       = excluded.faithfulness,
               answer_relevance   = excluded.answer_relevance,
               answer_correctness = excluded.answer_correctness,
               total_score        = excluded.total_score,
               submitted_at       = excluded.submitted_at
        """,
        (
            evaluator_id, question_idx,
            scores["context_relevance"], scores["faithfulness"],
            scores["answer_relevance"],  scores["answer_correctness"],
            total, int(time.time()),
        ),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

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
# CSS — light parchment theme matching app.py
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Moul&family=Hanuman:wght@300;400;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    /* Palette — mirrors app.py exactly */
    --ink:             #15201C;
    --ink-line:        #2C3D34;
    --parchment:       #EDE6D6;
    --parchment-soft:  #F5F0E4;
    --parchment-warm:  #FAF7F0;
    --lacquer:         #9C2B1F;
    --lacquer-deep:    #7A2118;
    --lacquer-mist:    rgba(156, 43, 31, 0.08);
    --lacquer-border:  rgba(156, 43, 31, 0.20);
    --brass:           #B68B4C;
    --brass-soft:      #D8BD8B;
    --brass-mist:      rgba(182, 139, 76, 0.12);
    --stone:           #3A352B;
    --stone-light:     #6B6358;
    --mist:            #9DB6AB;
    --signal:          #2E7D52;
    --signal-soft:     rgba(46, 125, 82, 0.10);
    --danger:          #C0524A;
    --danger-soft:     rgba(192, 82, 74, 0.08);

    /* Score colours */
    --score-1: #C0524A;
    --score-2: #D4843A;
    --score-3: #B68B4C;
    --score-4: #2E7D52;
    --score-5: #1A5C8A;
}

/* Reset Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

html, body,
[data-testid="stAppViewContainer"],
.stApp {
    background: var(--parchment-warm) !important;
    color: var(--stone);
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Remove Streamlit's default top padding */
[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2rem !important;
    max-width: 780px !important;
}

/* ── Login page ───────────────────────────────────────────── */
.ev-login {
    max-width: 440px;
    margin: 12vh auto 0;
    text-align: center;
    padding: 0 16px;
}
.ev-seal {
    margin-bottom: 20px;
}
.ev-login-title {
    font-family: 'Moul', serif;
    font-size: 22px;
    color: var(--lacquer);
    margin-bottom: 6px;
    line-height: 1.4;
}
.ev-login-sub {
    font-size: 13px;
    color: var(--stone-light);
    margin-bottom: 28px;
    font-family: 'IBM Plex Sans', sans-serif;
}
.ev-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 10px;
}

/* ── Progress bar ─────────────────────────────────────────── */
.ev-progress {
    margin-bottom: 24px;
}
.ev-progress-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--stone-light);
    margin-bottom: 7px;
}
.ev-progress-track {
    height: 4px;
    background: var(--brass-soft);
    border-radius: 99px;
    overflow: hidden;
}
.ev-progress-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--lacquer), var(--brass));
    transition: width 0.35s ease;
}

/* ── Nav dots ─────────────────────────────────────────────── */
.ev-dots {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-bottom: 22px;
}
.ev-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--brass-soft);
    flex-shrink: 0;
}
.ev-dot-done    { background: var(--signal); }
.ev-dot-current { background: var(--lacquer); }

/* ── Question card ───────────────────────────────────────── */
.ev-question-card {
    background: var(--parchment);
    border: 1px solid var(--brass-soft);
    border-left: 3px solid var(--brass);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.ev-question-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 10px;
}
.ev-question-text {
    font-family: 'Hanuman', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.8;
}

/* ── Answer blocks ───────────────────────────────────────── */
.ev-answer-block {
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.ev-answer-block.expected {
    background: var(--signal-soft);
    border: 1px solid rgba(46, 125, 82, 0.25);
    border-left: 3px solid var(--signal);
}
.ev-answer-block.generated {
    background: var(--parchment-soft);
    border: 1px solid var(--brass-soft);
    border-left: 3px solid var(--lacquer);
}
.ev-answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.ev-answer-label.expected  { color: var(--signal); }
.ev-answer-label.generated { color: var(--lacquer); }
.ev-answer-text {
    font-family: 'Hanuman', sans-serif;
    font-size: 14.5px;
    line-height: 1.9;
    color: var(--stone);
    white-space: pre-wrap;
}
.ev-answer-blank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--danger);
    font-style: italic;
}

/* ── Divider ─────────────────────────────────────────────── */
.ev-divider {
    border: none;
    border-top: 1px dashed var(--brass-soft);
    margin: 20px 0;
}

/* ── Metric cards ────────────────────────────────────────── */
.ev-metric-name {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--stone);
    margin-bottom: 3px;
}
.ev-metric-desc {
    font-size: 11.5px;
    color: var(--stone-light);
    line-height: 1.5;
    margin-bottom: 10px;
}

/* ── Score total bar ─────────────────────────────────────── */
.ev-total-row {
    background: var(--parchment);
    border: 1px solid var(--brass-soft);
    border-radius: 8px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 16px;
}
.ev-total-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--stone-light);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
}
.ev-total-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--stone);
    white-space: nowrap;
    min-width: 60px;
}
.ev-total-value span {
    font-size: 13px;
    font-weight: 400;
    color: var(--stone-light);
}
.ev-total-track {
    flex: 1;
    height: 5px;
    background: var(--brass-soft);
    border-radius: 99px;
    overflow: hidden;
}
.ev-total-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--score-1), var(--score-3) 50%, var(--score-5));
    transition: width 0.3s ease;
}

/* ── Completion screen ───────────────────────────────────── */
.ev-done {
    text-align: center;
    padding: 72px 24px;
}
.ev-done-seal { margin-bottom: 20px; }
.ev-done-title {
    font-family: 'Moul', serif;
    font-size: 22px;
    color: var(--lacquer);
    margin-bottom: 8px;
}
.ev-done-sub {
    font-size: 14px;
    color: var(--stone-light);
    margin-bottom: 28px;
}
.ev-done-stat {
    display: inline-block;
    background: var(--parchment);
    border: 1px solid var(--brass-soft);
    border-radius: 10px;
    padding: 18px 36px;
    font-size: 34px;
    font-weight: 700;
    color: var(--lacquer);
}
.ev-done-stat span {
    display: block;
    font-size: 11px;
    font-weight: 400;
    color: var(--stone-light);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Streamlit widget overrides ──────────────────────────── */

/* Text input */
[data-testid="stTextInput"] input {
    background: var(--parchment-soft) !important;
    border: 1px solid var(--brass-soft) !important;
    border-radius: 8px !important;
    color: var(--stone) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 11px 14px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 2px var(--brass-mist) !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* Primary submit button — login */
[data-testid="stButton"] button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--lacquer) !important;
    color: var(--parchment-soft) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
    width: 100% !important;
}
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--lacquer-deep) !important;
}

/* Secondary buttons — submit/next/finish */
[data-testid="stButton"] button,
[data-testid="stBaseButton-secondary"] {
    background: var(--lacquer) !important;
    color: var(--parchment-soft) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
[data-testid="stButton"] button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background: var(--lacquer-deep) !important;
}
[data-testid="stButton"] button:disabled {
    background: var(--brass-soft) !important;
    color: var(--stone-light) !important;
    cursor: not-allowed !important;
    opacity: 0.7 !important;
}

/* Radio buttons — score chips */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 8px !important;
    flex-wrap: wrap;
}
[data-testid="stRadio"] label {
    background: var(--parchment-soft) !important;
    border: 1.5px solid var(--brass-soft) !important;
    border-radius: 7px !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12.5px !important;
    color: var(--stone-light) !important;
    transition: all 0.12s ease !important;
    min-width: 80px;
    text-align: center;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--brass) !important;
    color: var(--stone) !important;
}
/* Selected state — one colour per score value via nth-child
   (Streamlit renders radio options in DOM order matching our 1→5 list) */
[data-testid="stRadio"] [aria-checked="true"] ~ * { }
[data-testid="stRadio"] label:has(input[value="1 — Poor"]) {
    /* handled by aria-checked below */ }
/* Generic selected style — brass/lacquer accent */
[data-testid="stRadio"] label:has(input:checked) {
    background: var(--lacquer-mist) !important;
    border-color: var(--lacquer) !important;
    color: var(--lacquer) !important;
    font-weight: 600 !important;
}
/* Hide the actual radio circle */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] { display: none; }
[data-testid="stRadio"] input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Collapse vertical gaps Streamlit inserts between widgets */
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    margin-bottom: -6px !important;
}

/* Warning for incomplete scoring */
.ev-incomplete-note {
    text-align: center;
    font-size: 11.5px;
    color: var(--stone-light);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 6px;
}
</style>
"""

SEAL_SVG = """<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg">
  <circle cx="22" cy="22" r="20" fill="none" stroke="#B68B4C" stroke-width="1.2"/>
  <circle cx="22" cy="22" r="13" fill="none" stroke="#B68B4C" stroke-width="1.2"/>
  <g stroke="#B68B4C" stroke-width="1.2">
    <line x1="22" y1="2"  x2="22" y2="9"/>
    <line x1="22" y1="35" x2="22" y2="42"/>
    <line x1="2"  y1="22" x2="9"  y2="22"/>
    <line x1="35" y1="22" x2="42" y2="22"/>
    <line x1="8"  y1="8"  x2="13" y2="13"/>
    <line x1="31" y1="31" x2="36" y2="36"/>
    <line x1="8"  y1="36" x2="13" y2="31"/>
    <line x1="31" y1="13" x2="36" y2="8"/>
  </g>
  <circle cx="22" cy="22" r="3.5" fill="#B68B4C"/>
</svg>"""


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def page_login():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="ev-login">'
        f'<div class="ev-seal">{SEAL_SVG}</div>'
        f'<div class="ev-eyebrow">RAG · Insurance Law Corpus</div>'
        f'<div class="ev-login-title">Human Answer Evaluation</div>'
        f'<div class="ev-login-sub">'
        f'Enter your name to begin. You will be assigned {QUESTIONS_PER_EVALUATOR} questions to rate.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col = st.columns([1, 2, 1])[1]
    with col:
        name = st.text_input("Your name", placeholder="e.g. Bunmeng", key="login_name")
        if st.button("Begin Evaluation", type="primary"):
            if name.strip():
                st.session_state["evaluator_name"] = name.strip()
                st.rerun()
            else:
                st.error("Please enter your name to continue.")


def page_evaluate(evaluator_id, evaluator_name, all_questions):
    st.markdown(CSS, unsafe_allow_html=True)

    assignment = get_or_create_assignment(evaluator_id, all_questions)
    responses  = get_responses(evaluator_id)
    total_qs   = len(assignment)
    done_count = len(responses)

    if "q_pos" not in st.session_state:
        st.session_state["q_pos"] = next(
            (i for i, qidx in enumerate(assignment) if qidx not in responses),
            total_qs,
        )

    q_pos = st.session_state["q_pos"]

    # ── Completion ──────────────────────────────────────────────────────────
    if q_pos >= total_qs:
        avg = (
            sum(r["total_score"] for r in responses.values()) / len(responses)
            if responses else 0
        )
        st.markdown(
            f'<div class="ev-done">'
            f'<div class="ev-done-seal">{SEAL_SVG}</div>'
            f'<div class="ev-done-title">Evaluation Complete</div>'
            f'<div class="ev-done-sub">'
            f'Thank you, {evaluator_name}. All {total_qs} questions have been rated.'
            f'</div>'
            f'<div class="ev-done-stat">{avg:.1f}'
            f'<span>avg total score / 20</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    q_idx    = assignment[q_pos]
    question = all_questions[q_idx]

    # ── Progress ─────────────────────────────────────────────────────────────
    pct = int(done_count / total_qs * 100)
    dots_html = "".join(
        f'<div class="ev-dot '
        f'{"ev-dot-done" if assignment[i] in responses else ("ev-dot-current" if i == q_pos else "")}'
        f'"></div>'
        for i in range(total_qs)
    )
    st.markdown(
        f'<div class="ev-progress">'
        f'<div class="ev-progress-meta">'
        f'<span>Question {q_pos + 1} of {total_qs}</span>'
        f'<span>{done_count} done · {evaluator_name}</span>'
        f'</div>'
        f'<div class="ev-progress-track">'
        f'<div class="ev-progress-fill" style="width:{pct}%"></div>'
        f'</div>'
        f'</div>'
        f'<div class="ev-dots">{dots_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Question ─────────────────────────────────────────────────────────────
    category = question.get("category", "").replace("_", " ").title()
    st.markdown(
        f'<div class="ev-question-card">'
        f'<div class="ev-question-eyebrow">Question {q_pos + 1} · {category}</div>'
        f'<div class="ev-question-text">{question["question"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Expected answer ───────────────────────────────────────────────────────
    expected = question.get("expected_answer", "")
    st.markdown(
        f'<div class="ev-answer-block expected">'
        f'<div class="ev-answer-label expected">Expected Answer</div>'
        f'<div class="ev-answer-text">{expected if expected else "—"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Generated answer ──────────────────────────────────────────────────────
    generated = question.get("generated_answer", "")
    is_blank  = not generated.strip()
    gen_body  = (
        f'<div class="ev-answer-blank">[ No answer was generated ]</div>'
        if is_blank
        else f'<div class="ev-answer-text">{generated}</div>'
    )
    st.markdown(
        f'<div class="ev-answer-block generated">'
        f'<div class="ev-answer-label generated">Chatbot Answer</div>'
        f'{gen_body}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="ev-divider">', unsafe_allow_html=True)

    # ── Metric rating ─────────────────────────────────────────────────────────
    existing     = responses.get(q_idx, {})
    current_scores = {}
    all_scored     = True

    score_opts = [f"{s} — {SCORE_LABELS[s]}" for s in range(1, 6)]
    reverse    = {label: s for s, label in enumerate(score_opts, start=1)}

    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]

    for i, metric in enumerate(METRICS):
        with cols[i]:
            pre = existing.get(metric["key"])
            pre_label = (
                f"{pre} — {SCORE_LABELS[pre]}" if pre in SCORE_LABELS else None
            )
            st.markdown(
                f'<div class="ev-metric-name">{metric["label"]}</div>'
                f'<div class="ev-metric-desc">{metric["desc"]}</div>',
                unsafe_allow_html=True,
            )
            chosen = st.radio(
                label=metric["key"],
                options=score_opts,
                index=score_opts.index(pre_label) if pre_label in score_opts else None,
                horizontal=True,
                key=f"radio_{metric['key']}_{q_pos}",
                label_visibility="collapsed",
            )
            val = reverse.get(chosen)
            current_scores[metric["key"]] = val
            if val is None:
                all_scored = False

    st.markdown('<hr class="ev-divider">', unsafe_allow_html=True)

    # ── Live total ─────────────────────────────────────────────────────────────
    filled = [v for v in current_scores.values() if v is not None]
    live_total = sum(filled)
    max_total  = len(METRICS) * 5
    bar_pct    = int(live_total / max_total * 100)

    st.markdown(
        f'<div class="ev-total-row">'
        f'<div class="ev-total-label">Total Score</div>'
        f'<div class="ev-total-value">{live_total}<span> / {max_total}</span></div>'
        f'<div class="ev-total-track">'
        f'<div class="ev-total-fill" style="width:{bar_pct}%"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Submit ──────────────────────────────────────────────────────────────
    btn_label = (
        "Submit & Next →"
        if q_pos < total_qs - 1
        else "Submit & Finish"
    )
    if st.button(btn_label, disabled=not all_scored):
        save_response(evaluator_id, q_idx, current_scores)
        st.session_state["q_pos"] = q_pos + 1
        st.rerun()

    if not all_scored:
        st.markdown(
            '<p class="ev-incomplete-note">Rate all 4 metrics to continue.</p>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RAG Evaluation · Khmer Insurance",
        page_icon="📋",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    init_db()
    all_questions = load_questions()

    if not all_questions:
        st.markdown(CSS, unsafe_allow_html=True)
        st.error(
            f"No questions found at `{JSONL_PATH}`. "
            "Run `scripts/10_compare_generator_models.py` first."
        )
        return

    if "evaluator_name" not in st.session_state:
        page_login()
        return

    name         = st.session_state["evaluator_name"]
    evaluator_id = get_or_create_evaluator(name)
    page_evaluate(evaluator_id, name, all_questions)


if __name__ == "__main__":
    main()