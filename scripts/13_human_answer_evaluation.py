import json
import random
import sqlite3
import time
from pathlib import Path
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DB_PATH = Path(
    "reports/human_answer_evaluation/"
    "human_answer_evaluation.db"
)
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

SCORE_LABELS = {
    1: "Poor",
    2: "Fair",
    3: "Good",
    4: "Very Good",
    5: "Excellent",
}

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
    row = conn.execute(
        "SELECT id FROM evaluators WHERE name = ?", (name,)
    ).fetchone()
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
            evaluator_id,
            question_idx,
            scores["context_relevance"],
            scores["faithfulness"],
            scores["answer_relevance"],
            scores["answer_correctness"],
            total,
            int(time.time()),
        ),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Load questions
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
# CSS
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Khmer:wght@400;500;600&display=swap');

:root {
    --bg:        #0F1117;
    --surface:   #1A1D27;
    --surface2:  #222637;
    --border:    #2E3248;
    --accent:    #5B6EF5;
    --accent-soft: rgba(91,110,245,0.12);
    --green:     #34C47C;
    --green-soft: rgba(52,196,124,0.12);
    --text:      #E8EAF6;
    --text-sub:  #8B91B5;
    --text-dim:  #555B7A;
    --score-1:   #EF5350;
    --score-2:   #FF8A65;
    --score-3:   #FFD54F;
    --score-4:   #66BB6A;
    --score-5:   #26C6DA;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* Login page */
.login-wrap {
    max-width: 400px;
    margin: 18vh auto 0 auto;
    text-align: center;
}
.login-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: -0.3px;
}
.login-sub {
    font-size: 13px;
    color: var(--text-sub);
    margin-bottom: 32px;
}

/* Progress bar */
.prog-wrap {
    margin-bottom: 28px;
}
.prog-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-sub);
    margin-bottom: 6px;
}
.prog-track {
    height: 4px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), #8B5CF6);
    transition: width 0.4s ease;
}

/* Question card */
.q-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.q-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}
.q-text {
    font-family: 'Noto Sans Khmer', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.7;
}

/* Answer blocks */
.ans-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.ans-block.expected { border-left: 3px solid var(--green); }
.ans-block.actual   { border-left: 3px solid var(--accent); }
.ans-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.ans-label.expected { color: var(--green); }
.ans-label.actual   { color: var(--accent); }
.ans-text {
    font-family: 'Noto Sans Khmer', sans-serif;
    font-size: 14.5px;
    line-height: 1.85;
    color: var(--text);
    white-space: pre-wrap;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 20px;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}
.metric-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
}
.metric-desc {
    font-size: 11.5px;
    color: var(--text-sub);
    line-height: 1.5;
    margin-bottom: 14px;
}

/* Score chips */
.score-row {
    display: flex;
    gap: 8px;
}
.score-chip {
    flex: 1;
    text-align: center;
    padding: 8px 4px 6px 4px;
    border-radius: 7px;
    border: 1.5px solid var(--border);
    cursor: pointer;
    transition: all 0.15s ease;
    background: transparent;
    font-size: 11px;
    color: var(--text-sub);
    line-height: 1.3;
}
.score-chip:hover { border-color: var(--accent); color: var(--text); }
.score-chip.selected-1 { background: rgba(239,83,80,0.18);  border-color: var(--score-1); color: var(--score-1);  font-weight: 600; }
.score-chip.selected-2 { background: rgba(255,138,101,0.18); border-color: var(--score-2); color: var(--score-2); font-weight: 600; }
.score-chip.selected-3 { background: rgba(255,213,79,0.18);  border-color: var(--score-3); color: var(--score-3);  font-weight: 600; }
.score-chip.selected-4 { background: rgba(102,187,106,0.18); border-color: var(--score-4); color: var(--score-4); font-weight: 600; }
.score-chip.selected-5 { background: rgba(38,198,218,0.18);  border-color: var(--score-5); color: var(--score-5);  font-weight: 600; }

/* Total score bar */
.total-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.total-label { font-size: 13px; color: var(--text-sub); }
.total-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
}
.total-value span { font-size: 14px; font-weight: 400; color: var(--text-sub); }
.total-bar-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
    margin: 0 20px;
}
.total-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--score-1), var(--score-3), var(--score-5));
}

/* Submit button override */
[data-testid="stButton"] button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}
[data-testid="stButton"] button:hover { opacity: 0.88 !important; }
[data-testid="stButton"] button:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}

/* Text input override */
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-soft) !important;
}

/* Completion screen */
.done-wrap {
    text-align: center;
    padding: 60px 20px;
}
.done-icon { font-size: 52px; margin-bottom: 16px; }
.done-title { font-size: 24px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.done-sub { font-size: 14px; color: var(--text-sub); }
.done-stat {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 32px;
    margin-top: 24px;
    font-size: 32px;
    font-weight: 700;
    color: var(--accent);
}
.done-stat span { display: block; font-size: 12px; font-weight: 400; color: var(--text-sub); margin-top: 2px; }

/* Nav dots */
.nav-dots {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.nav-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--border);
}
.nav-dot.done    { background: var(--green); }
.nav-dot.current { background: var(--accent); }
</style>
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def score_color(score):
    colors = {1: "#EF5350", 2: "#FF8A65", 3: "#FFD54F", 4: "#66BB6A", 5: "#26C6DA"}
    return colors.get(score, "#555B7A")


def render_score_chips(metric_key, current_value, question_pos):
    """Renders 5 score chips as Streamlit radio buttons styled via CSS."""
    options = {f"{s} — {SCORE_LABELS[s]}": s for s in range(1, 6)}
    radio_key = f"radio_{metric_key}_{question_pos}"

    # Find label matching current value
    reverse = {v: k for k, v in options.items()}
    current_label = reverse.get(current_value)

    chosen_label = st.radio(
        label=metric_key,
        options=list(options.keys()),
        index=list(options.keys()).index(current_label) if current_label else None,
        horizontal=True,
        key=radio_key,
        label_visibility="collapsed",
    )
    return options.get(chosen_label)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def page_login():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-title">Khmer Insurance RAG Evaluation</div>'
        '<div class="login-sub">Enter your name to begin. You will be assigned 25 questions.</div>',
        unsafe_allow_html=True,
    )
    name = st.text_input("Your name", placeholder="e.g. Bunmeng", label_visibility="collapsed")
    if st.button("Start Evaluation"):
        if name.strip():
            st.session_state["evaluator_name"] = name.strip()
            st.rerun()
        else:
            st.error("Please enter your name.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_evaluate(evaluator_id, evaluator_name, all_questions):
    st.markdown(CSS, unsafe_allow_html=True)

    assignment = get_or_create_assignment(evaluator_id, all_questions)
    responses  = get_responses(evaluator_id)
    total_qs   = len(assignment)
    done_count = len(responses)

    # Current question index within assignment
    if "q_pos" not in st.session_state:
        # Jump to first unanswered
        st.session_state["q_pos"] = next(
            (i for i, qidx in enumerate(assignment) if qidx not in responses),
            total_qs,
        )

    q_pos = st.session_state["q_pos"]

    # --- Completion screen ---
    if q_pos >= total_qs:
        avg = sum(r["total_score"] for r in responses.values()) / len(responses) if responses else 0
        st.markdown(
            f'<div class="done-wrap">'
            f'<div class="done-icon">✓</div>'
            f'<div class="done-title">Evaluation Complete</div>'
            f'<div class="done-sub">Thank you, {evaluator_name}. All {total_qs} questions have been rated.</div>'
            f'<div class="done-stat">{avg:.1f}<span>avg total score / 20</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    q_idx    = assignment[q_pos]
    question = all_questions[q_idx]

    # --- Progress ---
    pct = int(done_count / total_qs * 100)
    nav_dots_html = "".join(
        f'<div class="nav-dot {"done" if assignment[i] in responses else ("current" if i == q_pos else "")}"></div>'
        for i in range(total_qs)
    )
    st.markdown(
        f'<div class="prog-wrap">'
        f'<div class="prog-label"><span>Question {q_pos + 1} of {total_qs}</span>'
        f'<span>{done_count} done · {evaluator_name}</span></div>'
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
        f"</div>"
        f'<div class="nav-dots">{nav_dots_html}</div>',
        unsafe_allow_html=True,
    )

    # --- Question ---
    st.markdown(
        f'<div class="q-card">'
        f'<div class="q-label">Question {q_pos + 1}</div>'
        f'<div class="q-text">{question["question"]}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Expected answer ---
    expected = question.get("expected_answer", "—")
    st.markdown(
        f'<div class="ans-block expected">'
        f'<div class="ans-label expected">Expected Answer</div>'
        f'<div class="ans-text">{expected}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Actual answer ---
    actual = question.get("generated_answer", "—")
    st.markdown(
        f'<div class="ans-block actual">'
        f'<div class="ans-label actual">Chatbot Answer</div>'
        f'<div class="ans-text">{actual}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Metric cards ---
    # Pre-load existing scores for this question if already answered
    existing = responses.get(q_idx, {})

    st.markdown("---")

    current_scores = {}
    all_scored = True

    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]

    for i, metric in enumerate(METRICS):
        with cols[i]:
            st.markdown(
                f'<div class="metric-name">{metric["label"]}</div>'
                f'<div class="metric-desc">{metric["desc"]}</div>',
                unsafe_allow_html=True,
            )
            pre = existing.get(metric["key"])
            val = render_score_chips(metric["key"], pre, q_pos)
            current_scores[metric["key"]] = val
            if val is None:
                all_scored = False

    # --- Live total score ---
    filled_scores = [v for v in current_scores.values() if v is not None]
    live_total    = sum(filled_scores)
    max_total     = len(METRICS) * 5
    bar_pct       = int(live_total / max_total * 100)

    total_color = score_color(round(live_total / len(METRICS)) if filled_scores else 3)

    st.markdown(
        f'<div class="total-wrap">'
        f'<div><div class="total-label">Total Score</div>'
        f'<div class="total-value" style="color:{total_color}">'
        f'{live_total}<span> / {max_total}</span></div></div>'
        f'<div class="total-bar-track">'
        f'<div class="total-bar-fill" style="width:{bar_pct}%"></div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Submit ---
    if st.button(
        "Submit & Next →" if q_pos < total_qs - 1 else "Submit & Finish",
        disabled=not all_scored,
    ):
        save_response(evaluator_id, q_idx, current_scores)
        st.session_state["q_pos"] = q_pos + 1
        st.rerun()

    if not all_scored:
        st.markdown(
            '<p style="text-align:center;font-size:12px;color:#555B7A;margin-top:8px;">'
            "Rate all 4 metrics to continue.</p>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RAG Evaluation",
        page_icon="📋",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    init_db()
    all_questions = load_questions()

    if not all_questions:
        st.error(f"No questions found at {JSONL_PATH}. Run script 10 first.")
        return

    # Login state
    if "evaluator_name" not in st.session_state:
        page_login()
        return

    name         = st.session_state["evaluator_name"]
    evaluator_id = get_or_create_evaluator(name)

    page_evaluate(evaluator_id, name, all_questions)


if __name__ == "__main__":
    main()