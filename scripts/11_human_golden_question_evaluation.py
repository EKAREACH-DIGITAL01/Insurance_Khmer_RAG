"""
Khmer Insurance RAG — Human Golden Question Evaluation UI
---------------------------------------------------------
Launch:  python scripts/11_human_golden_question_evaluation.py
         (runs Streamlit on port 8503, separate from the chatbot on 8501)
"""

import html
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
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.port",
            "8503",
            "--server.headless",
            "true",
            "--client.toolbarMode",
            "minimal",
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
    )
    import webbrowser

    time.sleep(2.5)
    webbrowser.open("http://localhost:8503")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    sys.exit(0)

import streamlit as st
from streamlit.components.v1 import html as st_html

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DB_PATH = Path(
    "reports/human_golden_question_evaluation/human_golden_question_evaluation.db"
)

# Gold question sets — the SINGLE source of truth for this UI. Each entry
# carries everything the evaluator sees: the `question` text, its retrieval
# `context` (the full source chunk), and the `expected_answer` (the target
# answer). Question quality is judged entirely from these; the generator's
# produced answers are irrelevant here, so this UI reads nothing else.
GOLD_GLOB = "data/09_evaluation/*/gold_questions.json"

# question_id format: "<category>_qid_<NNN>" e.g. "guidance_qid_026"
# For every category, ids 001-025 are human-authored gold questions,
# everything above that (up to the category's max) is AI-authored.
HUMAN_QID_MAX = 25

# Each metric carries its own 5-point label set (still scored 1..5 by position).
METRICS = [
    {
        "key": "clarity",
        "label": "Clarity",
        "desc": "How clear and understandable is the question?",
        "options": ["Very Poor", "Poor", "Average", "Good", "Excellent"],
    },
    {
        "key": "relevance",
        "label": "Relevance",
        "desc": "How relevant is the question to insurance?",
        "options": [
            "Not Relevant",
            "Slightly",
            "Moderate",
            "Relevant",
            "Highly Relevant",
        ],
    },
    {
        "key": "difficulty",
        "label": "Difficulty",
        "desc": "Is the question appropriate for the RAG system?",
        "options": [
            "Too Easy/Hard",
            "Poor Level",
            "Adequate",
            "Good Level",
            "Perfect Level",
        ],
    },
    {
        "key": "answer_quality",
        "label": "Answer Quality",
        "desc": "How good is the target answer?",
        "options": ["Very Poor", "Poor", "Adequate", "Good", "Excellent"],
    },
    {
        "key": "uniqueness",
        "label": "Uniqueness",
        "desc": "How unique is this question?",
        "options": [
            "Very Redundant",
            "Redundant",
            "Somewhat Unique",
            "Unique",
            "Very Unique",
        ],
    },
]

MAX_TOTAL = len(METRICS) * 5  # 25

# Each evaluator is assigned a random subset of this many questions, drawn once
# from the full 400-question gold set and persisted in the assignments table.
QUESTIONS_PER_EVALUATOR = 50

# ----------------------------------------------------------------------------
# Question id parsing
# ----------------------------------------------------------------------------


def collapse_blank_lines(text: str) -> str:
    """
    Normalise the ragged whitespace in gold `context` / `expected_answer`
    text for display. Source chunks come from OCR'd PDFs and carry a mix of
    single '\\n' and runs of blank lines between list items, which render as
    uneven gaps under CSS `white-space: pre-wrap`. This strips each line's
    trailing whitespace and collapses any run of one-or-more blank lines to a
    single line break, so every list item sits on its own line with no blank
    line between. Content is otherwise untouched (no word joining).
    """
    # Keep only non-blank lines (trailing whitespace stripped); a single '\n'
    # join then puts each item on its own line with no blank line between.
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


def humanize_time_ago(ts: int) -> str:
    """Turn a unix timestamp into 'just now' / 'N minutes ago' / 'N hours ago'
    / 'N days ago' for the target-answer edit log."""
    delta = int(time.time()) - int(ts)
    if delta < 45:
        return "just now"
    if delta < 3600:
        n = max(1, delta // 60)
        return f"{n} minute{'s' if n != 1 else ''} ago"
    if delta < 86400:
        n = delta // 3600
        return f"{n} hour{'s' if n != 1 else ''} ago"
    n = delta // 86400
    return f"{n} day{'s' if n != 1 else ''} ago"


def parse_question_id(qid: str):
    """
    'guidance_qid_026' -> ('guidance', 'ai')
    'law_qid_007'      -> ('law', 'human')
    Falls back to ('unknown', 'unknown') if the id doesn't match the
    expected '<category>_qid_<NNN>' pattern, so a malformed row never
    crashes the app — it just gets filtered out of human/ai-only modes.
    """
    try:
        prefix, num_str = qid.rsplit("_qid_", 1)
        num = int(num_str)
    except (ValueError, AttributeError):
        return "unknown", "unknown"
    source = "human" if num <= HUMAN_QID_MAX else "ai"
    return prefix, source


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait (don't immediately raise) if another evaluator holds a write
    # lock. WAL mode lets readers and writers proceed concurrently instead of
    # blocking each other, and busy_timeout backs the wait with a 5s budget --
    # together these keep concurrent submits from multiple evaluators from ever
    # surfacing a "database is locked" error at human typing pace.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
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
            question_id  TEXT NOT NULL,
            position     INTEGER NOT NULL,
            PRIMARY KEY (evaluator_id, question_id),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
        CREATE TABLE IF NOT EXISTS responses (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator_id   INTEGER NOT NULL,
            question_id    TEXT NOT NULL,
            clarity        INTEGER,
            relevance      INTEGER,
            difficulty     INTEGER,
            answer_quality INTEGER,
            uniqueness     INTEGER,
            total_score    INTEGER,
            comment        TEXT,
            submitted_at   INTEGER NOT NULL,
            UNIQUE (evaluator_id, question_id),
            FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
        );
        -- Shared, global edits to a question's target answer. One row per edit
        -- (full history kept for the activity log); the current target answer
        -- for a question is the most recent row, falling back to the gold
        -- expected_answer when there are no edits.
        CREATE TABLE IF NOT EXISTS target_edits (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id    TEXT NOT NULL,
            evaluator_id   INTEGER NOT NULL,
            evaluator_name TEXT NOT NULL,
            new_answer     TEXT NOT NULL,
            edited_at      INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_target_edits_qid
            ON target_edits (question_id, edited_at DESC);
    """)
    # --- lightweight migrations for older DBs ---
    resp_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(responses)").fetchall()
    }
    for col in ("clarity", "relevance", "difficulty", "answer_quality", "uniqueness"):
        if col not in resp_cols:
            conn.execute(f"ALTER TABLE responses ADD COLUMN {col} INTEGER")
    if "question_id" not in resp_cols:
        conn.execute("ALTER TABLE responses ADD COLUMN question_id TEXT")
    if "comment" not in resp_cols:
        conn.execute("ALTER TABLE responses ADD COLUMN comment TEXT")

    eval_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(evaluators)").fetchall()
    }
    if "session_id" not in eval_cols:
        conn.execute("ALTER TABLE evaluators ADD COLUMN session_id TEXT")

    assign_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(assignments)").fetchall()
    }
    if "question_id" not in assign_cols:
        conn.execute("ALTER TABLE assignments ADD COLUMN question_id TEXT")

    conn.commit()
    conn.close()


def get_or_create_evaluator(name):
    """
    Resume support: an evaluator is identified by NAME ALONE (case-insensitive).
    Ratings are keyed by (evaluator_id, question_id), so a single name owns one
    fixed set of randomly-assigned questions. Returns (evaluator_id, session_id,
    resumed).
    """
    conn = get_db()
    row = conn.execute(
        "SELECT id, session_id FROM evaluators "
        "WHERE LOWER(name) = LOWER(?) "
        "ORDER BY created_at ASC LIMIT 1",
        (name,),
    ).fetchone()
    if row:
        eid, sid = row["id"], row["session_id"]
        conn.close()
        return eid, sid, True
    session_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    cur = conn.execute(
        "INSERT INTO evaluators (name, session_id, created_at) VALUES (?, ?, ?)",
        (name, session_id, int(time.time())),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid, session_id, False


def get_or_create_assignment(evaluator_id, all_question_ids):
    """
    all_question_ids: the full list of available question_ids (all 400 gold
    questions). On first login a subset of QUESTIONS_PER_EVALUATOR is drawn for
    this evaluator and persisted; on resume the same stored subset is returned.
    Assignment is keyed by question_id so each evaluator keeps their own fixed
    slice of the full id space.

    Coverage-aware selection (guarantees full coverage): instead of an
    independent random sample -- which leaves ~a quarter of the 400 questions
    unrated with only ~10 evaluators, because independent draws overlap -- this
    picks the QUESTIONS_PER_EVALUATOR questions that currently have the FEWEST
    existing assignments across all evaluators, breaking ties randomly. This
    means every question is assigned once before any is assigned twice, so the
    first ceil(400/50)=8 evaluators cover all 400 questions with no gaps, and
    each later evaluator begins the next redundancy layer on the least-covered
    questions.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT question_id, position FROM assignments WHERE evaluator_id = ? ORDER BY position",
        (evaluator_id,),
    ).fetchall()
    if rows:
        assignment = [r["question_id"] for r in rows]
    else:
        # Current assignment count per question across all evaluators.
        counts = {qid: 0 for qid in all_question_ids}
        for r in conn.execute(
            "SELECT question_id, COUNT(*) AS n FROM assignments GROUP BY question_id"
        ).fetchall():
            if r["question_id"] in counts:
                counts[r["question_id"]] = r["n"]
        k = min(QUESTIONS_PER_EVALUATOR, len(all_question_ids))
        # Random tie-break within each coverage level: shuffle first, then a
        # stable sort by count keeps equally-covered questions in random order.
        shuffled = list(all_question_ids)
        random.shuffle(shuffled)
        ordered = sorted(shuffled, key=lambda qid: counts[qid])
        assignment = ordered[:k]
        for pos, qid in enumerate(assignment):
            conn.execute(
                "INSERT INTO assignments (evaluator_id, question_id, position) VALUES (?, ?, ?)",
                (evaluator_id, qid, pos),
            )
        conn.commit()
    conn.close()
    return assignment


def get_responses(evaluator_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT question_id, clarity, relevance, difficulty, "
        "answer_quality, uniqueness, total_score, comment "
        "FROM responses WHERE evaluator_id = ?",
        (evaluator_id,),
    ).fetchall()
    conn.close()
    return {r["question_id"]: dict(r) for r in rows}


def save_response(evaluator_id, question_id, scores, comment=""):
    total = sum(scores.values())
    conn = get_db()
    conn.execute(
        """INSERT INTO responses
           (evaluator_id, question_id, clarity, relevance, difficulty,
            answer_quality, uniqueness, total_score, comment, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(evaluator_id, question_id) DO UPDATE SET
               clarity=excluded.clarity,
               relevance=excluded.relevance,
               difficulty=excluded.difficulty,
               answer_quality=excluded.answer_quality,
               uniqueness=excluded.uniqueness,
               total_score=excluded.total_score,
               comment=excluded.comment,
               submitted_at=excluded.submitted_at""",
        (
            evaluator_id,
            question_id,
            scores["clarity"],
            scores["relevance"],
            scores["difficulty"],
            scores["answer_quality"],
            scores["uniqueness"],
            total,
            comment.strip(),
            int(time.time()),
        ),
    )
    conn.commit()
    conn.close()


def load_gold():
    """
    Loads the gold question sets, keyed by question_id. Each entry is the
    complete authoritative record for a question: its `question` text, the
    retrieval `context` (the full source chunk), and the `expected_answer`
    (target answer). This is the only input the UI reads.
    """
    gold = {}
    from pathlib import Path as _P
    from glob import glob

    for path in sorted(glob(GOLD_GLOB)):
        try:
            entries = json.loads(_P(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for e in entries:
            qid = e.get("question_id")
            if qid:
                gold[qid] = e
    return gold


def get_target_override(question_id):
    """Most recent shared edit for this question's target answer, or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT new_answer, evaluator_name, edited_at FROM target_edits "
        "WHERE question_id = ? ORDER BY edited_at DESC, id DESC LIMIT 1",
        (question_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_target_history(question_id, limit=10):
    """Recent target-answer edits for this question (newest first)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT evaluator_name, edited_at FROM target_edits "
        "WHERE question_id = ? ORDER BY edited_at DESC, id DESC LIMIT ?",
        (question_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_target_edit(question_id, evaluator_id, evaluator_name, new_answer):
    conn = get_db()
    conn.execute(
        "INSERT INTO target_edits "
        "(question_id, evaluator_id, evaluator_name, new_answer, edited_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (question_id, evaluator_id, evaluator_name, new_answer, int(time.time())),
    )
    conn.commit()
    conn.close()


@st.cache_data
def load_questions():
    """
    Loads every gold question from `GOLD_GLOB` and annotates each with a
    derived `category` (from the question_id, falling back to a stored
    `category` field) and `source` ('human' | 'ai'). The gold entry already
    carries `question`, `context`, and `expected_answer`, so no other input
    is needed. Returned as a dict keyed by question_id for O(1) lookup, plus
    a list that preserves file order for stable default sorting.
    """
    gold = load_gold()
    by_id = {}
    order = []
    for qid, g in gold.items():
        cat_from_id, source = parse_question_id(qid)
        row = dict(g)
        row["question_id"] = qid
        row["category"] = g.get("category") or cat_from_id
        row["source"] = source
        row["context"] = g.get("context", "")
        row["expected_answer"] = g.get("expected_answer", "")
        by_id[qid] = row
        order.append(qid)
    return by_id, order


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
    --indigo:         #3B4C8C;
    --indigo-soft:    rgba(59,76,140,0.10);
    --indigo-mist:    rgba(59,76,140,0.12);
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
    padding-top: 0.6rem !important;
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
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    text-align: center;
    line-height: 1.1;
    margin: 0 0 6px 0 !important;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--brass-soft);
    min-height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    /* Keep every metric title on a single line so all six cards stay the same
       height and align perfectly -- without this, the longest label
       ("Answer Correctness") can wrap to two lines on some widths/zoom levels. */
    white-space: nowrap;
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

/* Native st.slider, used as the sole question-navigator / progress element.
   Filled portion uses the same lacquer->brass gradient as the old progress
   bar fill. Unfilled portion is a faint hairline, not a solid block. No
   numeric labels anywhere on the slider — min/max endpoints and the live
   thumb-value bubble are all hidden; only the thumb itself is visible. */
.st-key-question_slider_row [data-testid="stSlider"] { padding-top: 0 !important; margin-bottom: 4px !important; }

/* Wipe BaseWeb's own backgrounds — the gray rectangle wrappers AND the native
   inner track/fill — so ONLY our two custom overlay bars are visible. */
.st-key-question_slider_row [data-testid="stSlider"],
.st-key-question_slider_row [data-testid="stSlider"] > div,
.st-key-question_slider_row [data-baseweb="slider"],
.st-key-question_slider_row [data-baseweb="slider"] div:not([role="slider"]):not(.cust-track-under):not(.cust-track-fill) {
    background: transparent !important;
    box-shadow: none !important;
}

/* Root wrapper hosts the two absolutely-positioned overlay bars (added by JS). */
.st-key-question_slider_row [data-baseweb="slider"] > div { position: relative !important; }

/* Under layer: fixed faint hairline spanning the whole track. */
.cust-track-under {
    position: absolute; left: 0; right: 0; top: 50%;
    transform: translateY(-50%);
    height: 4px; border-radius: 99px;
    background: var(--brass-soft);
    z-index: 0; pointer-events: none;
}
/* Filled layer: lacquer->brass gradient; width is set by JS from the thumb
   position, so it grows as you slide. Sits on top of the under layer. */
.cust-track-fill {
    position: absolute; left: 0; top: 50%;
    transform: translateY(-50%);
    height: 4px; width: 0; border-radius: 99px;
    background: linear-gradient(90deg, var(--danger), var(--brass) 50%, var(--signal));
    z-index: 1; pointer-events: none;
}

/* Thumb — above both overlay bars. */
.st-key-question_slider_row [data-baseweb="slider"] [role="slider"] {
    position: relative !important;
    z-index: 3 !important;
    background: var(--lacquer) !important;
    border: 2px solid var(--parchment-soft) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
    height: 16px !important;
    width: 16px !important;
}

/* Hide ONLY the min/max endpoint labels at the two ends of the slider.
   The value bubble on the thumb (stSliderThumbValue) is intentionally kept.
   Streamlit 1.58 renames the test-id; keep the old name too for safety. */
.st-key-question_slider_row [data-testid="stSliderTickBar"],
.st-key-question_slider_row [data-testid="stTickBar"] { display: none !important; }

/* Re-theme the value bubble on the thumb — default is Streamlit orange. */
.st-key-question_slider_row [data-testid="stSliderThumbValue"] {
    color: var(--lacquer) !important;
    background: transparent !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
}

/* ── Top header: mode badge + submitted/remaining stat chips ─────────── */
.ev-top { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 2px; }
.ev-top-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ev-top-right { display: flex; align-items: center; gap: 8px; }
.stat-chip { position: relative; cursor: help; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.04em; padding: 4px 11px; border-radius: 99px; }
.stat-chip.submitted { background: var(--signal-soft); color: var(--signal); border: 1px solid rgba(46,125,82,0.25); }
.stat-chip.remaining { background: var(--lacquer-mist); color: var(--lacquer); border: 1px solid rgba(156,43,31,0.25); }

/* Hover tooltip listing the actual question numbers for each chip. */
.chip-tip {
    position: absolute; top: calc(100% + 8px); right: 0; left: auto;
    width: 280px; max-height: 220px; overflow-y: auto;
    background: var(--ink); color: var(--parchment-soft);
    font-family: 'IBM Plex Sans', sans-serif; font-size: 11.5px; font-weight: 400;
    line-height: 1.7; letter-spacing: 0; text-transform: none;
    border-radius: 7px; padding: 9px 11px;
    opacity: 0; visibility: hidden; transition: opacity 0.15s ease;
    z-index: 999; box-shadow: 0 4px 16px rgba(0,0,0,0.25); white-space: normal;
}
.stat-chip:hover .chip-tip { opacity: 1; visibility: visible; }
.chip-tip .tip-head { display: block; font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.65; margin-bottom: 5px; }
.ev-name-ic { width: 13px; height: 13px; flex-shrink: 0; }
.ev-eval-name { display: inline-flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; color: var(--brass); background: var(--brass-mist); border: 1px solid var(--brass-soft); border-radius: 99px; padding: 4px 11px; }
.ev-qpos-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; margin-top: 0; margin-bottom: 12px; }
.ev-qpos { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 700; color: var(--stone); }
.ev-qpos strong { color: var(--lacquer); font-weight: 700; }

/* ── Back / Next icon buttons flanking the slider ────────────────────── */
.st-key-question_slider_row [data-testid="stButton"] button {
    background: transparent !important;
    color: var(--lacquer) !important;
    border: 1px solid var(--brass-soft) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 4px 0 !important;
    min-height: 30px !important;
}
.st-key-question_slider_row [data-testid="stButton"] button:hover:not(:disabled) {
    background: var(--brass-mist) !important;
    border-color: var(--brass) !important;
}
.st-key-question_slider_row [data-testid="stButton"] button:disabled {
    background: transparent !important;
    color: var(--stone-light) !important;
    border-color: var(--brass-soft) !important;
    opacity: 0.45 !important;
}

/* ── Comment / feedback box ──────────────────────────────────────────── */
.ev-comment-label { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--stone-light); margin: 14px 0 6px 0; }
[data-testid="stTextArea"] textarea {
    background: var(--parchment-soft) !important;
    border: 1px solid var(--brass-soft) !important;
    border-radius: 8px !important;
    color: var(--stone) !important;
    font-family: 'Hanuman', 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    box-shadow: none !important;
}
[data-testid="stTextArea"] textarea:focus { border-color: var(--brass) !important; box-shadow: 0 0 0 2px var(--brass-mist) !important; }
.ev-comment-view {
    background: var(--parchment-soft);
    border: 1px solid var(--brass-soft);
    border-radius: 8px;
    padding: 10px 12px;
    font-family: 'Hanuman', sans-serif;
    font-size: 14px;
    color: var(--stone);
    line-height: 1.7;
    white-space: pre-wrap;
    min-height: 42px;
}

/* ── Target-answer edit log (replaces the old comment box) ───────────── */
.ev-log-list {
    list-style: none; margin: 6px 0 0 0;
    background: var(--parchment-soft);
    border: 1px solid var(--brass-soft);
    border-radius: 8px;
    padding: 2px 12px;
}
.ev-log-item {
    display: flex; align-items: center; gap: 7px;
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    color: var(--stone-light); padding: 4px 0;
    border-bottom: 1px solid var(--brass-soft);
}
.ev-log-item:last-child { border-bottom: none; }
.ev-log-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--indigo); flex-shrink: 0; }
.ev-log-when { color: var(--lacquer); font-weight: 600; }
.ev-log-who  { color: var(--indigo); font-weight: 600; }
.ev-log-empty {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 11px;
    color: var(--stone-light); font-style: italic;
    margin-top: 6px; padding: 6px 0;
}
.ev-target-edited-flag {
    font-family: 'IBM Plex Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;
    color: var(--indigo); background: var(--indigo-mist);
    border: 1px solid rgba(59,76,140,0.25); border-radius: 99px;
    padding: 1px 8px; margin-left: 6px;
}

/* ── Inline progress label: "Question X of Y" on the left, done-count/name
   on the right — same split as the original layout, sits above the slider ── */
.prog-label-inline {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--stone-light);
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 8px;
}

.slider-status-row { margin: 2px 0 4px 0; }
.slider-status {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 0;
}
.slider-status.submitted { color: var(--signal); }
.slider-status.open      { color: var(--lacquer); }

/* ── Mode badge (own row; progress label sits below it on the left) ── */
.mode-badge-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.mode-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 99px;
    font-weight: 600;
}
.mode-badge.human { background: var(--signal-soft); color: var(--signal); border: 1px solid rgba(46,125,82,0.25); }
.mode-badge.ai    { background: var(--indigo-mist); color: var(--indigo); border: 1px solid rgba(59,76,140,0.25); }
.mode-badge.all   { background: var(--lacquer-mist); color: var(--lacquer); border: 1px solid rgba(156,43,31,0.25); }

/* ── Question block ──────────────────────────────────────────────── */
.ev-pill-row { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.ev-category-pill, .ev-source-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    border-radius: 99px;
    padding: 3px 10px;
    margin: 0;
}
.ev-category-pill {
    color: var(--brass);
    background: var(--brass-mist);
    border: 1px solid var(--brass-soft);
}
.ev-source-pill.human {
    color: var(--signal);
    background: var(--signal-soft);
    border: 1px solid rgba(46,125,82,0.25);
}
.ev-source-pill.ai {
    color: var(--indigo);
    background: var(--indigo-mist);
    border: 1px solid rgba(59,76,140,0.25);
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
.ev-answer-label { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 7px; }
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
/* NOTE: height is fixed (42px) and the outer BaseWeb wrapper is forced to
   the same height with visible overflow, otherwise Streamlit's wrapper clips
   the input's bottom edge — which was eating the golden focus border. */
[data-testid="stTextInput"] input {
    background: var(--parchment-soft) !important;
    border: 1px solid var(--brass-soft) !important;
    border-radius: 8px !important;
    color: var(--stone) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    height: 42px !important;
    padding: 0 12px !important;
    box-sizing: border-box !important;
    box-shadow: none !important;
    text-align: center !important;
}
/* Keep BaseWeb's input shells transparent / unclipped so the focus ring shows. */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    overflow: visible !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--brass) !important; box-shadow: 0 0 0 2px var(--brass-mist) !important; }
[data-testid="stTextInput"] div[data-baseweb="input"] input::placeholder { text-align: center !important; }
[data-testid="InputInstructions"] { display: none !important; }

/* ── Mode radio (login page) ──────────────────────────────────────── */
/* "Choose mode:" label sits to the left of the radio row. Its top padding is
   tuned to vertically centre it against the radio options, which carry their
   own internal padding. */
/* Centre the whole mode line. The row is an inline-flex element that shrinks to
   its content, and text-align:center on the ancestors centres that inline block
   on screen — robust regardless of the widget's own width.
   .st-key-login_mode_radio is the one class we know Streamlit 1.58 emits. */
/* Force EVERY wrapper from the key class down to the options row to be a
   centring flex container, so whichever level was holding the line to the left
   is overridden. */
.st-key-login_mode_radio,
.st-key-login_mode_radio > div,
.st-key-login_mode_radio [data-testid="stRadio"],
.st-key-login_mode_radio [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
}
.st-key-login_mode_radio { margin-bottom: -18px !important; }
/* The options row itself: one non-wrapping line, shrunk to content and centred.
   The "Choose Gold Question Mode:" caption is the first flex item via ::before. */
.st-key-login_mode_radio div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 16px !important;
    width: fit-content !important;
    margin: 0 auto !important;
    /* small manual nudge to the right of dead-centre */
    padding-left: 20px !important;
}
.st-key-login_mode_radio div[role="radiogroup"]::before {
    content: "Choose Gold Question Mode:";
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--stone);
    white-space: nowrap;
    margin-right: 6px;
}
/* align-items:center lines each radio circle up with its label text; the small
   margin-top nudges the circle down so it isn't sitting slightly high. */
[data-testid="stRadio"] label { font-family: 'IBM Plex Sans', sans-serif !important; align-items: center !important; white-space: nowrap !important; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { margin-top: 2px !important; }
[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
    color: var(--stone) !important;
    margin: 0 !important;
}
/* Theme the radio control to the lacquer red instead of Streamlit's default.
   accent-color covers the native input; the :has() rules below override
   BaseWeb's own styled circle (ring + filled dot) for full coverage. */
[data-testid="stRadio"] input[type="radio"] { accent-color: var(--lacquer) !important; }
/* Unselected ring (covers every nested div + box-shadow ring BaseWeb may use). */
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child,
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child div {
    border-color: var(--brass-soft) !important;
    box-shadow: none !important;
}
/* Selected control — three layers, all kept on-theme (was: lacquer border,
   ORANGE middle, lacquer dot):
     • OUTER ring  → lacquer border + box-shadow
     • MIDDLE fill → soft lacquer mist (the orange band) so it reads as a halo
     • INNER dot   → solid lacquer */
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:first-child {
    border-color: var(--lacquer) !important;
    background-color: rgba(156,43,31,0.32) !important;
    box-shadow: 0 0 0 1px var(--lacquer) !important;
}
/* Selected INNER dot: solid lacquer fill. */
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:first-child div {
    background-color: var(--lacquer) !important;
    border-color: var(--lacquer) !important;
}

/* ── Begin Evaluation button (login page) — matched to the name input ──── */
/* Forced to the same 42px height as the text input so the two sit flush on the
   shared row; box-sizing keeps the height honest regardless of padding. */
.st-key-begin_eval_row [data-testid="stButton"] button {
    width: 100% !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 14px !important;
    box-sizing: border-box !important;
}

/* ── Explanation panel (login page) ──────────────────────────────── */
.login-explain {
    margin-top: 26px;
    padding: 18px 20px;
    background: var(--parchment-soft);
    border: 1px solid var(--brass-soft);
    border-radius: 10px;
}
.login-explain h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--brass);
    margin: 0 0 10px 0;
}
.login-explain h4.second { margin-top: 16px; }
.login-mode-line, .login-metric-line {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: var(--stone);
    line-height: 1.6;
    margin: 0 0 5px 0;
}
.login-mode-line strong, .login-metric-line strong { color: var(--ink); }
.login-mode-line .tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 1px 7px;
    border-radius: 99px;
    margin-right: 6px;
    font-weight: 600;
}
.login-mode-line .tag.human { background: var(--signal-soft); color: var(--signal); }
.login-mode-line .tag.ai    { background: var(--indigo-mist); color: var(--indigo); }
.login-mode-line .tag.all   { background: var(--lacquer-mist); color: var(--lacquer); }
.login-hint {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: var(--stone-light);
    font-style: italic;
    margin: 12px 0 0 0;
}

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
# Slider fill JS
# ----------------------------------------------------------------------------
# BaseWeb bakes the slider's fill percentage into an inline gradient that CSS
# can't read, so a pure-CSS gradient-that-grows-to-the-thumb is impossible.
# This runs in a 0-height component iframe (same-origin → can reach the parent
# document), reads the thumb position on every move, and sets the width of the
# custom .cust-track-fill bar accordingly. Result: a fixed faint under-track
# plus a lacquer->brass gradient that grows as you drag, with the round thumb
# kept on top.
SLIDER_FILL_JS = """
<script>
(function () {
    const pdoc = window.parent.document;
    if (window.parent.__custSliderTimer) {
        clearInterval(window.parent.__custSliderTimer);
    }
    function tick() {
        const slider = pdoc.querySelector(
            '.st-key-question_slider_row [data-baseweb="slider"]'
        );
        if (!slider) return;
        const root  = slider.querySelector(':scope > div');
        const thumb = slider.querySelector('[role="slider"]');
        if (!root || !thumb) return;
        let under = root.querySelector('.cust-track-under');
        let fill  = root.querySelector('.cust-track-fill');
        if (!under) {
            under = pdoc.createElement('div');
            under.className = 'cust-track-under';
            root.appendChild(under);
        }
        if (!fill) {
            fill = pdoc.createElement('div');
            fill.className = 'cust-track-fill';
            root.appendChild(fill);
        }
        const rr = root.getBoundingClientRect();
        const tr = thumb.getBoundingClientRect();
        if (rr.width === 0) return;
        let pct = ((tr.left + tr.width / 2) - rr.left) / rr.width * 100;
        pct = Math.max(0, Math.min(100, pct));
        fill.style.width = pct + '%';
    }
    window.parent.__custSliderTimer = setInterval(tick, 80);
    tick();
})();
</script>
"""

# ----------------------------------------------------------------------------
# Target-answer textarea auto-grow JS
# ----------------------------------------------------------------------------
# Streamlit's text_area has a fixed pixel height (and would show an inner
# scrollbar for long answers). This runs in a 0-height same-origin component
# iframe, reaches into the parent document, and continuously sizes the target
# editor's <textarea> to its exact content height — so the whole answer shows
# with no empty gap below and no inner scroll, growing live as the user types.
# The surrounding Streamlit wrappers (which carry the fixed height) are set to
# auto so they don't clip or leave a gap.
TARGET_AUTOGROW_JS = """
<script>
(function () {
    const pdoc = window.parent.document;
    if (window.parent.__targetGrowTimer) {
        clearInterval(window.parent.__targetGrowTimer);
    }
    function grow() {
        const box = pdoc.querySelector('.st-key-target_editor_box');
        if (!box) return;
        const ta = box.querySelector('textarea');
        if (!ta) return;
        // measure natural content height
        ta.style.setProperty('height', 'auto', 'important');
        const h = ta.scrollHeight;
        ta.style.setProperty('height', h + 'px', 'important');
        ta.style.setProperty('overflow-y', 'hidden', 'important');
        ta.style.setProperty('resize', 'none', 'important');
        // release fixed height on the wrappers Streamlit sized, so the pane
        // hugs the textarea with no gap.
        let p = ta.parentElement, hops = 0;
        while (p && hops < 5 && !p.classList.contains('st-key-target_editor_box')) {
            p.style.setProperty('height', 'auto', 'important');
            p.style.setProperty('max-height', 'none', 'important');
            p = p.parentElement;
            hops++;
        }
    }
    window.parent.__targetGrowTimer = setInterval(grow, 60);
    grow();
})();
</script>
"""

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------


def page_login():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; padding-top:10vh; padding-bottom:18px;">'
        '<div style="font-size:28px;font-weight:700;color:var(--stone);margin-bottom:8px;letter-spacing:-0.5px;">Khmer Rag Insurance Golden Question Evaluation</div>'
        f'<div style="font-size:15px;color:var(--stone-light);">Enter your name to begin — or to continue an evaluation you started earlier. '
        f"You'll be assigned a random {QUESTIONS_PER_EVALUATOR} out of our total 400 golden questions.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Name input + Begin button: nested inside a single centred middle column
    #    so the pair is centred as one unit (not packed against either side). ──
    _, c_pair, _ = st.columns([1, 2, 1])
    with c_pair:
        c_name, c_btn = st.columns([1.4, 0.6])
    with c_name:
        name = st.text_input(
            "name", placeholder="Enter your name", label_visibility="collapsed"
        )
    name_clean = name.strip()
    can_begin = bool(name_clean)
    with c_btn:
        with st.container(key="begin_eval_row"):
            if st.button(
                "Begin Evaluation", disabled=not can_begin, use_container_width=True
            ):
                st.session_state.pop("evaluator_id", None)
                st.session_state.pop("session_id", None)
                st.session_state.pop("q_pos", None)
                st.session_state["evaluator_name"] = name_clean
                st.rerun()

    # ── Contextual hint: tell the user exactly what's still missing ──────────
    if not name_clean:
        st.markdown(
            '<p style="text-align:center;color:var(--stone-light);font-size:12.5px;'
            "font-family:'IBM Plex Mono',monospace;margin-top:2px;\">"
            "Enter your name above to continue.</p>",
            unsafe_allow_html=True,
        )

    # ── Explanation panel ───────────────────────────────────────────────────
    # Wider than the input rows above (which stay at [1,2,1]) so the info card
    # reads fuller while remaining centred — extended out to both sides.
    _, c_exp, _ = st.columns([0.5, 3, 0.5])
    with c_exp:
        metric_lines = "".join(
            f'<p class="login-metric-line"><strong>{m["label"]}</strong> — {m["desc"]}</p>'
            for m in METRICS
        )
        st.markdown(
            f'<div class="login-explain">'
            f"<h4>How the assignment works</h4>"
            f'<p class="login-metric-line">When you begin, you are assigned a random '
            f"<strong>{QUESTIONS_PER_EVALUATOR} questions</strong> drawn from our full set of "
            f"<strong>400 golden questions</strong>. That same set stays yours every time you return.</p>"
            f'<h4 class="second">Your progress is always saved</h4>'
            f'<p class="login-metric-line">Don\'t worry about losing your work — every rating is '
            f"saved the moment you submit it. You can close the browser at any time and come back "
            f"later to continue right where you left off.</p>"
            f'<p class="login-metric-line">To resume, just <strong>enter the same name</strong> — '
            f"that's all. Your assigned questions and ratings are kept against your name, so nothing "
            f"is ever repeated or lost.</p>"
            f'<h4 class="second">What do the rating metrics mean?</h4>'
            f"{metric_lines}"
            f'<p class="login-hint">Tip: once you start evaluating, hover over any metric '
            f"title to see this same explanation again.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


def page_evaluate(evaluator_id, evaluator_name, assignment, by_id):
    st.markdown(CSS, unsafe_allow_html=True)

    # `assignment` is this evaluator's fixed random slice of question_ids,
    # persisted at first login. responses holds their ratings for those ids.
    responses = get_responses(evaluator_id)
    total_qs = len(assignment)
    done_count = sum(1 for qid in assignment if qid in responses)

    if total_qs == 0:
        st.error(
            f"No questions assigned. Check that `{GOLD_GLOB}` contains questions."
        )
        return

    if "slider_pos" not in st.session_state:
        first_unanswered = next(
            (i for i, qid in enumerate(assignment) if qid not in responses),
            0,
        )
        st.session_state["slider_pos"] = first_unanswered + 1  # slider is 1-indexed

    # Apply a pending jump requested by the Submit button on the previous run.
    # Must happen BEFORE the slider widget is instantiated, otherwise Streamlit
    # forbids modifying its session-state key.
    if "_pending_slider_pos" in st.session_state:
        st.session_state["slider_pos"] = st.session_state.pop("_pending_slider_pos")

    # Defensive clamp: slider_pos is 1-indexed into the CURRENT mode's question
    # list but persists in session state. Should it ever fall outside this mode's
    # range, assignment[q_pos] below would IndexError and the slider widget would
    # reject the out-of-range value. Snapping it back keeps the page from
    # crashing regardless of how the mode/length changed.
    if not 1 <= st.session_state.get("slider_pos", 1) <= total_qs:
        st.session_state["slider_pos"] = 1

    # ── Completion screen ────────────────────────────────────────────
    if done_count >= total_qs:
        rated = [responses[qid] for qid in assignment if qid in responses]
        avg = (
            sum(r["total_score"] for r in rated) / len(rated) if rated else 0
        )
        st.markdown(
            f'<div class="done-wrap"><div class="done-icon">✓</div>'
            f'<div class="done-title">Evaluation Complete</div>'
            f'<div class="done-sub">Thank you, {evaluator_name}. All {total_qs} '
            f"assigned questions have been rated.</div>"
            f'<div class="done-stat">{avg:.1f}<span>avg score / {MAX_TOTAL}</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Top header: mode badge on the left; submitted / remaining counters
    #    plus the evaluator name on the right. ──────────────────────────────
    # The slider's value equals its session-state key on every run, so we can
    # resolve the current question (and its submitted state) BEFORE drawing the
    # slider — that lets the status note live up in the header next to the
    # "Question X of Y" label.
    cur_pos = st.session_state["slider_pos"]
    q_pos = cur_pos - 1
    q_idx = assignment[q_pos]
    question = by_id[q_idx]
    existing = responses.get(q_idx, {})
    is_submitted = q_idx in responses

    status_note = (
        '<span class="slider-status submitted">✓ Already submitted — view only</span>'
        if is_submitted
        else '<span class="slider-status open">Awaiting your rating — score it below</span>'
    )

    # Question numbers (1-indexed, in assignment order) for the hover tooltips.
    submitted_nums = [i + 1 for i, qid in enumerate(assignment) if qid in responses]
    remaining_nums = [i + 1 for i, qid in enumerate(assignment) if qid not in responses]
    submitted_list = ", ".join(map(str, submitted_nums)) or "None yet"
    remaining_list = ", ".join(map(str, remaining_nums)) or "None — all done!"

    # Professional inline user glyph (no emoji).
    name_icon = (
        '<svg class="ev-name-ic" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2'
        "c-2.67 0-8 1.34-8 4v1.5c0 .28.22.5.5.5h15c.28 0 .5-.22.5-.5V18"
        'c0-2.66-5.33-4-8-4z"/></svg>'
    )

    submitted_n = done_count
    remaining_n = total_qs - done_count
    st.markdown(
        f'<div class="ev-top">'
        f'<div class="ev-top-left"></div>'
        f'<div class="ev-top-right">'
        f'<span class="stat-chip submitted">✓ {submitted_n} submitted'
        f'<span class="chip-tip"><span class="tip-head">Submitted questions</span>{submitted_list}</span>'
        f"</span>"
        f'<span class="stat-chip remaining">{remaining_n} remaining'
        f'<span class="chip-tip"><span class="tip-head">Remaining questions</span>{remaining_list}</span>'
        f"</span>"
        f'<span class="ev-eval-name">{name_icon}{evaluator_name}</span>'
        f"</div>"
        f"</div>"
        f'<div class="ev-qpos-row">'
        f'<span class="ev-qpos">Question <strong>{cur_pos}</strong> of {total_qs}</span>'
        f"{status_note}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Back / Next: step one question at a time. Callbacks run before the
    #    next script run, so updating the slider's session-state key here is
    #    the safe way to move it without "modified after instantiation" errors.
    def _go_prev():
        st.session_state["slider_pos"] = max(1, st.session_state["slider_pos"] - 1)

    def _go_next():
        st.session_state["slider_pos"] = min(
            total_qs, st.session_state["slider_pos"] + 1
        )

    # ── Slider flanked by Back / Next icon buttons — the single progress
    #    element (replaces the old dot row + prog-track bar). ───────────────
    with st.container(key="question_slider_row"):
        c_back, c_slider, c_next = st.columns(
            [1, 14, 1], gap="small", vertical_alignment="center"
        )
        with c_back:
            st.button(
                "←",
                key="nav_back",
                on_click=_go_prev,
                disabled=cur_pos <= 1,
                use_container_width=True,
            )
        with c_slider:
            # The slider writes its value to st.session_state["slider_pos"] via
            # its key; the return value is unused (read back from session state).
            st.slider(
                "Question navigator",
                min_value=1,
                max_value=total_qs,
                value=st.session_state["slider_pos"],
                key="slider_pos",
                label_visibility="collapsed",
            )
        with c_next:
            st.button(
                "→",
                key="nav_next",
                on_click=_go_next,
                disabled=cur_pos >= total_qs,
                use_container_width=True,
            )
    # Drives the custom gradient fill bar from the live thumb position.
    st_html(SLIDER_FILL_JS, height=0)

    # ── Question + answers ───────────────────────────────────────────
    category = question.get("category", "").replace("_", " ").title()
    source = question.get("source", "unknown")
    source_label = (
        "Human-Created"
        if source == "human"
        else ("AI-Created" if source == "ai" else "Unknown")
    )
    # Escape every model/data-driven string before it goes into an
    # unsafe_allow_html block: a stray "<", ">" or "&" in an answer would
    # otherwise be parsed as markup (broken layout at best, injection at worst).
    # html.escape leaves newlines intact, so the answer blocks' white-space:
    # pre-wrap still renders multi-line lists correctly.
    # Left pane ("Context") shows the full source chunk the answer was drawn
    # from (static). The gold `context` comes from load_questions.
    # The panes render with white-space: pre-wrap, so the source's ragged
    # newlines (mixed single \n and runs of \n\n\n between list items) would
    # show as uneven blank gaps. collapse_blank_lines() trims per-line
    # trailing space and squashes any run of blank lines to a single break,
    # so lists render compactly and evenly.
    context = html.escape(collapse_blank_lines(question.get("context") or ""))
    context_body = context if context else "—"

    # Right pane ("Target Answer") is EDITABLE and SHARED across evaluators.
    # The effective target is the most recent shared edit (from target_edits),
    # falling back to the gold expected_answer when nobody has edited it yet.
    override = get_target_override(q_idx)
    gold_target = collapse_blank_lines(question.get("expected_answer") or "")
    effective_target = (
        override["new_answer"] if override else gold_target
    )
    edited_flag = (
        '<span class="ev-target-edited-flag">edited</span>' if override else ""
    )

    st.markdown(
        f'<div class="ev-pill-row">'
        f'<span class="ev-category-pill">{category}</span>'
        f'<span class="ev-source-pill {source}">{source_label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ev-question-text">{html.escape(question["question"])}</div>',
        unsafe_allow_html=True,
    )

    # Two side-by-side columns: static Context (left), editable Target (right).
    # Streamlit widgets can't live inside a raw-HTML grid, so real columns
    # replace the old .ev-answer-grid markup here.
    col_ctx, col_tgt = st.columns(2, gap="medium")
    with col_ctx:
        st.markdown(
            '<div class="ev-answer-label expected"><span class="dot"></span>Context</div>'
            f'<div class="ev-answer-block expected">{context_body}</div>',
            unsafe_allow_html=True,
        )
    with col_tgt:
        st.markdown(
            '<div class="ev-answer-label generated"><span class="dot"></span>'
            f"Target Answer{edited_flag}"
            "</div>",
            unsafe_allow_html=True,
        )
        # Seed the editor once per question visit; keyed by question_id so the
        # slider revisiting a question doesn't reset an in-progress edit. We do
        # NOT reseed on every run (that would clobber typing), so a concurrent
        # edit by someone else surfaces via the log below rather than live.
        target_key = f"target_edit_{q_idx}"
        if target_key not in st.session_state:
            st.session_state[target_key] = effective_target
        # The box auto-grows to fit its content exactly (no gap, no inner
        # scroll) via TARGET_AUTOGROW_JS below — it grows as the evaluator
        # types too. The height= here is just a pre-JS fallback minimum.
        with st.container(key="target_editor_box"):
            st.text_area(
                "target_answer_editor",
                key=target_key,
                label_visibility="collapsed",
                height=120,
                disabled=is_submitted,
            )
        st_html(TARGET_AUTOGROW_JS, height=0)

    # ── Rating cards (5 in a row) ─────────────────────────────────────
    current_scores: dict = {}
    all_scored = True

    # Seed session state from DB data only on first visit to this question.
    # Keyed by q_idx (question_id) rather than q_pos, since the slider lets
    # the same question be revisited at different times — q_pos alone would
    # not reliably identify "have we already seeded this question's state".
    for metric in METRICS:
        sk = f"eval_{metric['key']}_{q_idx}"
        if sk not in st.session_state:
            st.session_state[sk] = existing.get(metric["key"])

    _metric_colors = ["#1D6A9F", "#2E7D52", "#6B41A0", "#B68B4C", "#1B7A8C"]
    card_defs = [
        {
            "key": m["key"],
            "label": m["label"],
            "desc": m["desc"],
            "options": m["options"],
            "title_color": c,
        }
        for m, c in zip(METRICS, _metric_colors)
    ]

    # Per-card theming: key class sits ON stVerticalBlockBorderWrapper, so
    # ".st-key-card_X [...]" is a valid descendant selector for buttons inside.
    # Injected after main CSS → wins on same-specificity cascade.
    _card_css = "<style>"
    for _cd in card_defs:
        _c = _cd["title_color"]
        _k = _cd["key"]
        _card_css += f"""
.st-key-metric_cards_row .st-key-card_{_k} {{
    background:
        linear-gradient(var(--parchment-soft), var(--parchment-soft)) padding-box,
        linear-gradient(160deg, {_c}44, {_c}cc) border-box !important;
    border: 1.5px solid transparent !important;
}}
.st-key-metric_cards_row .st-key-card_{_k}:hover {{
    box-shadow: 0 4px 14px {_c}33 !important;
}}
.st-key-card_{_k} [data-testid="stBaseButton-primary"] {{
    background: {_c} !important;
    border: 1px solid {_c} !important;
    color: var(--parchment-soft) !important;
}}
.st-key-card_{_k} [data-testid="stBaseButton-primary"]:hover {{
    background: {_c}dd !important;
    border-color: {_c}dd !important;
}}
.st-key-card_{_k} [data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: 1px solid {_c}55 !important;
    color: var(--stone-light) !important;
}}
.st-key-card_{_k} [data-testid="stBaseButton-secondary"]:hover {{
    background: {_c}22 !important;
    border-color: {_c}aa !important;
    color: var(--stone) !important;
}}
/* View-only (submitted) state: disabled buttons. These selectors carry more
   specificity than the global `button:disabled` rule, so the chosen option
   keeps its full card colour and the rest stay clean faint outlines —
   instead of everything turning washed-out yellow. */
.st-key-metric_cards_row .st-key-card_{_k} [data-testid="stBaseButton-primary"]:disabled {{
    background: {_c} !important;
    border: 1px solid {_c} !important;
    color: var(--parchment-soft) !important;
    opacity: 1 !important;
    cursor: default !important;
}}
.st-key-metric_cards_row .st-key-card_{_k} [data-testid="stBaseButton-secondary"]:disabled {{
    background: transparent !important;
    border: 1px solid {_c}33 !important;
    color: var(--stone-light) !important;
    opacity: 0.7 !important;
    cursor: default !important;
}}
"""
    _card_css += "</style>"
    st.markdown(_card_css, unsafe_allow_html=True)

    with st.container(key="metric_cards_row"):
        cols = st.columns(5, gap="small")
        for col, card in zip(cols, card_defs):
            with col:
                with st.container(border=True, key=f"card_{card['key']}"):
                    st.markdown(
                        f'<p class="metric-card-title" style="color:{card["title_color"]};">{card["label"]}'
                        f'<span class="metric-tooltip">{card["desc"]}</span>'
                        f"</p>",
                        unsafe_allow_html=True,
                    )
                    sk = f"eval_{card['key']}_{q_idx}"
                    for i, opt in enumerate(card["options"]):
                        val = i + 1
                        is_sel = st.session_state.get(sk) == val
                        if is_submitted:
                            # Locked view: render as a disabled button so the
                            # selected value is still visually obvious (primary
                            # styling), but nothing is clickable.
                            st.button(
                                opt,
                                key=f"eval_locked_{card['key']}_{i}_{q_idx}",
                                type="primary" if is_sel else "secondary",
                                use_container_width=True,
                                disabled=True,
                            )
                        else:
                            if st.button(
                                opt,
                                key=f"eval_btn_{card['key']}_{i}_{q_idx}",
                                type="primary" if is_sel else "secondary",
                                use_container_width=True,
                            ):
                                st.session_state[sk] = val
                                st.rerun()

    # Collect scores from session state
    for metric in METRICS:
        val = st.session_state.get(f"eval_{metric['key']}_{q_idx}")
        current_scores[metric["key"]] = val
        if val is None:
            all_scored = False

    # ── Target-answer edit log (shared, replaces the old comment box) ──────
    # Shows who last changed this question's target answer and how long ago,
    # so evaluators see each other's curation activity.
    st.markdown(
        '<div class="ev-comment-label">Target answer edit history</div>',
        unsafe_allow_html=True,
    )
    history = get_target_history(q_idx)
    if history:
        items = "".join(
            f'<li class="ev-log-item"><span class="ev-log-dot"></span>'
            f'Edited <span class="ev-log-when">{humanize_time_ago(h["edited_at"])}</span> '
            f'by <span class="ev-log-who">{html.escape(h["evaluator_name"])}</span></li>'
            for h in history
        )
        st.markdown(f'<ul class="ev-log-list">{items}</ul>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="ev-log-empty">No edits yet — the target answer above is the '
            "original golden answer. Edit and save it to start a history.</div>",
            unsafe_allow_html=True,
        )

    # ── Footer: score bar + submit (or locked notice) ──────────────────
    metric_sum = sum(v for v in current_scores.values() if v is not None)
    bar_pct = int(metric_sum / MAX_TOTAL * 100)

    st.markdown(
        '<hr style="border:none;border-top:1px dashed var(--brass-soft);margin:16px 0;">',
        unsafe_allow_html=True,
    )

    if is_submitted:
        st.markdown(
            f'<div class="score-row">'
            f'<div class="score-label">Score <strong>{metric_sum}</strong> / {MAX_TOTAL}</div>'
            f'<div class="score-track"><div class="score-fill" style="width:{bar_pct}%"></div></div>'
            f"</div>"
            f'<p class="ev-incomplete-note">This question was already submitted — drag the slider to '
            f"an unanswered question to continue rating.</p>",
            unsafe_allow_html=True,
        )
    else:
        bar_col, btn_col = st.columns([5, 1], gap="large")
        with bar_col:
            st.markdown(
                f'<div class="score-row">'
                f'<div class="score-label">Score <strong>{metric_sum}</strong> / {MAX_TOTAL}</div>'
                f'<div class="score-track"><div class="score-fill" style="width:{bar_pct}%"></div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with btn_col:
            is_last_remaining = (done_count + 1) >= total_qs
            btn_label = "Submit & Finish" if is_last_remaining else "Submit & Next →"
            if st.button(btn_label, disabled=not all_scored):
                final_scores = {
                    k: v for k, v in current_scores.items() if v is not None
                }
                # Comment field was replaced by the shared target-answer editor;
                # responses.comment is retained in the schema but no longer used.
                save_response(
                    evaluator_id,
                    q_idx,
                    final_scores,
                    "",
                )
                # Persist any target-answer edit as part of the same submit —
                # there's no separate save button. Only writes when the text
                # actually changed from the current effective target.
                edited_target = st.session_state.get(target_key, "").strip()
                if edited_target and edited_target != effective_target.strip():
                    save_target_edit(
                        q_idx, evaluator_id, evaluator_name, edited_target
                    )
                # Advance to the next unanswered question after this one in
                # assignment order; wrap to the first unanswered overall if
                # none remain after the current position (out-of-order rating
                # via the slider means "the next one" isn't always q_pos + 1).
                updated_responses = get_responses(evaluator_id)
                next_pos = next(
                    (
                        i
                        for i in range(q_pos + 1, total_qs)
                        if assignment[i] not in updated_responses
                    ),
                    None,
                )
                if next_pos is None:
                    next_pos = next(
                        (
                            i
                            for i in range(total_qs)
                            if assignment[i] not in updated_responses
                        ),
                        total_qs - 1,
                    )
                st.session_state["_pending_slider_pos"] = next_pos + 1
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
    by_id, order = load_questions()
    if not by_id:
        st.markdown(CSS, unsafe_allow_html=True)
        st.error(f"No gold questions found at `{GOLD_GLOB}`.")
        return

    if "evaluator_name" not in st.session_state:
        page_login()
        return

    name = st.session_state["evaluator_name"]

    if "evaluator_id" not in st.session_state:
        eid, sid, resumed = get_or_create_evaluator(name)
        st.session_state["evaluator_id"] = eid
        st.session_state["session_id"] = sid
        if resumed:
            st.toast(f"Welcome back, {name} — resuming your saved session.", icon="👋")
    evaluator_id = st.session_state["evaluator_id"]

    # Draw (or resume) this evaluator's fixed random slice of the full 400.
    assignment = get_or_create_assignment(evaluator_id, order)
    page_evaluate(evaluator_id, name, assignment, by_id)


if __name__ == "__main__":
    main()
