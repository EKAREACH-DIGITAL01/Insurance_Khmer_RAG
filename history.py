"""Persistent chat history for the assistant's sidebar.

The chat UI keeps the *working* conversation in st.session_state["messages"];
this module is what makes that survive a browser refresh or a server restart.
Every message is written through as soon as it lands in session state, so a
crash mid-session loses at most the answer that was still streaming.

SQLite (stdlib, no new dependency) rather than one JSON file per conversation:
renaming, deleting, and ordering by "most recently used" are single statements,
and a half-finished write can't leave a conversation file truncated. Same
connection settings as the human-evaluation app in
scripts/11_human_golden_question_evaluation.py -- WAL plus a busy timeout, so
two browser tabs pointed at the same server don't surface "database is locked".
"""

import json
import os
import sqlite3
import time
import uuid

import numpy as np

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "chat_history.db"
)

# A conversation's title is derived from its first question, which is Khmer
# and often a full sentence. Long enough to stay distinguishable in the
# sidebar, short enough that the CSS ellipsis is the exception, not the rule.
TITLE_MAX_CHARS = 48


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")  # makes the ON DELETE CASCADE real
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL
                            REFERENCES conversations(id) ON DELETE CASCADE,
            position        INTEGER NOT NULL,
            payload         TEXT NOT NULL,
            created_at      INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, position);
        """
    )
    conn.commit()
    conn.close()


def derive_title(question):
    """Turns the opening question into a sidebar label."""
    title = " ".join(str(question).split())
    if not title:
        return "សំណួរថ្មី"
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title


def list_conversations():
    """Most recently used first -- the order the sidebar renders in."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_conversation(title):
    conv_id = str(uuid.uuid4())
    now = int(time.time())
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return conv_id


def rename_conversation(conv_id, title):
    title = " ".join(str(title).split())
    if not title:
        return
    conn = get_db()
    # Deliberately does NOT touch updated_at: renaming is housekeeping, not
    # use, and reordering the sidebar under the user's cursor mid-rename
    # would make the row they just edited jump somewhere else.
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (title[:TITLE_MAX_CHARS], conv_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def load_messages(conv_id):
    """Rehydrates a conversation into the exact shape render_message() and the
    diagnostics page expect. The full message dict is round-tripped through
    JSON, diagnostics payload included, so reopening an old conversation still
    has working "មើលព័ត៌មានលម្អិត" links rather than dead buttons."""
    conn = get_db()
    rows = conn.execute(
        "SELECT payload FROM messages WHERE conversation_id = ? ORDER BY position",
        (conv_id,),
    ).fetchall()
    conn.close()
    return [json.loads(r["payload"]) for r in rows]


def _json_default(obj):
    """Retrieval scores arrive as numpy scalars (np.float32 from FAISS, np.int64
    from index positions), which json.dumps rejects outright. Everything the
    app does with them afterwards is formatting and comparison, so they're
    written as plain Python numbers and come back as floats/ints."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def load_message(msg_id):
    """Fetches one message by its id, without needing its conversation.

    This is what makes a diagnostics URL work on its own: the page is reached
    by message id straight from the address bar, in a session that may never
    have opened -- or may no longer contain -- the conversation it belongs to.
    Returns None if the id isn't in the database."""
    conn = get_db()
    row = conn.execute(
        "SELECT payload FROM messages WHERE id = ?", (str(msg_id),)
    ).fetchone()
    conn.close()
    return json.loads(row["payload"]) if row else None


def append_message(conv_id, msg):
    """Persists one message and marks its conversation as most recently used.

    JSON can't represent the (score, chunk) tuples in used_sources/all_results,
    so they come back as [score, chunk] lists. Every consumer unpacks them
    positionally (`for score, chunk in ...`), which works identically on both.
    """
    now = int(time.time())
    conn = get_db()
    # The next position is computed INSIDE the insert rather than by a separate
    # SELECT MAX(...) first. Two statements leave a gap in which another writer
    # can read the same maximum and insert the same position -- two tabs open on
    # one conversation is enough, and duplicate positions make ORDER BY position
    # ambiguous, so a reloaded conversation can come back out of order.
    conn.execute(
        "INSERT INTO messages (id, conversation_id, position, payload, created_at) "
        "SELECT ?, ?, COALESCE(MAX(position), -1) + 1, ?, ? "
        "FROM messages WHERE conversation_id = ?",
        (
            msg["id"],
            conv_id,
            json.dumps(msg, ensure_ascii=False, default=_json_default),
            now,
            conv_id,
        ),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
    )
    conn.commit()
    conn.close()
