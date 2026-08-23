#!/usr/bin/env python3
"""Export answer feedback from the Streamlit chat history database.

Use after a demo or testing session:

    python scripts/18_export_chat_feedback.py

The app stores the current thumb state inside each message payload and keeps
every click in feedback_events. This exporter writes the latest useful state
for review, dataset fixes, or future evaluation work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "chat_history.db"
DEFAULT_OUT_DIR = ROOT / "reports" / "chat_feedback"


def ts_to_local(ts: int | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ts)))


def compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def source_fields(payload: dict) -> tuple[str, str]:
    names = []
    urls = []
    for item in payload.get("used_sources") or []:
        try:
            _score, chunk = item
        except (TypeError, ValueError):
            continue
        metadata = (chunk or {}).get("metadata", {})
        name = metadata.get("document_name")
        url = metadata.get("source_url")
        if name and name not in names:
            names.append(name)
        if url and url not in urls:
            urls.append(url)
    return " | ".join(names), " | ".join(urls)


def load_rows(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH latest_feedback AS (
                SELECT e.message_id, e.value, e.created_at
                FROM feedback_events e
                JOIN (
                    SELECT message_id, MAX(created_at) AS created_at
                    FROM feedback_events
                    GROUP BY message_id
                ) last
                  ON last.message_id = e.message_id
                 AND last.created_at = e.created_at
            )
            SELECT
                m.id AS message_id,
                m.conversation_id,
                c.title AS conversation_title,
                m.created_at AS message_created_at,
                m.payload,
                latest_feedback.value AS event_feedback,
                latest_feedback.created_at AS feedback_created_at,
                (
                    SELECT COUNT(*)
                    FROM feedback_events e
                    WHERE e.message_id = m.id
                ) AS feedback_clicks
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            LEFT JOIN latest_feedback ON latest_feedback.message_id = m.id
            ORDER BY COALESCE(latest_feedback.created_at, m.created_at), m.position
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise
    finally:
        conn.close()

    exported = []
    for row in rows:
        payload = json.loads(row["payload"])
        feedback = payload.get("feedback") or row["event_feedback"]
        if feedback not in {"up", "down"}:
            continue
        source_names, source_urls = source_fields(payload)
        exported.append(
            {
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "message_created_at": ts_to_local(row["message_created_at"]),
                "feedback": feedback,
                "feedback_created_at": ts_to_local(
                    payload.get("feedback_updated_at") or row["feedback_created_at"]
                ),
                "feedback_clicks": row["feedback_clicks"],
                "question": compact_text(payload.get("question")),
                "answer": compact_text(payload.get("answer")),
                "best_score": payload.get("best_score", ""),
                "low_confidence": bool(payload.get("low_confidence")),
                "out_of_domain": bool(payload.get("out_of_domain")),
                "source_names": source_names,
                "source_urls": source_urls,
            }
        )
    return exported


def write_outputs(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "chat_feedback.csv"
    jsonl_path = out_dir / "chat_feedback.jsonl"

    fieldnames = list(rows[0].keys()) if rows else [
        "message_id",
        "conversation_id",
        "conversation_title",
        "message_created_at",
        "feedback",
        "feedback_created_at",
        "feedback_clicks",
        "question",
        "answer",
        "best_score",
        "low_confidence",
        "out_of_domain",
        "source_names",
        "source_urls",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.db)
    csv_path, jsonl_path = write_outputs(rows, args.out_dir)
    print(f"Exported {len(rows)} feedback row(s).")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
