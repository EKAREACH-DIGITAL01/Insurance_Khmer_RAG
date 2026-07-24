#!/usr/bin/env python3
# ============================================================================
# 16_prepare_llm_judge_input.py
#
# PURPOSE: make each model's generated_answer_results.jsonl SELF-CONTAINED and
# ready for an LLM judge to score in place.
#
# Run this once after scripts/13 finishes. It does three things per record:
#
#   1. Resolves `context_chunk_ids` into the actual chunk TEXT and stores it as
#      `retrieved_context`. This is the whole reason this script exists: the
#      JSONL only records chunk ids, the text lives in
#      data/08_merged_chunked_documents/*.json, and one answer's context
#      routinely spans several category files. Without the text a judge cannot
#      score context_relevance or faithfulness at all.
#   2. Resets every RAGAS metric to 0 and adds `hallucination: ""`. 0 and ""
#      mean NOT YET JUDGED -- the LLM judge overwrites them.
#   3. Drops `detail`, the internals of the deleted proxy scorer.
#
# After this, an LLM judge only ever needs the one JSONL per model.
#
# INPUT / OUTPUT (rewritten in place)
#   reports/generated_answers/<model>/generated_answer_results.jsonl
#   data/08_merged_chunked_documents/*.json          (chunk_id -> content)
#
# USAGE:
#   python scripts/16_prepare_llm_judge_input.py
#   python scripts/16_prepare_llm_judge_input.py --dry-run   # report, no writes
# ============================================================================

import argparse
import json
from glob import glob
from pathlib import Path

ANSWERS_DIR = Path("reports/generated_answers")
CHUNK_GLOB = "data/08_merged_chunked_documents/*.json"

# The five RAGAS metrics the judge fills in. 0 = not yet judged.
METRICS = [
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
    "completeness",
]


def load_chunks():
    """chunk_id -> content, merged across every category file."""
    chunks = {}
    for path in sorted(glob(CHUNK_GLOB)):
        for entry in json.loads(Path(path).read_text(encoding="utf-8")):
            cid = entry.get("chunk_id")
            if cid:
                chunks[cid] = entry.get("content", "")
    return chunks


def prepare(record, chunks, missing):
    """Returns the record rewritten for judging (mutates nothing in place)."""
    context = []
    for cid in record.get("context_chunk_ids", []):
        if cid in chunks:
            context.append({"chunk_id": cid, "text": chunks[cid]})
        else:
            missing.add(cid)

    out = {
        "question_id":       record.get("question_id"),
        "category":          record.get("category"),
        "question":          record.get("question"),
        "expected_chunks":   record.get("expected_chunks"),
        "context_chunk_ids": record.get("context_chunk_ids"),
        "retrieved_context": context,
        "expected_answer":   record.get("expected_answer"),
        "generated_answer":  record.get("generated_answer"),
        "gate_reason":       record.get("gate_reason"),
        # --- judged by the LLM: 0 / "" mean NOT YET JUDGED ---
        "scores":            {m: 0 for m in METRICS},
        "combined":          0,
        "hallucination":     "",
        "timing":            record.get("timing"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything.")
    args = ap.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNK_GLOB}\n")

    if not ANSWERS_DIR.exists():
        raise SystemExit(f"No {ANSWERS_DIR}/ -- run scripts/13 first.")

    total = 0
    for model_dir in sorted(p for p in ANSWERS_DIR.iterdir() if p.is_dir()):
        path = model_dir / "generated_answer_results.jsonl"
        if not path.exists():
            continue

        missing = set()
        records = [
            prepare(json.loads(line), chunks, missing)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        empty = sum(1 for r in records if not r["retrieved_context"])
        note = ""
        if missing:
            note += f"  ({len(missing)} unresolved chunk ids)"
        if empty:
            note += f"  ({empty} records with EMPTY context)"

        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total += len(records)
        print(f"  {model_dir.name:38s} {len(records):3d} records{note}")

    verb = "would be prepared" if args.dry_run else "ready to judge"
    print(f"\n{total} records {verb}.")
    if not args.dry_run:
        print("Scores are 0 and hallucination is \"\" until the LLM judge fills them in.")
        print("Next: follow docs/llm_judge_prompt.md")


if __name__ == "__main__":
    main()
