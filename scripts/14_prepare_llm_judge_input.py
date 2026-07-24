#!/usr/bin/env python3
# ============================================================================
# 14_prepare_llm_judge_input.py
#
# PURPOSE: get each model's generated_answer_results.jsonl ready for an LLM
# judge, and score the one metric that needs no judgement at all.
#
# Run this once after scripts/13 finishes. Per record it:
#
#   1. Scores context_relevance DETERMINISTICALLY from the chunk ids -- what
#      fraction of expected_chunks the retriever actually returned in
#      context_chunk_ids. This is an exact set comparison; an LLM cannot do it
#      better and in practice just returns 5 every time. It is retrieval
#      quality, and it is measured, not judged.
#   2. Resets the four metrics an LLM must judge to 0, and hallucination /
#      hallucination_evidence to "". 0 and "" mean NOT YET JUDGED.
#   3. Drops `detail` (internals of the deleted proxy scorer) and
#      `retrieved_context` if a previous run inlined it -- the judge compares
#      generated_answer against expected_answer, so the chunk TEXT is not
#      needed, and its bulk is what made the file unwieldy.
#
# The judge then only ever needs the one JSONL per model, ~1.6k chars/record.
#
# INPUT / OUTPUT (rewritten in place)
#   reports/generator_model_comparison/<model>/generated_answer_results.jsonl
#
# USAGE:
#   python scripts/14_prepare_llm_judge_input.py
#   python scripts/14_prepare_llm_judge_input.py --dry-run   # report, no writes
# ============================================================================

import argparse
import json
from pathlib import Path

ANSWERS_DIR = Path("reports/generator_model_comparison")

# Metrics the LLM judge fills in. context_relevance is NOT here -- it is
# computed below.
JUDGED_METRICS = [
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
    "completeness",
]


def score_context_relevance(expected_chunks, context_chunk_ids):
    """Recall of the expected chunks -> 1-5.

    Recall (not precision) is the right measure here: extra retrieved chunks
    are cheap, whereas a MISSING expected chunk means the answer could not have
    been grounded properly no matter how good the generator is.
    """
    expected = {c for c in (expected_chunks or []) if c}
    if not expected:
        return None  # nothing to measure against
    got = expected & set(context_chunk_ids or [])
    recall = len(got) / len(expected)
    if recall == 1.0:
        return 5
    if recall >= 0.75:
        return 4
    if recall >= 0.50:
        return 3
    if recall > 0:
        return 2
    return 1


def prepare(r):
    ctx = score_context_relevance(r.get("expected_chunks"),
                                  r.get("context_chunk_ids"))
    scores = {"context_relevance": ctx if ctx is not None else 0}
    scores.update({m: 0 for m in JUDGED_METRICS})

    return {
        "question_id":       r.get("question_id"),
        "category":          r.get("category"),
        "question":          r.get("question"),
        "expected_chunks":   r.get("expected_chunks"),
        "context_chunk_ids": r.get("context_chunk_ids"),
        "expected_answer":   r.get("expected_answer"),
        "generated_answer":  r.get("generated_answer"),
        "gate_reason":       r.get("gate_reason"),
        # context_relevance is already measured; the rest await the judge
        "scores":                 scores,
        "combined":               0,
        "hallucination":          "",
        "hallucination_evidence": "",
        "timing":                 r.get("timing"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything.")
    args = ap.parse_args()

    if not ANSWERS_DIR.exists():
        raise SystemExit(f"No {ANSWERS_DIR}/ -- run scripts/13 first.")

    total = 0
    for model_dir in sorted(p for p in ANSWERS_DIR.iterdir() if p.is_dir()):
        path = model_dir / "generated_answer_results.jsonl"
        if not path.exists():
            continue

        records = [prepare(json.loads(line))
                   for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]

        if not args.dry_run:
            # utf-8 in AND out -- Khmer must survive the round trip byte-exact
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ctx = [r["scores"]["context_relevance"] for r in records]
        perfect = sum(1 for c in ctx if c == 5)
        poor    = sum(1 for c in ctx if c and c <= 2)
        size    = sum(len(json.dumps(r, ensure_ascii=False)) for r in records)
        total  += len(records)

        print(f"  {model_dir.name:38s} {len(records):3d} records | "
              f"context_relevance: {perfect} perfect / {poor} poor | "
              f"{size:,} chars")

    verb = "would be prepared" if args.dry_run else "ready to judge"
    print(f"\n{total} records {verb}.")
    if not args.dry_run:
        print("context_relevance is already scored (measured from chunk ids).")
        print("faithfulness / answer_relevance / answer_correctness / "
              "completeness are 0")
        print("and hallucination is \"\", until the LLM judge fills them in.")
        print("\nNext: follow docs/llm_judge_prompt.md")


if __name__ == "__main__":
    main()
