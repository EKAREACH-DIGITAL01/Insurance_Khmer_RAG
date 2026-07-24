#!/usr/bin/env python3
# ============================================================================
# check_judge_progress.py
#
# Verify one model's JSONL after an LLM-judge batch. Not a pipeline stage --
# a guard you run between batches, which is why it has no number.
#
# The failure that matters is SILENT TRUNCATION: a batch that looks scored but
# whose later records were never read. `records` must stay 100 and `scored`
# must climb by exactly the batch size.
#
# USAGE:
#   python scripts/check_judge_progress.py <path-to-generated_answer_results.jsonl>
# ============================================================================

import json, sys
p = sys.argv[1]
rs = [json.loads(l) for l in open(p, encoding="utf-8")]
print("records      :", len(rs), "(must be 100)")
print("scored       :", sum(1 for r in rs if any(r["scores"].values())))
print("unscored     :", sum(1 for r in rs if not any(r["scores"].values())))
print("hallucinated :", sum(1 for r in rs if r.get("hallucination") == "true"))
bad = [r["question_id"] for r in rs
       if any(v and not 1 <= v <= 5 for v in r["scores"].values())]
print("out of range :", bad or "none")
noev = [r["question_id"] for r in rs
        if r.get("hallucination") == "true" and not r.get("hallucination_evidence")]
print("flagged w/o evidence:", noev or "none")
