#!/usr/bin/env python3
# ============================================================================
# check_judge_progress.py
#
# Verify one model's JSONL after an LLM-judge run. Not a pipeline stage -- a
# guard you run after judging, which is why it has no number.
#
# Two failures this catches:
#   - SILENT TRUNCATION: records missing, or left unscored, while the judge
#     reports success.
#   - MOJIBAKE: Khmer UTF-8 read as Latin-1 and re-encoded, which happens when
#     a judge rewrites the file without encoding="utf-8". The Khmer turns into
#     byte soup and is unrecoverable except from git.
#
# USAGE:
#   python scripts/check_judge_progress.py <path-to-generated_answer_results.jsonl>
# ============================================================================

import json
import sys

JUDGED = ["faithfulness", "answer_relevance", "answer_correctness", "completeness"]

# Khmer is U+1780-U+17FF, whose UTF-8 lead byte is 0xE1. Mis-decoded as
# Latin-1 that byte becomes U+00E1. Real Khmer text never contains it.
MOJIBAKE_MARKER = "á"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_judge_progress.py <generated_answer_results.jsonl>")
    path = sys.argv[1]

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"records      : {len(rows)}  (must be 100)")

    judged = [r for r in rows if all(r["scores"].get(m) for m in JUDGED)]
    print(f"judged       : {len(judged)}")
    print(f"unjudged     : {len(rows) - len(judged)}")
    print(f"hallucinated : {sum(1 for r in rows if r.get('hallucination') == 'true')}")

    problems = []

    bad = [r["question_id"] for r in rows
           if any(v and not 1 <= v <= 5 for v in r["scores"].values())]
    if bad:
        problems.append(f"scores outside 1-5: {bad}")

    dec = [r["question_id"] for r in rows
           if any(isinstance(v, float) for v in r["scores"].values())]
    if dec:
        problems.append(f"non-integer metric scores: {dec}")

    hal = [r["question_id"] for r in rows
           if r.get("hallucination") not in ("true", "false", "")]
    if hal:
        problems.append(f'hallucination not "true"/"false": {hal}')

    noev = [r["question_id"] for r in rows
            if r.get("hallucination") == "true" and not r.get("hallucination_evidence")]
    if noev:
        problems.append(f"flagged without evidence: {noev}")

    # the failure a record count cannot catch
    corrupt = [r["question_id"] for r in rows
               if MOJIBAKE_MARKER in json.dumps(r, ensure_ascii=False)]
    if corrupt:
        more = " ..." if len(corrupt) > 5 else ""
        problems.append(
            f"MOJIBAKE -- Khmer corrupted in {len(corrupt)} record(s): "
            f"{corrupt[:5]}{more}\n"
            f"    The judge rewrote the file with the wrong encoding. The text "
            f"cannot be repaired from this file.\n"
            f"    Restore it:  git checkout {path}"
        )

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("OK -- no format or encoding problems.")


if __name__ == "__main__":
    main()
