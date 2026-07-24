#!/usr/bin/env python3
# ============================================================================
# 16_prepare_llm_judge_input.py
#
# PURPOSE: flatten each generator model's answers into ONE self-contained JSON
# file that an LLM judge can score without touching any other file.
#
# scripts/13 stores only `context_chunk_ids` in its JSONL -- the retrieved text
# itself lives in data/08_merged_chunked_documents/*.json, and a single answer's
# context routinely spans more than one category file. A judge (human or LLM)
# cannot score Context Relevance or Faithfulness without that text, so this
# script resolves the ids once, here, instead of making the judge do a 4-file
# join per record.
#
# INPUT
#   reports/generated_answers/<model>/generated_answer_results.jsonl
#   data/08_merged_chunked_documents/*.json          (chunk_id -> content)
#
# OUTPUT (under reports/llm_judge_evaluation/input/)
#   <model>.json    100 records: question, context, expected, generated
#
# USAGE:
#   python scripts/16_prepare_llm_judge_input.py
# ============================================================================

import json
from glob import glob
from pathlib import Path

ANSWERS_DIR = Path("reports/generated_answers")
CHUNK_GLOB = "data/08_merged_chunked_documents/*.json"
OUTPUT_DIR = Path("reports/llm_judge_evaluation/input")


def load_chunks():
    """chunk_id -> content, merged across every category file."""
    chunks = {}
    for path in sorted(glob(CHUNK_GLOB)):
        for entry in json.loads(Path(path).read_text(encoding="utf-8")):
            cid = entry.get("chunk_id")
            if cid:
                chunks[cid] = entry.get("content", "")
    return chunks


def build(model_dir, chunks):
    jsonl = model_dir / "generated_answer_results.jsonl"
    if not jsonl.exists():
        return None

    records, missing = [], set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)

        # keep the retrieved chunks separate and labelled -- the judge needs to
        # be able to point at WHICH chunk grounds (or fails to ground) a claim
        context = []
        for cid in r.get("context_chunk_ids", []):
            if cid in chunks:
                context.append({"chunk_id": cid, "text": chunks[cid]})
            else:
                missing.add(cid)

        records.append({
            "question_id":      r["question_id"],
            "category":         r.get("category"),
            "question":         r.get("question"),
            "retrieved_context": context,
            "expected_answer":  r.get("expected_answer"),
            "generated_answer": r.get("generated_answer"),
        })

    return records, missing


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNK_GLOB}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for model_dir in sorted(p for p in ANSWERS_DIR.iterdir() if p.is_dir()):
        built = build(model_dir, chunks)
        if built is None:
            continue
        records, missing = built

        out = OUTPUT_DIR / f"{model_dir.name}.json"
        out.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total += len(records)

        empty = sum(1 for r in records if not r["retrieved_context"])
        note = f"  ({len(missing)} unresolved chunk ids)" if missing else ""
        note += f"  ({empty} records with EMPTY context)" if empty else ""
        print(f"  {model_dir.name:38s} {len(records):3d} records -> {out}{note}")

    print(f"\nTotal records to judge: {total}")


if __name__ == "__main__":
    main()
