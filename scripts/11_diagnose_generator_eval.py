import json
from collections import Counter
from pathlib import Path

# ============================================================================
# Generator Evaluation Diagnostics
# ============================================================================
# Permanent home for the diagnostic checks used to calibrate
# scripts/10_compare_generator_models.py's scoring thresholds (see
# docs/SCORING_CALIBRATION.md for the full writeup of what each check
# found and why it mattered). Re-run this any time the calibration needs
# re-checking -- e.g. after adding new gold questions, swapping in a
# different embedding model, or adding a 5th generator model.
#
# Reads ONLY from reports/generator_model_comparison/<model>/generated_answer_results.jsonl --
# never touches llama-server, never re-generates anything. Pure
# read-and-report over data scripts/10 already produced.
#
# USAGE:
#   python scripts/diagnose_generator_eval.py              # all checks
#   python scripts/diagnose_generator_eval.py --check NAME # one check only
#
# Available check names: faithfulness, completeness, context_relevance,
# answer_correctness, answer_relevance
# ============================================================================

import argparse

REPORTS_DIR = Path("reports/generator_model_comparison")
MODEL_DIRS = [
    "SeaLLMs-v3-7B-Chat-Q4_K_M",
    "Qwen3.5-9B-Q4_K_M",
    "gemma-4-E4B-it-Q4_K_M",
    "Llama-3.1-8B-Instruct-Q4_K_M",
]

SEPARATOR_WIDTH = 100
SEPARATOR = "=" * SEPARATOR_WIDTH

def load_entries(model_dir):
    """Loads all non-error entries for one model's JSONL file. Returns
    an empty list with a warning if the file doesn't exist, rather than
    crashing -- useful if this is run before all 4 models have finished,
    or with a different model lineup than the original 4."""
    path = (REPORTS_DIR / model_dir / f"generated_answer_results.jsonl")
    if not path.exists():
        print(f"  [skip] {path} not found")
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "error" not in entry:
                entries.append(entry)
    return entries


def percentiles(values, ps=(10, 25, 50, 75, 90, 95)):
    """Returns a dict of {p: value} for a list of floats. Uses simple
    nearest-rank indexing (consistent with every percentile check run
    during the original calibration pass) -- not interpolated, since the
    original analysis that derived the live thresholds used this same
    method and re-deriving with a different interpolation scheme would
    make future comparisons inconsistent with the documented evidence."""
    if not values:
        return {p: None for p in ps}
    s = sorted(values)
    n = len(s)
    return {p: s[int(p / 100 * (n - 1))] for p in ps}


def check_context_relevance():
    """Context Relevance is deterministic (exact chunk-ID overlap), so
    this check just confirms the score distribution still looks sane --
    i.e. nearly identical across all models (since retrieval is
    generator-independent) and not surprisingly skewed by something like
    a vector_db/ rebuild or a change to CONTEXT_K."""
    print(f"\n{SEPARATOR}")
    print("CONTEXT RELEVANCE -- score distribution per model")
    print("(expected: nearly identical across all models -- retrieval is")
    print(" generator-independent. A model that looks different here")
    print(" suggests something broke, not a generator quality difference.)")
    print(SEPARATOR)
    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        counts = Counter(e.get("scores", {}).get("context_relevance") for e in entries)
        print(f"  {model_dir}: {dict(sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0])))}")


def check_faithfulness():
    """Aggregates faithfulness_grounded / faithfulness_total_numbers
    across every answer per model. A model writing far more numbers
    than the others AND grounding a smaller fraction of them is a real
    hallucination signal, not a length artifact -- the score is already
    fraction-based per answer. See docs/SCORING_CALIBRATION.md for what
    this check found on the original 400-answer run (Qwen: 418 numbers
    total, 72.0% grounded, vs 94-100% for the other three)."""
    print(f"\n{SEPARATOR}")
    print("FAITHFULNESS -- number-grounding breakdown per model")
    print(SEPARATOR)
    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        total_numbers = sum(e.get("detail", {}).get("faithfulness_total_numbers", 0) for e in entries)
        total_grounded = sum(e.get("detail", {}).get("faithfulness_grounded", 0) for e in entries)
        zero_number_count = sum(
            1 for e in entries if e.get("detail", {}).get("faithfulness_total_numbers", 0) == 0
        )
        frac = total_grounded / total_numbers if total_numbers else None
        print(
            f"  {model_dir}: n={len(entries)}, total_numbers={total_numbers}, "
            f"grounded={total_grounded}, "
            f"overall_fraction={f'{frac:.4f}' if frac is not None else 'N/A'}, "
            f"answers_with_zero_numbers={zero_number_count}"
        )


def check_completeness():
    """Reports ends_cleanly rate and the full length_ratio percentile
    distribution per model. The >2.0x / >3.0x counts are what originally
    revealed the missing-ceiling bug in score_completeness() -- if a
    model's length_ratio distribution shows its TYPICAL answer (p50) is
    already past 2x the expected answer's length, that model is
    routinely over-long, not just occasionally verbose, and the
    ceiling bands in scripts/10's score_completeness() should be
    re-checked against this distribution."""
    print(f"\n{SEPARATOR}")
    print("COMPLETENESS -- ends_cleanly rate + length_ratio distribution per model")
    print(SEPARATOR)
    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        ends_cleanly_count = sum(
            1 for e in entries if e.get("detail", {}).get("completeness_ends_cleanly")
        )
        ratios = [
            e["detail"]["completeness_length_ratio"]
            for e in entries
            if e.get("detail", {}).get("completeness_length_ratio") is not None
        ]
        n = len(entries)
        if ratios:
            p = percentiles(ratios)
            over_2x = sum(1 for r in ratios if r > 2.0)
            over_3x = sum(1 for r in ratios if r > 3.0)
            print(
                f"  {model_dir}: n={n}, ends_cleanly={ends_cleanly_count}/{n} "
                f"({ends_cleanly_count/n:.1%}), "
                f"length_ratio min={min(ratios):.3f} p50={p[50]:.3f} p75={p[75]:.3f} "
                f"p90={p[90]:.3f} p95={p[95]:.3f} max={max(ratios):.3f} "
                f"| count>2.0x={over_2x} count>3.0x={over_3x}"
            )
        else:
            print(f"  {model_dir}: n={n}, no length_ratio data found")


def check_similarity_metric(metric_key, detail_field, label_text):
    """Shared logic for both Answer Correctness and Answer Relevance --
    reports the percentile distribution of the underlying raw cosine
    similarity per model. This is the check that originally revealed
    Answer Relevance needed its OWN threshold set (max observed across
    all 400 real answers was 0.9037, well below where Answer
    Correctness's bands expect a 5) -- re-run this any time the corpus
    or embedding model changes, since these distributions are specific
    to BGE-M3 + this corpus's actual text, not a universal constant."""
    print(f"\n{SEPARATOR}")
    print(f"{label_text} -- raw similarity distribution per model")
    print(SEPARATOR)
    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        sims = [
            e["detail"][detail_field]
            for e in entries
            if e.get("detail", {}).get(detail_field) is not None
        ]
        if not sims:
            print(f"  {model_dir}: no {detail_field} values found")
            continue
        p = percentiles(sims)
        print(
            f"  {model_dir}: n={len(sims)} min={min(sims):.4f} p10={p[10]:.4f} "
            f"p25={p[25]:.4f} p50={p[50]:.4f} p75={p[75]:.4f} p90={p[90]:.4f} "
            f"max={max(sims):.4f}"
        )


def check_answer_correctness():
    check_similarity_metric(
        "answer_correctness", "answer_correctness_similarity", "ANSWER CORRECTNESS"
    )


def check_answer_relevance():
    check_similarity_metric(
        "answer_relevance", "answer_relevance_similarity", "ANSWER RELEVANCE"
    )


CHECKS = {
    "context_relevance": check_context_relevance,
    "faithfulness": check_faithfulness,
    "completeness": check_completeness,
    "answer_correctness": check_answer_correctness,
    "answer_relevance": check_answer_relevance,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=list(CHECKS.keys()),
        help="Run only one specific check (default: run all checks).",
    )
    args = parser.parse_args()

    if not REPORTS_DIR.exists():
        raise SystemExit(
            f"{REPORTS_DIR}/ not found -- run scripts/10_compare_generator_models.py first."
        )

    if args.check:
        CHECKS[args.check]()
    else:
        for check_fn in CHECKS.values():
            check_fn()

    print(f"\n{SEPARATOR}")
    print("Done. See docs/SCORING_CALIBRATION.md for how these numbers map")
    print("to the actual scoring thresholds in scripts/10_compare_generator_models.py")
    print("and scripts/rescore_generator_eval.py.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()