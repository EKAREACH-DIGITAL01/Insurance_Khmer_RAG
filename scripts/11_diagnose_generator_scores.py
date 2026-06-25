import json
import argparse
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
# Reads ONLY from:
#   reports/generator_model_comparison/<model>/generated_answer_results.jsonl
# Never touches llama-server, never re-generates anything.
#
# OUTPUT:
#   reports/generator_model_comparison/generator_score_diagnostics.json
#   reports/generator_model_comparison/generator_score_diagnostics.md
#
# USAGE:
#   python scripts/11_diagnose_generator_scores.py              # all checks
#   python scripts/11_diagnose_generator_scores.py --check NAME # one check
#
# Available check names: faithfulness, completeness, context_relevance,
#                        answer_correctness, answer_relevance, blank_answers,
#                        score_distributions, per_category
# ============================================================================

REPORTS_DIR = Path("reports/generator_model_comparison")
OUTPUT_JSON = REPORTS_DIR / "generator_score_diagnostics.json"
OUTPUT_MD   = REPORTS_DIR / "generator_score_diagnostics.md"

MODEL_DIRS = [
    "SeaLLMs-v3-7B-Chat-Q4_K_M",
    "Qwen3.5-9B-Q4_K_M",
    "gemma-4-E4B-it-Q4_K_M",
    "Llama-3.1-8B-Instruct-Q4_K_M",
]

# Display labels matching the model dir names
MODEL_LABELS = {
    "SeaLLMs-v3-7B-Chat-Q4_K_M":      "SeaLLM v3 7B",
    "Qwen3.5-9B-Q4_K_M":               "Qwen 3.5 9B",
    "gemma-4-E4B-it-Q4_K_M":           "Gemma 4 E4B",
    "Llama-3.1-8B-Instruct-Q4_K_M":    "Llama 3.1 8B",
}

SEPARATOR_WIDTH = 100
SEPARATOR = "=" * SEPARATOR_WIDTH

WEIGHTS = {
    "context_relevance":  0.20,
    "faithfulness":       0.25,
    "answer_correctness": 0.25,
    "answer_relevance":   0.15,
    "completeness":       0.15,
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_entries(model_dir):
    """Loads all non-error entries for one model's JSONL file. Returns
    an empty list with a warning if the file doesn't exist."""
    path = REPORTS_DIR / model_dir / "generated_answer_results.jsonl"
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


def is_blank(entry):
    detail = entry.get("detail", {})
    if "blank_answer" in detail:
        return bool(detail["blank_answer"])
    return not entry.get("generated_answer", "").strip()


# ----------------------------------------------------------------------------
# Percentile helper
# ----------------------------------------------------------------------------

def percentiles(values, ps=(10, 25, 50, 75, 90, 95)):
    """Nearest-rank percentiles. Consistent with the calibration pass that
    derived the live thresholds -- using the same method here ensures future
    comparisons against the documented evidence stay apples-to-apples."""
    if not values:
        return {p: None for p in ps}
    s = sorted(values)
    n = len(s)
    return {p: round(s[int(p / 100 * (n - 1))], 4) for p in ps}


# ----------------------------------------------------------------------------
# Individual checks -- each returns a structured dict for JSON output
# and a list of terminal-print lines
# ----------------------------------------------------------------------------

def check_blank_answers():
    """How many answers were blank per model, and per category."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("BLANK ANSWERS -- count per model and per category")
    lines.append("(A blank answer scores 1 on Answer Correctness and Answer Relevance,")
    lines.append(" NOT None -- it is a computable outcome, not a missing metric.)")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label     = MODEL_LABELS[model_dir]
        blank     = [e for e in entries if is_blank(e)]
        by_cat    = Counter(e.get("category", "unknown") for e in blank)
        result[model_dir] = {
            "label":       label,
            "total":       len(entries),
            "blank_count": len(blank),
            "blank_pct":   round(len(blank) / len(entries) * 100, 1) if entries else 0,
            "by_category": dict(sorted(by_cat.items())),
        }
        lines.append(
            f"  {label}: {len(blank)}/{len(entries)} blank "
            f"({result[model_dir]['blank_pct']}%) | by category: {dict(sorted(by_cat.items()))}"
        )

    return result, lines


def check_context_relevance():
    """Score distribution (should be nearly identical across all models
    since retrieval is generator-independent)."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("CONTEXT RELEVANCE -- score distribution per model")
    lines.append("(expected: nearly identical across all models -- retrieval is")
    lines.append(" generator-independent. A model that looks different here")
    lines.append(" suggests something broke, not a generator quality difference.)")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label  = MODEL_LABELS[model_dir]
        counts = Counter(
            e.get("scores", {}).get("context_relevance") for e in entries
        )
        dist   = dict(sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0])))
        result[model_dir] = {"label": label, "score_distribution": dist}
        lines.append(f"  {label}: {dist}")

    return result, lines


def check_faithfulness():
    """Number-grounding breakdown -- the key signal for hallucination risk."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("FAITHFULNESS -- number-grounding breakdown per model")
    lines.append("(Qwen expected to write more numbers with lower grounding fraction)")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label         = MODEL_LABELS[model_dir]
        total_numbers = sum(e.get("detail", {}).get("faithfulness_total_numbers", 0) for e in entries)
        total_grounded = sum(e.get("detail", {}).get("faithfulness_grounded", 0) for e in entries)
        zero_number_count = sum(
            1 for e in entries if e.get("detail", {}).get("faithfulness_total_numbers", 0) == 0
        )
        frac = round(total_grounded / total_numbers, 4) if total_numbers else None
        result[model_dir] = {
            "label":                   label,
            "n":                       len(entries),
            "total_numbers":           total_numbers,
            "total_grounded":          total_grounded,
            "overall_fraction":        frac,
            "answers_with_zero_numbers": zero_number_count,
        }
        lines.append(
            f"  {label}: n={len(entries)}, total_numbers={total_numbers}, "
            f"grounded={total_grounded}, "
            f"overall_fraction={frac if frac is not None else 'N/A'}, "
            f"answers_with_zero_numbers={zero_number_count}"
        )

    return result, lines


def check_completeness():
    """ends_cleanly rate + length_ratio percentile distribution.
    The >2.0x / >3.0x counts were the key evidence for the ceiling fix."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("COMPLETENESS -- ends_cleanly rate + length_ratio distribution per model")
    lines.append("(>2.0x and >3.0x counts were the evidence for the ceiling in score_completeness)")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label = MODEL_LABELS[model_dir]
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
            over_2x  = sum(1 for r in ratios if r > 2.0)
            over_3x  = sum(1 for r in ratios if r > 3.0)
            result[model_dir] = {
                "label":              label,
                "n":                  n,
                "ends_cleanly":       ends_cleanly_count,
                "ends_cleanly_pct":   round(ends_cleanly_count / n * 100, 1),
                "length_ratio_min":   round(min(ratios), 4),
                "length_ratio_max":   round(max(ratios), 4),
                "length_ratio_p10":   p[10],
                "length_ratio_p25":   p[25],
                "length_ratio_p50":   p[50],
                "length_ratio_p75":   p[75],
                "length_ratio_p90":   p[90],
                "length_ratio_p95":   p[95],
                "count_over_2x":      over_2x,
                "count_over_3x":      over_3x,
            }
            lines.append(
                f"  {label}: n={n}, ends_cleanly={ends_cleanly_count}/{n} "
                f"({result[model_dir]['ends_cleanly_pct']}%), "
                f"length_ratio min={min(ratios):.3f} p50={p[50]:.3f} p75={p[75]:.3f} "
                f"p90={p[90]:.3f} p95={p[95]:.3f} max={max(ratios):.3f} "
                f"| count>2.0x={over_2x} count>3.0x={over_3x}"
            )
        else:
            result[model_dir] = {"label": label, "n": n, "no_length_ratio_data": True}
            lines.append(f"  {label}: n={n}, no length_ratio data found")

    return result, lines


def check_similarity_metric(detail_field, label_text, note):
    """Shared logic for Answer Correctness and Answer Relevance -- reports
    the percentile distribution of the underlying raw cosine similarity.
    Blank answers (similarity=0.0) are INCLUDED in the distribution, since
    they represent a real outcome, not missing data."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append(f"{label_text} -- raw similarity distribution per model")
    lines.append(f"({note})")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label = MODEL_LABELS[model_dir]
        # Include blank-answer entries: their similarity was set to 0.0
        # by scripts/10 (or scripts/12 rescore). Filter only genuinely
        # missing values (None = no expected_answer in gold set).
        sims = [
            e["detail"][detail_field]
            for e in entries
            if e.get("detail", {}).get(detail_field) is not None
        ]
        if not sims:
            result[model_dir] = {"label": label, "no_data": True}
            lines.append(f"  {label}: no {detail_field} values found")
            continue
        p = percentiles(sims)
        blank_at_zero = sum(1 for s in sims if s == 0.0)
        result[model_dir] = {
            "label":          label,
            "n":              len(sims),
            "blank_at_zero":  blank_at_zero,
            "min":            round(min(sims), 4),
            "max":            round(max(sims), 4),
            "p10":            p[10],
            "p25":            p[25],
            "p50":            p[50],
            "p75":            p[75],
            "p90":            p[90],
            "p95":            p[95],
        }
        lines.append(
            f"  {label}: n={len(sims)} (incl. {blank_at_zero} blank@0.0) "
            f"min={min(sims):.4f} p10={p[10]:.4f} "
            f"p25={p[25]:.4f} p50={p[50]:.4f} p75={p[75]:.4f} p90={p[90]:.4f} "
            f"p95={p[95]:.4f} max={max(sims):.4f}"
        )

    return result, lines


def check_answer_correctness():
    return check_similarity_metric(
        "answer_correctness_similarity",
        "ANSWER CORRECTNESS",
        "answer vs. expected_answer | blank answers included at similarity=0.0"
    )


def check_answer_relevance():
    return check_similarity_metric(
        "answer_relevance_similarity",
        "ANSWER RELEVANCE",
        "answer vs. question | structurally lower ceiling than Answer Correctness | "
        "blank answers included at similarity=0.0"
    )


def check_score_distributions():
    """Final 1-5 score distributions for every metric across all models.
    Shows how the bucketing thresholds spread the data."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("SCORE DISTRIBUTIONS -- final 1-5 scores per metric per model")
    lines.append("(Shows how thresholds spread actual data across the 1-5 scale)")
    lines.append(SEPARATOR)

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label = MODEL_LABELS[model_dir]
        model_result = {"label": label}
        lines.append(f"\n  {label}:")
        for metric in WEIGHTS:
            dist = Counter(
                e.get("scores", {}).get(metric)
                for e in entries
                if e.get("scores", {}).get(metric) is not None
            )
            d = dict(sorted(dist.items()))
            model_result[metric] = d
            avg_vals = [
                e["scores"][metric]
                for e in entries
                if e.get("scores", {}).get(metric) is not None
            ]
            avg = round(sum(avg_vals) / len(avg_vals), 4) if avg_vals else None
            lines.append(f"    {metric}: {d}  avg={avg}")
        # combined
        combined_vals = [r["combined"] for r in entries if r.get("combined") is not None]
        combined_avg  = round(sum(combined_vals) / len(combined_vals), 4) if combined_vals else None
        model_result["combined_avg"] = combined_avg
        lines.append(f"    combined (weighted avg): {combined_avg}")
        result[model_dir] = model_result

    return result, lines


def check_per_category():
    """Per-category metric averages for each model. Useful for identifying
    whether retrieval or generation weaknesses are category-specific."""
    result = {}
    lines  = []
    lines.append(f"\n{SEPARATOR}")
    lines.append("PER-CATEGORY BREAKDOWN -- metric averages per model per category")
    lines.append(SEPARATOR)

    categories = ["guidance", "laws", "prakas", "sub_decrees"]

    for model_dir in MODEL_DIRS:
        entries = load_entries(model_dir)
        if not entries:
            continue
        label = MODEL_LABELS[model_dir]
        model_result = {"label": label, "categories": {}}
        lines.append(f"\n  {label}:")

        for cat in categories:
            cat_entries = [e for e in entries if e.get("category") == cat]
            if not cat_entries:
                continue
            cat_result = {"n": len(cat_entries)}
            for metric in WEIGHTS:
                vals = [
                    e["scores"][metric]
                    for e in cat_entries
                    if e.get("scores", {}).get(metric) is not None
                ]
                cat_result[metric] = round(sum(vals) / len(vals), 4) if vals else None
            blank_in_cat = sum(1 for e in cat_entries if is_blank(e))
            cat_result["blank_count"] = blank_in_cat
            model_result["categories"][cat] = cat_result

            metric_str = "  ".join(
                f"{m[:6]}={cat_result.get(m, 'N/A'):.4f}"
                if isinstance(cat_result.get(m), float) else f"{m[:6]}=N/A"
                for m in WEIGHTS
            )
            lines.append(f"    [{cat}] n={len(cat_entries)} blank={blank_in_cat} | {metric_str}")

        result[model_dir] = model_result

    return result, lines


# ----------------------------------------------------------------------------
# Write JSON + MD reports
# ----------------------------------------------------------------------------

def write_reports(all_results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved diagnostics JSON : {OUTPUT_JSON}")

    # --- Markdown ---
    lines = []
    lines.append("# Generator Score Diagnostics\n\n")
    lines.append(
        "Evidence file for the calibration of `scripts/10_compare_generator_models.py` "
        "scoring thresholds. See `docs/SCORING_CALIBRATION.md` for the full interpretation.\n\n"
    )
    lines.append(
        "> **Blank answer policy**: blank answers score **1** on Answer Correctness "
        "and Answer Relevance (not excluded). `similarity = 0.0` is stored in "
        "`detail.answer_correctness_similarity` and `detail.answer_relevance_similarity` "
        "for traceability.\n\n"
    )

    # Blank answers
    lines.append("## Blank Answers\n\n")
    blank_data = all_results.get("blank_answers", {})
    rows = []
    for md, d in blank_data.items():
        rows.append(
            f"| {d['label']} | {d['blank_count']}/{d['total']} ({d['blank_pct']}%) "
            f"| {d.get('by_category', {})} |"
        )
    lines.append("| Model | Blank / Total | By Category |\n")
    lines.append("|-------|--------------|-------------|\n")
    lines.extend(r + "\n" for r in rows)
    lines.append("\n")

    # Context Relevance
    lines.append("## Context Relevance Score Distributions\n\n")
    lines.append("Expected to be nearly identical across all models (retrieval is generator-independent).\n\n")
    cr_data = all_results.get("context_relevance", {})
    for md, d in cr_data.items():
        lines.append(f"- **{d['label']}**: `{d.get('score_distribution', {})}`\n")
    lines.append("\n")

    # Faithfulness
    lines.append("## Faithfulness — Number Grounding\n\n")
    lines.append(
        "Fraction of Arabic/Khmer numerals in the generated answer that "
        "appear in the retrieved context. Lower fraction = higher hallucination risk.\n\n"
    )
    faith_data = all_results.get("faithfulness", {})
    lines.append("| Model | n | Total Numbers | Grounded | Fraction | Zero-Number Answers |\n")
    lines.append("|-------|---|--------------|---------|---------|--------------------|\n")
    for md, d in faith_data.items():
        frac = f"{d['overall_fraction']:.4f}" if d.get("overall_fraction") is not None else "N/A"
        lines.append(
            f"| {d['label']} | {d['n']} | {d['total_numbers']} | "
            f"{d['total_grounded']} | {frac} | {d['answers_with_zero_numbers']} |\n"
        )
    lines.append("\n")

    # Completeness
    lines.append("## Completeness — Length Ratio Distribution\n\n")
    lines.append(
        "The `>2.0x` and `>3.0x` counts were the primary evidence for adding "
        "the ceiling to `score_completeness()`. Qwen's typical answer (p50=1.98x) "
        "was getting a free 5 under the old floor-only logic.\n\n"
    )
    comp_data = all_results.get("completeness", {})
    lines.append("| Model | n | Ends Cleanly | p50 | p90 | max | >2.0x | >3.0x |\n")
    lines.append("|-------|---|-------------|-----|-----|-----|-------|-------|\n")
    for md, d in comp_data.items():
        if d.get("no_length_ratio_data"):
            lines.append(f"| {d['label']} | {d['n']} | N/A | N/A | N/A | N/A | N/A | N/A |\n")
        else:
            lines.append(
                f"| {d['label']} | {d['n']} | "
                f"{d['ends_cleanly']}/{d['n']} ({d['ends_cleanly_pct']}%) | "
                f"{d.get('length_ratio_p50', 'N/A')} | "
                f"{d.get('length_ratio_p90', 'N/A')} | "
                f"{d.get('length_ratio_max', 'N/A')} | "
                f"{d.get('count_over_2x', 'N/A')} | "
                f"{d.get('count_over_3x', 'N/A')} |\n"
            )
    lines.append("\n")

    # Answer Correctness
    lines.append("## Answer Correctness — Similarity Distribution\n\n")
    lines.append(
        "BGE-M3 cosine similarity between generated answer and `expected_answer`. "
        "Blank answers are included at similarity=0.0 (not excluded).\n\n"
    )
    ac_data = all_results.get("answer_correctness", {})
    lines.append("| Model | n | Blank@0.0 | min | p50 | p90 | max |\n")
    lines.append("|-------|---|-----------|-----|-----|-----|-----|\n")
    for md, d in ac_data.items():
        if d.get("no_data"):
            lines.append(f"| {d['label']} | N/A | N/A | N/A | N/A | N/A | N/A |\n")
        else:
            lines.append(
                f"| {d['label']} | {d['n']} | {d['blank_at_zero']} | "
                f"{d['min']} | {d['p50']} | {d['p90']} | {d['max']} |\n"
            )
    lines.append("\n")

    # Answer Relevance
    lines.append("## Answer Relevance — Similarity Distribution\n\n")
    lines.append(
        "BGE-M3 cosine similarity between generated answer and the original question. "
        "Structurally lower ceiling than Answer Correctness — these use separate threshold sets. "
        "Blank answers included at similarity=0.0.\n\n"
    )
    ar_data = all_results.get("answer_relevance", {})
    lines.append("| Model | n | Blank@0.0 | min | p50 | p90 | max |\n")
    lines.append("|-------|---|-----------|-----|-----|-----|-----|\n")
    for md, d in ar_data.items():
        if d.get("no_data"):
            lines.append(f"| {d['label']} | N/A | N/A | N/A | N/A | N/A | N/A |\n")
        else:
            lines.append(
                f"| {d['label']} | {d['n']} | {d['blank_at_zero']} | "
                f"{d['min']} | {d['p50']} | {d['p90']} | {d['max']} |\n"
            )
    lines.append("\n")

    # Score distributions
    lines.append("## Final Score Distributions (1–5) per Metric\n\n")
    sd_data = all_results.get("score_distributions", {})
    for md, d in sd_data.items():
        lines.append(f"### {d['label']}\n\n")
        for metric in WEIGHTS:
            dist = d.get(metric, {})
            lines.append(f"- **{metric.replace('_', ' ').title()}**: `{dist}`\n")
        lines.append(f"- **Combined (weighted avg)**: `{d.get('combined_avg')}`\n\n")

    # Per-category
    lines.append("## Per-Category Breakdown\n\n")
    pc_data = all_results.get("per_category", {})
    metrics_short = list(WEIGHTS.keys())
    header = "| Model | Category | n | Blank | " + " | ".join(
        m.replace("_", " ").title() for m in metrics_short
    ) + " |\n"
    sep    = "|-------|----------|---|-------|" + "-------|" * len(metrics_short) + "\n"
    lines.append(header)
    lines.append(sep)
    for md, d in pc_data.items():
        for cat, cat_d in d.get("categories", {}).items():
            row_vals = []
            for m in metrics_short:
                v = cat_d.get(m)
                row_vals.append(f"{v:.4f}" if isinstance(v, float) else "N/A")
            lines.append(
                f"| {d['label']} | {cat} | {cat_d['n']} | {cat_d['blank_count']} | "
                + " | ".join(row_vals) + " |\n"
            )
    lines.append("\n")

    # Threshold reference
    lines.append("## Threshold Reference\n\n")
    lines.append("### Answer Correctness\n\n")
    lines.append("| Min Similarity | Score |\n|---------------|-------|\n")
    thresholds_ac = [(0.90,5),(0.80,4),(0.70,3),(0.60,2),(0.50,1),(0.00,1)]
    for t, s in thresholds_ac:
        lines.append(f"| ≥ {t:.2f} | {s} |\n")
    lines.append("\n### Answer Relevance\n\n")
    lines.append("| Min Similarity | Score |\n|---------------|-------|\n")
    thresholds_ar = [(0.82,5),(0.75,4),(0.65,3),(0.55,2),(0.00,1)]
    for t, s in thresholds_ar:
        lines.append(f"| ≥ {t:.2f} | {s} |\n")
    lines.append("\n### Completeness (length_ratio = len(generated) / len(expected))\n\n")
    lines.append(
        "| Condition | Score |\n|-----------|-------|\n"
        "| Ends cleanly AND 0.80 ≤ ratio ≤ 2.00 | 5 |\n"
        "| Ends cleanly AND (0.60 ≤ ratio < 0.80 OR 2.00 < ratio ≤ 3.00) | 4 |\n"
        "| Ends cleanly AND (ratio < 0.60 OR ratio > 3.00) | 3 |\n"
        "| Truncated AND ratio ≥ 0.80 | 3 |\n"
        "| Truncated AND ratio ≥ 0.40 | 2 |\n"
        "| Truncated AND ratio < 0.40 (incl. blank) | 1 |\n\n"
    )

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved diagnostics MD   : {OUTPUT_MD}")


# ----------------------------------------------------------------------------
# Check registry
# ----------------------------------------------------------------------------

CHECKS = {
    "blank_answers":       check_blank_answers,
    "context_relevance":   check_context_relevance,
    "faithfulness":        check_faithfulness,
    "completeness":        check_completeness,
    "answer_correctness":  check_answer_correctness,
    "answer_relevance":    check_answer_relevance,
    "score_distributions": check_score_distributions,
    "per_category":        check_per_category,
}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        choices=list(CHECKS.keys()),
        help="Run only one specific check (default: run all and save reports).",
    )
    args = parser.parse_args()

    if not REPORTS_DIR.exists():
        raise SystemExit(
            f"{REPORTS_DIR}/ not found -- run scripts/10_compare_generator_models.py first."
        )

    if args.check:
        # Single check: print only, no file output
        fn = CHECKS[args.check]
        _, lines = fn()
        for line in lines:
            print(line)
        print(f"\n{SEPARATOR}")
        print("Single-check mode: no files written. Run without --check to save reports.")
        print(SEPARATOR)
        return

    # Full run: collect all results, print, write files
    all_results = {}
    all_lines   = []

    for check_name, fn in CHECKS.items():
        result, lines = fn()
        all_results[check_name] = result
        all_lines.extend(lines)

    # Print all terminal output
    for line in all_lines:
        print(line)

    print(f"\n{SEPARATOR}")
    print("Done. Writing diagnostic reports...")
    write_reports(all_results)
    print(f"\nSee docs/SCORING_CALIBRATION.md for threshold interpretation.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()