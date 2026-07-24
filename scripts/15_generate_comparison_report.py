#!/usr/bin/env python3
# ============================================================================
# 15_generate_comparison_report.py
#
# PURPOSE: turn the LLM judge's scores into the generator comparison reports.
#
# By the time this runs, each model's generated_answer_results.jsonl has been
# scored in place by the LLM judge (see docs/llm_judge_prompt.md). This script
# only aggregates -- it never scores anything itself.
#
# INPUT
#   reports/generator_model_comparison/<model>/generated_answer_results.jsonl
#
# OUTPUT
#   <model>/generated_answer_results_summary.json   per-model, machine-readable
#   <model>/generated_answer_results_summary.md     per-model, human-readable
#   generator_model_comparison.json                 cross-model + winner
#   generator_model_comparison.md                   cross-model + winner
#
# The winner is the model with the highest weighted combined score. That model
# is the one shipped in app.py and replayed by scripts/16.
#
# USAGE:
#   python scripts/15_generate_comparison_report.py
# ============================================================================

import json
import re
from pathlib import Path
from statistics import mean

from tabulate import tabulate

REPORTS_DIR = Path("reports/generator_model_comparison")

# context_relevance is measured deterministically by scripts/14 from the chunk
# ids; the other four are assigned by the LLM judge.
JUDGED_METRICS = ["faithfulness", "answer_relevance",
                  "answer_correctness", "completeness"]

METRICS = [
    ("context_relevance",  "Context Relevance"),
    ("faithfulness",       "Faithfulness"),
    ("answer_relevance",   "Answer Relevance"),
    ("answer_correctness", "Answer Correctness"),
    ("completeness",       "Completeness"),
]

# Must match the weights the judge is told to use in docs/llm_judge_prompt.md.
WEIGHTS = {
    "context_relevance":  0.20,
    "faithfulness":       0.25,
    "answer_correctness": 0.25,
    "answer_relevance":   0.15,
    "completeness":       0.15,
}

# Khmer sentence-final punctuation, plus the trailing decoration that may
# legitimately follow it. Used ONLY for the descriptive "truncated answers"
# count -- it is not a score and feeds nothing the judge decides.
KHMER_END_PUNCT   = {"។", "៕"}
_TRAILING_DECOR   = re.compile(r"[\s*_`#>~]+$")
_TRAILING_BRACKET = re.compile(
    r"[(\[{（【「『][^()\[\]{}（）【】「」『』]*[)\]\}）】」』]\s*$"
)

BOLD  = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105


def ends_with_sentence_final(text):
    """True if the answer ends on ។ or ៕ once trailing markdown / bracketed
    closing tags are stripped. A genuinely truncated answer has no such mark."""
    s = (text or "").strip()
    prev = None
    while s and s != prev:
        prev = s
        s = _TRAILING_BRACKET.sub("", s).rstrip()
        s = _TRAILING_DECOR.sub("", s)
    return bool(s) and s[-1] in KHMER_END_PUNCT


def _truncate(text, n=60):
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _avg(values, n=4):
    vals = [v for v in values if v is not None]
    return round(mean(vals), n) if vals else None


def load_rows(model_dir):
    path = model_dir / "generated_answer_results.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def weighted_combined(scores):
    """Recomputed here rather than trusting the judge's own `combined`, so the
    ranking cannot be skewed by an arithmetic slip in a single judged record."""
    total = weight = 0.0
    for key, w in WEIGHTS.items():
        v = scores.get(key)
        if v:  # 0 == not judged, and is excluded rather than counted as zero
            total  += v * w
            weight += w
    return round(total / weight, 4) if weight else None


def summarise(rows, label):
    # A record counts as judged only when the LLM has filled in all four of
    # ITS metrics. context_relevance is pre-filled by scripts/14, so testing
    # "any score is non-zero" would mark every record judged before the judge
    # ever ran.
    scored = [r for r in rows
              if all(r.get("scores", {}).get(m) for m in JUDGED_METRICS)]

    scores_avg = {
        key: _avg([r["scores"].get(key) for r in scored])
        for key, _ in METRICS
    }
    combined_vals = [weighted_combined(r.get("scores", {})) for r in scored]

    def halluc_flag(r):
        return str(r.get("hallucination", "")).strip().lower()

    flagged_rows = [r for r in scored if halluc_flag(r) == "true"]
    halluc_true  = len(flagged_rows)
    halluc_known = sum(1 for r in scored if halluc_flag(r) in ("true", "false"))

    # Every flagged answer, with the fabricated span the judge quoted. This is
    # the actionable part of the hallucination report -- a bare count tells a
    # reader nothing they can check.
    flagged_detail = [
        {
            "question_id":  r.get("question_id"),
            "category":     r.get("category"),
            "question":     r.get("question"),
            "evidence":     r.get("hallucination_evidence", ""),
            "faithfulness": r.get("scores", {}).get("faithfulness"),
            "combined":     weighted_combined(r.get("scores", {})),
        }
        for r in flagged_rows
    ]

    # A hallucination that the faithfulness score did NOT catch is the most
    # interesting case: the judge called the answer grounded (4-5) and then
    # flagged it as fabricating something. Either reading is suspect.
    contradictory = [f for f in flagged_detail
                     if f["faithfulness"] is not None and f["faithfulness"] >= 4]

    halluc_by_category = {}
    for cat in sorted({r.get("category") for r in scored if r.get("category")}):
        crows = [r for r in scored if r.get("category") == cat]
        known = sum(1 for r in crows if halluc_flag(r) in ("true", "false"))
        flagged = sum(1 for r in crows if halluc_flag(r) == "true")
        halluc_by_category[cat] = {
            "flagged": flagged,
            "judged":  known,
            "rate":    round(flagged / known, 4) if known else None,
        }

    timings = [r.get("timing") or {} for r in rows]
    def t_avg(key):
        return _avg([t.get(key) for t in timings])

    per_category = {}
    for cat in sorted({r.get("category") for r in scored if r.get("category")}):
        crows = [r for r in scored if r.get("category") == cat]
        per_category[cat] = {
            "questions": len(crows),
            "scores": {k: _avg([r["scores"].get(k) for r in crows])
                       for k, _ in METRICS},
            "combined": _avg([weighted_combined(r.get("scores", {})) for r in crows]),
        }

    return {
        "label":              label,
        "total_questions":    len(rows),
        "judged_questions":   len(scored),
        "unjudged_questions": len(rows) - len(scored),
        "blank_answers":      sum(1 for r in rows if not (r.get("generated_answer") or "").strip()),
        "truncated_answers":  sum(1 for r in rows
                                  if (r.get("generated_answer") or "").strip()
                                  and not ends_with_sentence_final(r["generated_answer"])),
        "scores_avg":         scores_avg,
        "combined_avg":       _avg(combined_vals),
        "hallucination": {
            "flagged":              halluc_true,
            "judged":               halluc_known,
            "rate":                 round(halluc_true / halluc_known, 4) if halluc_known else None,
            "unjudged_flag":        len(scored) - halluc_known,
            "by_category":          halluc_by_category,
            "flagged_questions":    flagged_detail,
            "contradicts_faithfulness": contradictory,
        },
        "timing_avg": {
            "first_token_time": t_avg("first_token_time"),
            "generation_time":  t_avg("generation_time"),
            "tokens_out":       t_avg("tokens_out"),
            "tokens_per_sec":   t_avg("tokens_per_sec"),
        },
        "per_category": per_category,
    }


def write_model_summary(summary, model_dir):
    (model_dir / "generated_answer_results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# {summary['label']} — Answer Summary\n\n",
          f"Scored by an LLM judge (see `docs/llm_judge_prompt.md`). "
          f"{summary['judged_questions']}/{summary['total_questions']} answers judged.\n\n",
          "## Quality (1–5)\n\n"]

    rows = [[name, summary["scores_avg"].get(key)] for key, name in METRICS]
    rows.append(["**Combined (weighted)**", summary["combined_avg"]])
    md.append(tabulate(rows, headers=["Metric", "Avg"], tablefmt="github") + "\n\n")

    h = summary["hallucination"]
    md.append("## Hallucination\n\n")
    md.append(f"- **Flagged:** {h['flagged']}/{h['judged']}"
              + (f" ({h['rate']:.0%})\n" if h["rate"] is not None else "\n"))
    md.append("- Judged separately; never folded into the combined score.\n")
    if h["unjudged_flag"]:
        md.append(f"- {h['unjudged_flag']} scored answer(s) have no "
                  f"hallucination verdict.\n")
    md.append("\n")

    if h["by_category"]:
        md.append("### By category\n\n")
        md.append(tabulate(
            [[cat, c["flagged"], c["judged"],
              f"{c['rate']:.0%}" if c["rate"] is not None else "—"]
             for cat, c in h["by_category"].items()],
            headers=["Category", "Flagged", "Judged", "Rate"],
            tablefmt="github") + "\n\n")

    if h["flagged_questions"]:
        md.append("### Flagged answers\n\n")
        md.append("The span quoted is what the judge said was fabricated — "
                  "check it against the retrieved context before acting on it.\n\n")
        md.append(tabulate(
            [[f["question_id"], f["category"], _truncate(f["question"], 48),
              _truncate(f["evidence"], 60) or "_(no evidence given)_",
              f["faithfulness"]]
             for f in h["flagged_questions"]],
            headers=["Question ID", "Category", "Question", "Fabricated span", "Faith."],
            tablefmt="github") + "\n\n")

    if h["contradicts_faithfulness"]:
        md.append("### Flagged despite a high faithfulness score\n\n")
        md.append(f"**{len(h['contradicts_faithfulness'])}** answer(s) were scored "
                  f"4–5 on faithfulness *and* flagged as hallucinating. One of the "
                  f"two judgements is wrong — review these by hand.\n\n")
        md.append(tabulate(
            [[f["question_id"], f["faithfulness"], _truncate(f["evidence"], 70)]
             for f in h["contradicts_faithfulness"]],
            headers=["Question ID", "Faithfulness", "Fabricated span"],
            tablefmt="github") + "\n\n")

    md.append("## Coverage & latency\n\n")
    t = summary["timing_avg"]
    md.append(tabulate([
        ["Blank answers",     summary["blank_answers"]],
        ["Truncated answers", summary["truncated_answers"]],
        ["Avg TTFT (s)",      t["first_token_time"]],
        ["Avg gen time (s)",  t["generation_time"]],
        ["Avg tokens out",    t["tokens_out"]],
        ["Avg tok/s",         t["tokens_per_sec"]],
    ], headers=["Metric", "Value"], tablefmt="github") + "\n\n")

    if summary["per_category"]:
        md.append("## Per category\n\n")
        headers = ["Category", "Qs"] + [n for _, n in METRICS] + ["Combined"]
        crows = []
        for cat, c in summary["per_category"].items():
            crows.append([cat, c["questions"]]
                         + [c["scores"].get(k) for k, _ in METRICS]
                         + [c["combined"]])
        md.append(tabulate(crows, headers=headers, tablefmt="github") + "\n")

    (model_dir / "generated_answer_results_summary.md").write_text(
        "".join(md), encoding="utf-8")


def write_comparison(summaries):
    ranked = sorted(summaries,
                    key=lambda s: (s["combined_avg"] is not None,
                                   s["combined_avg"] or 0.0),
                    reverse=True)
    # A model with no judged answers has no combined score, so it cannot win.
    # Without this guard an entirely unjudged run still names a "winner" --
    # whichever model happens to sort first -- which reads as a real result.
    scored_models = [s for s in ranked if s["combined_avg"] is not None]
    winner = scored_models[0]["label"] if scored_models else None

    report = {
        "judge": "LLM-as-judge (RAGAS + hallucination) — see docs/llm_judge_prompt.md",
        "weights": WEIGHTS,
        "ranking": ranked,
        "recommendation": winner,
        "note": (
            "Scores are assigned by an LLM judge reading each answer against its "
            "retrieved context and the gold expected_answer; they are not "
            "deterministic proxies. combined is recomputed here from the five "
            "metrics using the weights above. Hallucination is judged separately "
            "and never folded into combined. An LLM judge on Khmer legal text is "
            "not reliable by default -- validate it against a hand-scored sample "
            "before treating this ranking as measured rather than asserted."
        ),
    }
    (REPORTS_DIR / "generator_model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = [s["label"] for s in ranked]
    md = ["# Generator Model Comparison Report\n\n",
          "> **Context Relevance** is measured deterministically by `scripts/14` "
          "as recall of `expected_chunks` in `context_chunk_ids` — it scores "
          "retrieval, not the answer. The other four metrics are assigned by an "
          "**LLM judge** comparing `generated_answer` against `expected_answer`. "
          "Hallucination is judged separately and is **never** folded into the "
          "combined score. See `docs/llm_judge_prompt.md`.\n\n",
          "## Quality Scores (macro-avg, 1–5 scale)\n\n"]

    qrows = [[name] + [f"{s['scores_avg'].get(key):.4f}"
                       if s["scores_avg"].get(key) is not None else "—"
                       for s in ranked]
             for key, name in METRICS]
    qrows.append(["**Combined (weighted)**"]
                 + [f"**{s['combined_avg']:.4f}**" if s["combined_avg"] is not None
                    else "—" for s in ranked])
    md.append(tabulate(qrows, headers=["Metric"] + labels, tablefmt="github") + "\n\n")

    md.append("## Hallucination (separate from the score)\n\n")
    md.append("A hallucination is any specific fact asserted by the answer that "
              "does not appear in its retrieved context. Judged independently of "
              "the five metrics and **never** folded into the combined score — a "
              "model can rank first and still fabricate.\n\n")
    md.append(tabulate(
        [["── Flagged ──"] + [s["hallucination"]["flagged"] for s in ranked],
         ["── Judged ──"]  + [s["hallucination"]["judged"] for s in ranked],
         ["── Rate ──"]    + [f"{s['hallucination']['rate']:.0%}"
                             if s["hallucination"]["rate"] is not None else "—"
                             for s in ranked]],
        headers=["Metric"] + labels, tablefmt="github") + "\n\n")

    # rate per category, per model
    cats = sorted({c for s in ranked for c in s["hallucination"]["by_category"]})
    if cats:
        md.append("### Rate by category\n\n")
        crows = []
        for cat in cats:
            row = [cat]
            for s in ranked:
                c = s["hallucination"]["by_category"].get(cat)
                row.append(f"{c['rate']:.0%}" if c and c["rate"] is not None else "—")
            crows.append(row)
        md.append(tabulate(crows, headers=["Category"] + labels,
                           tablefmt="github") + "\n\n")

    # questions that more than one model hallucinated on -- these point at a
    # retrieval or gold-set problem rather than a single model's weakness
    seen = {}
    for s in ranked:
        for f in s["hallucination"]["flagged_questions"]:
            seen.setdefault(f["question_id"], []).append(s["label"])
    shared = {q: ms for q, ms in seen.items() if len(ms) > 1}
    if shared:
        md.append("### Questions multiple models hallucinated on\n\n")
        md.append("More than one model fabricating on the same question usually "
                  "means the retrieved context was inadequate, not that every "
                  "model failed independently. Check retrieval for these first.\n\n")
        md.append(tabulate(
            [[q, len(ms), ", ".join(ms)] for q, ms in
             sorted(shared.items(), key=lambda kv: -len(kv[1]))],
            headers=["Question ID", "Models", "Which"], tablefmt="github") + "\n\n")

    contradictions = sum(len(s["hallucination"]["contradicts_faithfulness"])
                         for s in ranked)
    if contradictions:
        md.append(f"### Judge self-contradictions\n\n")
        md.append(f"**{contradictions}** answer(s) across all models scored 4–5 on "
                  f"faithfulness yet were flagged as hallucinating. Faithfulness and "
                  f"the hallucination flag measure the same thing, so these are "
                  f"judge errors somewhere — a useful sanity check on the judge "
                  f"itself. Per-model detail is in each "
                  f"`generated_answer_results_summary.md`.\n\n")

    md.append("## Latency & Coverage\n\n")
    md.append(tabulate([
        ["── Avg TTFT (s) ──"]      + [s["timing_avg"]["first_token_time"] for s in ranked],
        ["── Avg gen time (s) ──"]  + [s["timing_avg"]["generation_time"] for s in ranked],
        ["── Avg tok/s ──"]         + [s["timing_avg"]["tokens_per_sec"] for s in ranked],
        ["── Blank answers ──"]     + [s["blank_answers"] for s in ranked],
        ["── Truncated answers ──"] + [s["truncated_answers"] for s in ranked],
        ["── Unjudged ──"]          + [s["unjudged_questions"] for s in ranked],
    ], headers=["Metric"] + labels, tablefmt="github") + "\n\n")

    md.append("## Ranking\n\n")
    for i, s in enumerate(ranked, 1):
        combined = f"{s['combined_avg']:.4f}" if s["combined_avg"] is not None else "—"
        md.append(f"{i}. **{s['label']}** — combined={combined} | "
                  f"hallucinated={s['hallucination']['flagged']} | "
                  f"blank={s['blank_answers']}\n")

    if winner:
        md.append(f"\n**Recommendation: {winner}**\n\n")
        md.append("Use this model for the live chatbot and "
                  "`scripts/16_evaluate_safety_gates.py`.\n")
    else:
        md.append("\n**No recommendation — no answer has been judged yet.**\n\n")
        md.append("Every metric is still 0. Run the LLM judge "
                  "(`docs/llm_judge_prompt.md`) and re-run this script.\n")

    (REPORTS_DIR / "generator_model_comparison.md").write_text(
        "".join(md), encoding="utf-8")

    return ranked, winner


def main():
    if not REPORTS_DIR.exists():
        raise SystemExit(f"No {REPORTS_DIR}/ -- run scripts/13 first.")

    summaries = []
    for model_dir in sorted(p for p in REPORTS_DIR.iterdir() if p.is_dir()):
        rows = load_rows(model_dir)
        if rows is None:
            continue
        summary = summarise(rows, model_dir.name)
        write_model_summary(summary, model_dir)
        summaries.append(summary)
        print(f"  {model_dir.name:38s} "
              f"{summary['judged_questions']}/{summary['total_questions']} judged")

    if not summaries:
        raise SystemExit("No generated_answer_results.jsonl found.")

    unjudged = sum(s["unjudged_questions"] for s in summaries)
    if unjudged:
        print(f"\n  WARNING: {unjudged} answer(s) still unscored (all metrics 0). "
              f"They are EXCLUDED from every average below.\n"
              f"  Finish judging before citing this report.")

    ranked, winner = write_comparison(summaries)

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"{BOLD}RANKING{RESET}")
    for i, s in enumerate(ranked, 1):
        combined = f"{s['combined_avg']:.4f}" if s["combined_avg"] is not None else "—"
        print(f"  {i}. {s['label']:38s} combined={combined} "
              f"hallucinated={s['hallucination']['flagged']}")
    print(f"{'=' * SEPARATOR_WIDTH}")
    if winner:
        print(f"\n{BOLD}Recommendation: {winner}{RESET}")
    else:
        print(f"\n{BOLD}No recommendation -- nothing judged yet.{RESET}")
    print(f"\nWrote {REPORTS_DIR}/generator_model_comparison.json + .md")


if __name__ == "__main__":
    main()
