import json
import sqlite3
from collections import defaultdict
from glob import glob
from pathlib import Path
from statistics import mean, pstdev

from tabulate import tabulate

# ----------------------------------------------------------------------------
# Golden-Question Evaluation Report
# ----------------------------------------------------------------------------
# PURPOSE: read the human scores + target-answer corrections collected by
# scripts/11_human_golden_question_evaluation.py (the Streamlit rating UI) from
# SQLite and roll them into ONE report (JSON + Markdown) describing the QUALITY
# OF THE GOLD QUESTION SET itself -- not the RAG system's answers.
#
# This is the companion to scripts/11. scripts/11 validates the
# test set that every downstream evaluation (13 generator comparison,
# 18 safety gates) is measured against; this script turns that validation
# into (a) aggregate test-set quality metrics and (b) a per-question CURATION
# flag list a maintainer acts on by hand.
#
# Reads ONLY from:
#   reports/human_golden_question_evaluation/human_golden_question_evaluation.db
#   data/09_evaluation/*/gold_questions.json   (question text + gold answers)
# Writes NOTHING back to the gold set. Target-answer corrections captured in the
# DB are REPORTED (old vs corrected, side by side) but never applied -- ground
# truth is never mutated automatically. Applying accepted corrections back to
# gold_questions.json is a deliberate, separate manual step.
#
# OUTPUT (under reports/human_golden_question_evaluation/):
#   human_golden_question_evaluation_report.json   machine-readable
#   human_golden_question_evaluation_report.md      human-readable
# ----------------------------------------------------------------------------

REPORT_DIR = Path("reports/human_golden_question_evaluation")
DB_PATH = REPORT_DIR / "human_golden_question_evaluation.db"
GOLD_GLOB = "data/09_evaluation/*/gold_questions.json"

JSON_OUT = REPORT_DIR / "human_golden_question_evaluation_report.json"
MD_OUT = REPORT_DIR / "human_golden_question_evaluation_report.md"

# ids 001-025 in every category are human-authored gold questions; anything
# above that is AI-authored. Kept in sync with scripts/11.
HUMAN_QID_MAX = 25

# The five metrics, with their per-metric 1..5 label sets (kept in sync with
# scripts/11 so the distribution tables read the same as the rating UI).
METRICS = [
    ("clarity",        "Clarity",        ["Very Poor", "Poor", "Average", "Good", "Excellent"]),
    ("relevance",      "Relevance",      ["Not Relevant", "Slightly", "Moderate", "Relevant", "Highly Relevant"]),
    ("difficulty",     "Difficulty",     ["Too Easy/Hard", "Poor Level", "Adequate", "Good Level", "Perfect Level"]),
    ("answer_quality", "Answer Quality", ["Very Poor", "Poor", "Adequate", "Good", "Excellent"]),
    ("uniqueness",     "Uniqueness",     ["Very Redundant", "Redundant", "Somewhat Unique", "Unique", "Very Unique"]),
]
METRIC_KEYS = [m[0] for m in METRICS]

# Per-question mean at/below which a metric flags the question for a curation
# action. These are advisory -- the report never drops or edits anything.
FLAG_RULES = {
    "relevance":      (2.5, "DROP — off-topic / not an insurance-law question"),
    "answer_quality": (2.5, "FIX — target answer needs legal correction"),
    "clarity":        (2.5, "REWORD — question is unclear/ambiguous"),
    "uniqueness":     (2.5, "DEDUPE — redundant with another question"),
    "difficulty":     (2.0, "REVIEW — difficulty is off (too easy/hard)"),
}


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def parse_question_id(qid):
    """'guidance_qid_026' -> ('guidance', 'ai'); 'law_qid_007' -> ('law','human')."""
    try:
        prefix, num_str = qid.rsplit("_qid_", 1)
        num = int(num_str)
    except (ValueError, AttributeError):
        return "unknown", "unknown"
    return prefix, ("human" if num <= HUMAN_QID_MAX else "ai")


def load_gold():
    """question_id -> {question, expected_answer, ...} from the gold sets."""
    gold = {}
    for path in sorted(glob(GOLD_GLOB)):
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for e in entries:
            qid = e.get("question_id")
            if qid:
                gold[qid] = e
    return gold


def load_db():
    """Returns (evaluators, responses, target_edits) as lists of dicts."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ev = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM evaluators")}
    responses = [dict(r) for r in conn.execute(
        "SELECT evaluator_id, question_id, clarity, relevance, difficulty, "
        "answer_quality, uniqueness, total_score, submitted_at FROM responses"
    )]
    # latest correction per question (most recent edit wins)
    edits = {}
    for r in conn.execute(
        "SELECT question_id, evaluator_name, new_answer, edited_at "
        "FROM target_edits ORDER BY edited_at ASC, id ASC"
    ):
        edits[r["question_id"]] = dict(r)  # ASC order => last write kept = newest
    conn.close()
    return ev, responses, edits


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def _mean(xs):
    return round(mean(xs), 3) if xs else None


def metric_means(rows):
    """Mean per metric over a list of response dicts (skips NULL scores)."""
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = _mean(vals)
    present = [out[k] for k in METRIC_KEYS if out[k] is not None]
    out["combined"] = round(mean(present), 3) if present else None
    return out


def build_report():
    gold = load_gold()
    evaluators, responses, edits = load_db()

    # group responses by question
    by_q = defaultdict(list)
    for r in responses:
        by_q[r["question_id"]].append(r)

    gold_ids = set(gold)
    evaluated_ids = set(by_q)
    human_ids = {q for q in gold_ids if parse_question_id(q)[1] == "human"}
    ai_ids = {q for q in gold_ids if parse_question_id(q)[1] == "ai"}

    coverage = {
        "evaluators": len(evaluators),
        "gold_questions": len(gold_ids),
        "questions_evaluated": len(evaluated_ids & gold_ids),
        "questions_unevaluated": len(gold_ids - evaluated_ids),
        "total_responses": len(responses),
        "responses_per_question_mean": round(
            len(responses) / len(evaluated_ids), 2) if evaluated_ids else 0,
        "gold_human_authored": len(human_ids),
        "gold_ai_authored": len(ai_ids),
    }

    # overall / per-source / per-category means
    overall = metric_means(responses)
    human_rows = [r for r in responses if parse_question_id(r["question_id"])[1] == "human"]
    ai_rows = [r for r in responses if parse_question_id(r["question_id"])[1] == "ai"]
    by_source = {"human_authored": metric_means(human_rows),
                 "ai_authored": metric_means(ai_rows)}

    by_category = {}
    cat_rows = defaultdict(list)
    for r in responses:
        cat_rows[parse_question_id(r["question_id"])[0]].append(r)
    for cat, rows in sorted(cat_rows.items()):
        by_category[cat] = metric_means(rows)

    # score distribution (1..5) per metric
    distribution = {}
    for k in METRIC_KEYS:
        counts = {i: 0 for i in range(1, 6)}
        for r in responses:
            v = r.get(k)
            if v in counts:
                counts[v] += 1
        distribution[k] = counts

    # inter-evaluator agreement: mean population stdev per metric over questions
    # that were scored by >= 2 evaluators (lower = more agreement)
    agreement = {}
    multi = {q: rows for q, rows in by_q.items() if len(rows) >= 2}
    for k in METRIC_KEYS:
        spreads = []
        for rows in multi.values():
            vals = [r[k] for r in rows if r.get(k) is not None]
            if len(vals) >= 2:
                spreads.append(pstdev(vals))
        agreement[k] = round(mean(spreads), 3) if spreads else None
    agreement["_questions_with_overlap"] = len(multi)

    # per-question curation flags (flag-only)
    flags = []
    for q in sorted(evaluated_ids & gold_ids):
        rows = by_q[q]
        qm = metric_means(rows)
        reasons = []
        for metric, (thresh, label) in FLAG_RULES.items():
            m = qm.get(metric)
            if m is not None and m <= thresh:
                reasons.append({"metric": metric, "mean": m, "action": label})
        if reasons:
            cat, source = parse_question_id(q)
            flags.append({
                "question_id": q,
                "category": cat,
                "source": source,
                "n_evaluators": len(rows),
                "combined_mean": qm["combined"],
                "question": (gold.get(q, {}).get("question") or "").strip(),
                "reasons": reasons,
            })
    flags.sort(key=lambda f: (f["combined_mean"] if f["combined_mean"] is not None else 99))

    # target-answer corrections captured in the DB (reported, not applied)
    corrections = []
    for q, e in edits.items():
        original = (gold.get(q, {}).get("expected_answer") or "").strip()
        corrected = (e["new_answer"] or "").strip()
        if corrected and corrected != original:
            corrections.append({
                "question_id": q,
                "category": parse_question_id(q)[0],
                "source": parse_question_id(q)[1],
                "edited_by": e["evaluator_name"],
                "edited_at": e["edited_at"],
                "gold_answer": original,
                "corrected_answer": corrected,
            })
    corrections.sort(key=lambda c: c["question_id"])

    return {
        "coverage": coverage,
        "overall_means": overall,
        "means_by_source": by_source,
        "means_by_category": by_category,
        "score_distribution": distribution,
        "inter_evaluator_agreement": agreement,
        "curation_flags": flags,
        "target_answer_corrections": corrections,
    }


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------

def _truncate(text, n=90):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _blockquote(text):
    """Render text IN FULL as a markdown blockquote, preserving line breaks and
    never truncating — a reviewer needs the complete legal wording to judge a
    correction. Returns "" for empty input."""
    text = (text or "").strip()
    if not text:
        return ""
    return "\n".join("> " + line if line else ">" for line in text.split("\n"))


def render_md(rep):
    cov = rep["coverage"]
    md = ["# Golden-Question Evaluation Report", ""]
    md.append(
        "Human validation of the gold question set collected by "
        "`scripts/11_human_golden_question_evaluation.py`. Measures the quality "
        "of the **questions and target answers** — the ground truth every "
        "downstream evaluation is scored against — not the RAG system's answers."
    )
    md.append("")

    md.append("## Coverage")
    md.append("")
    md.append(tabulate([
        ["Evaluators", cov["evaluators"]],
        ["Gold questions", cov["gold_questions"]],
        ["  — human-authored", cov["gold_human_authored"]],
        ["  — AI-authored", cov["gold_ai_authored"]],
        ["Questions evaluated", cov["questions_evaluated"]],
        ["Questions NOT yet evaluated", cov["questions_unevaluated"]],
        ["Total responses", cov["total_responses"]],
        ["Mean responses / evaluated question", cov["responses_per_question_mean"]],
    ], headers=["Metric", "Value"], tablefmt="github"))
    md.append("")

    # means: overall + by source
    md.append("## Metric means (1–5)")
    md.append("")
    md.append("`combined` = mean of the five metric means. AI-authored vs "
              "human-authored is the key comparison — it shows whether the "
              "auto-generated questions hold up.")
    md.append("")
    header = ["Scope"] + [lbl for _, lbl, _ in METRICS] + ["Combined"]
    rows = []
    for label, block in [("All", rep["overall_means"]),
                         ("Human-authored", rep["means_by_source"]["human_authored"]),
                         ("AI-authored", rep["means_by_source"]["ai_authored"])]:
        rows.append([label] + [block.get(k) for k in METRIC_KEYS] + [block.get("combined")])
    for cat, block in rep["means_by_category"].items():
        rows.append([f"cat: {cat}"] + [block.get(k) for k in METRIC_KEYS] + [block.get("combined")])
    md.append(tabulate(rows, headers=header, tablefmt="github"))
    md.append("")

    # distributions
    md.append("## Score distribution")
    md.append("")
    for k, lbl, labels in METRICS:
        dist = rep["score_distribution"][k]
        header = [lbl] + [f"{i} {labels[i-1]}" for i in range(1, 6)]
        md.append(tabulate([["count"] + [dist[i] for i in range(1, 6)]],
                           headers=header, tablefmt="github"))
        md.append("")

    # agreement
    agr = rep["inter_evaluator_agreement"]
    md.append("## Inter-evaluator agreement")
    md.append("")
    md.append(f"Population std-dev per metric across the "
              f"**{agr['_questions_with_overlap']}** questions scored by ≥2 "
              f"evaluators (lower = more agreement; blank = no overlap yet).")
    md.append("")
    md.append(tabulate([[lbl, agr.get(k)] for k, lbl, _ in METRICS],
                       headers=["Metric", "Mean std-dev"], tablefmt="github"))
    md.append("")

    # curation flags
    flags = rep["curation_flags"]
    md.append("## Curation flags")
    md.append("")
    if not flags:
        md.append("_No questions fell below the flag thresholds._")
    else:
        md.append(f"**{len(flags)}** question(s) fell below a threshold. These are "
                  "advisory — nothing is dropped or edited automatically.")
        md.append("")
        rows = []
        for f in flags:
            actions = "; ".join(f"{r['metric']}={r['mean']} → {r['action']}" for r in f["reasons"])
            rows.append([f["question_id"], f["source"], f["combined_mean"],
                        f["n_evaluators"], _truncate(f["question"], 55), actions])
        md.append(tabulate(rows, headers=["Question ID", "Src", "Comb.", "#Ev",
                                          "Question", "Flags → action"],
                           tablefmt="github"))
    md.append("")

    # corrections
    corr = rep["target_answer_corrections"]
    md.append("## Target-answer corrections (captured, NOT applied)")
    md.append("")
    if not corr:
        md.append("_No target-answer corrections recorded._")
    else:
        md.append(f"**{len(corr)}** target answer(s) were corrected by evaluators in "
                  "the rating UI. They are **not** written back to "
                  "`data/09_evaluation/*/gold_questions.json` — review these and "
                  "apply the accepted ones by hand, then re-run "
                  "`scripts/13_compare_generator_models.py --rescore`.")
        md.append("")
        for c in corr:
            md.append(f"### `{c['question_id']}`  ({c['source']}, edited by {c['edited_by']})")
            md.append("")
            md.append("**Gold (current):**")
            md.append("")
            md.append(_blockquote(c["gold_answer"]) or "> _(empty)_")
            md.append("")
            md.append("**Corrected:**")
            md.append("")
            md.append(_blockquote(c["corrected_answer"]))
            md.append("")

    return "\n".join(md)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        raise SystemExit(
            f"No evaluation DB at {DB_PATH}. Run "
            "scripts/11_human_golden_question_evaluation.py and collect scores first."
        )
    rep = build_report()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_md(rep)
    MD_OUT.write_text(md, encoding="utf-8")

    cov = rep["coverage"]
    print(f"Evaluators: {cov['evaluators']}  |  responses: {cov['total_responses']}  "
          f"|  evaluated {cov['questions_evaluated']}/{cov['gold_questions']} gold questions")
    if rep["overall_means"]["combined"] is not None:
        print("Overall combined mean: "
              f"{rep['overall_means']['combined']}   "
              f"(human={rep['means_by_source']['human_authored']['combined']}, "
              f"ai={rep['means_by_source']['ai_authored']['combined']})")
    print(f"Curation flags: {len(rep['curation_flags'])}  |  "
          f"target-answer corrections: {len(rep['target_answer_corrections'])}")
    print(f"\nWrote:\n  {JSON_OUT}\n  {MD_OUT}")


if __name__ == "__main__":
    main()
