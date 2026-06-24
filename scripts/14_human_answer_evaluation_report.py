import json
import sqlite3
import time
from pathlib import Path
from tabulate import tabulate

# ----------------------------------------------------------------------------
# RAG Human Evaluation Report (Internship Proposal §11)
# ----------------------------------------------------------------------------
# PURPOSE: reads human scores collected by eval_ui.py from SQLite and
# produces the final human-verified RAGAS-style evaluation report.
#
# reports/human_answer_evaluation/human_answer_evaluation.db
# reports/human_answer_evaluation/human_answer_evaluation_report.json
# reports/human_answer_evaluation/human_answer_evaluation_report.md
#
# What it calculates:
#   1. Per-question avg human score across all evaluators (per metric + total)
#   2. Per-metric macro-avg across all 100 questions
#   3. Per-evaluator stats (coverage, avg scores) -- detects lazy/outlier raters
#   4. Inter-annotator agreement per metric (avg pairwise score difference)
#      on questions rated by more than one evaluator
#   5. Human vs automated proxy score comparison (script 10 vs eval_ui.py)
#   6. Hallucination audit -- questions where avg faithfulness < 3.0
# ----------------------------------------------------------------------------

DB_PATH = Path(
    "reports/human_answer_evaluation/"
    "human_answer_evaluation.db"
)
REPORTS_DIR = Path(
    "reports/human_answer_evaluation"
)
GENERATOR_RESULTS_DIR = Path(
    "reports/generator_model_comparison/"
    "SeaLLMs-v3-7B-Chat-Q4_K_M"
)

METRICS = [
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
]

METRIC_LABELS = {
    "context_relevance":  "Context Relevance",
    "faithfulness":       "Faithfulness",
    "answer_relevance":   "Answer Relevance",
    "answer_correctness": "Answer Correctness",
}

BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH


# ----------------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------------

def get_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}.\n"
            "Run eval_ui.py first and collect human scores."
        )
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_responses():
    """Returns all submitted responses with evaluator name and question index."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            e.name           AS evaluator_name,
            r.evaluator_id,
            r.question_idx,
            r.context_relevance,
            r.faithfulness,
            r.answer_relevance,
            r.answer_correctness,
            r.total_score,
            r.submitted_at
        FROM responses r
        JOIN evaluators e ON e.id = r.evaluator_id
        ORDER BY r.question_idx, e.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_evaluators():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, created_at FROM evaluators ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_assignments():
    """Returns {evaluator_id: [question_idx, ...]}"""
    conn = get_db()
    rows = conn.execute(
        "SELECT evaluator_id, question_idx FROM assignments ORDER BY evaluator_id, position"
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        result.setdefault(r["evaluator_id"], []).append(r["question_idx"])
    return result


# ----------------------------------------------------------------------------
# Load automated proxy scores from script 10 JSONL
# ----------------------------------------------------------------------------

def find_jsonl():
    path = GENERATOR_RESULTS_DIR / "generated_answer_results.jsonl"

    if not path.exists():
        return None, None

    return path, "SeaLLM v3 7B"


def load_proxy_scores():
    """Returns {question_text: {metric: score, ...}} from script 10 JSONL."""
    path, label = find_jsonl()
    if not path:
        return {}, None
    proxy = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                q     = entry.get("question", "")
                if q and entry.get("scores"):
                    proxy[q] = {
                        "scores":   entry["scores"],
                        "combined": entry.get("combined"),
                        "idx":      None,
                    }
            except json.JSONDecodeError:
                pass
    return proxy, label


def load_questions_from_jsonl():
    """Returns list of question dicts from script 10 JSONL, indexed by position."""
    path, _ = find_jsonl()
    if not path:
        return []
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return questions


# ----------------------------------------------------------------------------
# Core calculations
# ----------------------------------------------------------------------------

def per_question_stats(responses, questions):
    """
    Returns dict keyed by question_idx:
    {
        "question":     str,
        "category":     str,
        "evaluator_count": int,
        "metric_avgs":  {metric: avg},
        "total_avg":    float,
        "evaluators":   [name, ...],
    }
    """
    by_q = {}
    for r in responses:
        qidx = r["question_idx"]
        by_q.setdefault(qidx, []).append(r)

    stats = {}
    for qidx, rows in by_q.items():
        q_text   = questions[qidx]["question"]   if qidx < len(questions) else f"Q{qidx}"
        category = questions[qidx].get("category", "unknown") if qidx < len(questions) else "unknown"

        metric_avgs = {}
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None]
            metric_avgs[m] = round(sum(vals) / len(vals), 4) if vals else None

        total_vals = [r["total_score"] for r in rows if r["total_score"] is not None]
        total_avg  = round(sum(total_vals) / len(total_vals), 4) if total_vals else None

        stats[qidx] = {
            "question":        q_text,
            "category":        category,
            "evaluator_count": len(rows),
            "evaluators":      [r["evaluator_name"] for r in rows],
            "metric_avgs":     metric_avgs,
            "total_avg":       total_avg,
        }

    return stats


def macro_avg_per_metric(q_stats):
    """Macro-avg of each metric across all rated questions."""
    result = {}
    for m in METRICS:
        vals = [
            s["metric_avgs"][m]
            for s in q_stats.values()
            if s["metric_avgs"].get(m) is not None
        ]
        result[m] = round(sum(vals) / len(vals), 4) if vals else None
    return result


def per_evaluator_stats(responses, assignments):
    """Per-evaluator coverage + avg scores."""
    by_eval = {}
    for r in responses:
        eid = r["evaluator_id"]
        by_eval.setdefault(eid, {
            "name":      r["evaluator_name"],
            "responses": [],
        })
        by_eval[eid]["responses"].append(r)

    stats = []
    for eid, data in by_eval.items():
        rows     = data["responses"]
        assigned = len(assignments.get(eid, []))
        completed = len(rows)

        metric_avgs = {}
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None]
            metric_avgs[m] = round(sum(vals) / len(vals), 4) if vals else None

        total_vals = [r["total_score"] for r in rows]
        total_avg  = round(sum(total_vals) / len(total_vals), 2) if total_vals else None

        stats.append({
            "evaluator_id":  eid,
            "name":          data["name"],
            "assigned":      assigned,
            "completed":     completed,
            "coverage":      f"{completed}/{assigned}",
            "metric_avgs":   metric_avgs,
            "total_avg":     total_avg,
        })

    return sorted(stats, key=lambda s: s["name"])


def inter_annotator_agreement(q_stats):
    """
    For questions rated by >1 evaluator, compute avg pairwise absolute
    difference per metric. Lower = more agreement.
    Returns {metric: avg_pairwise_diff} and count of multi-rated questions.
    """
    from itertools import combinations

    by_metric = {m: [] for m in METRICS}
    multi_count = 0

    for qidx, stat in q_stats.items():
        if stat["evaluator_count"] < 2:
            continue
        multi_count += 1
        # We need per-evaluator scores for this question -- stored in responses
        # but q_stats only has avgs. We'll return the avg diff from responses
        # directly in the caller.

    return by_metric, multi_count


def inter_annotator_from_responses(responses):
    """
    Computes avg pairwise absolute score difference per metric
    on questions that were rated by more than one evaluator.
    Lower = higher agreement (0 = perfect agreement, 4 = max disagreement).
    """
    from itertools import combinations

    by_q = {}
    for r in responses:
        by_q.setdefault(r["question_idx"], []).append(r)

    diffs = {m: [] for m in METRICS}
    multi_count = 0

    for qidx, rows in by_q.items():
        if len(rows) < 2:
            continue
        multi_count += 1
        for r1, r2 in combinations(rows, 2):
            for m in METRICS:
                v1, v2 = r1.get(m), r2.get(m)
                if v1 is not None and v2 is not None:
                    diffs[m].append(abs(v1 - v2))

    result = {}
    for m in METRICS:
        vals = diffs[m]
        result[m] = round(sum(vals) / len(vals), 4) if vals else None

    return result, multi_count


def human_vs_proxy(q_stats, questions, proxy_scores):
    """
    Compares avg human score vs automated proxy score per metric
    for questions that have both.
    Returns list of {question_idx, question, metric, human_avg, proxy, diff}
    and macro summary {metric: {human_avg, proxy_avg, diff}}.
    """
    rows      = []
    by_metric = {m: {"human": [], "proxy": []} for m in METRICS}

    for qidx, stat in q_stats.items():
        if qidx >= len(questions):
            continue
        q_text  = questions[qidx].get("question", "")
        p_entry = proxy_scores.get(q_text)
        if not p_entry:
            continue

        for m in METRICS:
            h_val = stat["metric_avgs"].get(m)
            p_val = p_entry["scores"].get(m)
            if h_val is None or p_val is None:
                continue
            rows.append({
                "question_idx": qidx,
                "question":     q_text,
                "metric":       m,
                "human_avg":    h_val,
                "proxy":        p_val,
                "diff":         round(h_val - p_val, 4),
            })
            by_metric[m]["human"].append(h_val)
            by_metric[m]["proxy"].append(p_val)

    macro = {}
    for m in METRICS:
        h_vals = by_metric[m]["human"]
        p_vals = by_metric[m]["proxy"]
        if h_vals and p_vals:
            macro[m] = {
                "human_avg": round(sum(h_vals) / len(h_vals), 4),
                "proxy_avg": round(sum(p_vals) / len(p_vals), 4),
                "diff":      round(
                    sum(h_vals) / len(h_vals) - sum(p_vals) / len(p_vals), 4
                ),
            }

    return rows, macro


def hallucination_audit(q_stats, questions, threshold=3.0):
    """Questions where avg faithfulness < threshold -- mandatory review."""
    flagged = []
    for qidx, stat in q_stats.items():
        faith = stat["metric_avgs"].get("faithfulness")
        if faith is not None and faith < threshold:
            flagged.append({
                "question_idx": qidx,
                "question":     stat["question"],
                "category":     stat["category"],
                "faithfulness": faith,
                "total_avg":    stat["total_avg"],
                "evaluators":   stat["evaluators"],
            })
    return sorted(flagged, key=lambda x: x["faithfulness"])


# ----------------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------------

def print_report(
    q_stats,
    macro_avgs,
    eval_stats,
    iaa,
    iaa_multi_count,
    human_proxy_macro,
    flagged,
    proxy_label,
    total_responses,
    total_questions_rated,
):
    banner = "=" * SEPARATOR_WIDTH

    # --- Overview ---
    print(f"\n{banner}")
    print(f"{BOLD}RAG HUMAN EVALUATION REPORT{RESET}")
    print(banner)
    print(tabulate(
        [
            ["Total evaluators",         len(eval_stats)],
            ["Total responses submitted", total_responses],
            ["Questions rated (≥1 eval)", total_questions_rated],
            ["Questions rated by >1 eval", iaa_multi_count],
            ["Proxy model (script 10)",   proxy_label or "not found"],
        ],
        tablefmt="grid",
    ))

    # --- Macro metric averages ---
    print(f"\n{banner}")
    print(f"{BOLD}MACRO-AVG HUMAN SCORES (across all rated questions){RESET}")
    print(banner)
    macro_rows = []
    combined   = []
    for m in METRICS:
        v = macro_avgs.get(m)
        macro_rows.append([METRIC_LABELS[m], f"{v:.4f}" if v else "N/A"])
        if v:
            combined.append(v)
    combined_avg = sum(combined) / len(combined) if combined else 0
    macro_rows.append(["── Combined avg ──", f"{combined_avg:.4f}"])
    print(tabulate(macro_rows, headers=["Metric", "Avg (1-5)"], tablefmt="grid"))

    # --- Inter-annotator agreement ---
    print(f"\n{banner}")
    print(f"{BOLD}INTER-ANNOTATOR AGREEMENT (avg pairwise |diff|, lower = better){RESET}")
    print(f"{BOLD}Based on {iaa_multi_count} questions rated by more than one evaluator{RESET}")
    print(banner)
    iaa_rows = []
    for m in METRICS:
        v = iaa.get(m)
        if v is None:
            iaa_rows.append([METRIC_LABELS[m], "N/A", "—"])
        else:
            if v <= 0.5:
                agreement = f"{GREEN}Strong{RESET}"
            elif v <= 1.0:
                agreement = f"{YELLOW}Moderate{RESET}"
            else:
                agreement = f"{RED}Weak{RESET}"
            iaa_rows.append([METRIC_LABELS[m], f"{v:.4f}", agreement])
    print(tabulate(iaa_rows, headers=["Metric", "Avg |diff|", "Agreement"], tablefmt="grid"))

    # --- Per-evaluator stats ---
    print(f"\n{banner}")
    print(f"{BOLD}PER-EVALUATOR STATS{RESET}")
    print(banner)
    eval_rows = []
    for s in eval_stats:
        metric_cols = [
            f"{s['metric_avgs'].get(m):.2f}" if s["metric_avgs"].get(m) else "N/A"
            for m in METRICS
        ]
        eval_rows.append([
            s["name"],
            s["coverage"],
            *metric_cols,
            f"{s['total_avg']:.2f}" if s["total_avg"] else "N/A",
        ])
    print(tabulate(
        eval_rows,
        headers=["Evaluator", "Coverage"] + [METRIC_LABELS[m] for m in METRICS] + ["Total /20"],
        tablefmt="grid",
    ))

    # --- Human vs proxy comparison ---
    if human_proxy_macro:
        print(f"\n{banner}")
        print(f"{BOLD}HUMAN vs AUTOMATED PROXY SCORES (macro-avg){RESET}")
        print(f"{BOLD}Proxy = script 10 automated scores | Human = eval_ui.py scores{RESET}")
        print(f"{BOLD}Diff = Human − Proxy (positive = humans rated higher){RESET}")
        print(banner)
        hp_rows = []
        for m in METRICS:
            v = human_proxy_macro.get(m)
            if not v:
                hp_rows.append([METRIC_LABELS[m], "N/A", "N/A", "N/A"])
                continue
            diff_str = f"{v['diff']:+.4f}"
            if abs(v["diff"]) <= 0.3:
                diff_color = GREEN
            elif abs(v["diff"]) <= 0.7:
                diff_color = YELLOW
            else:
                diff_color = RED
            hp_rows.append([
                METRIC_LABELS[m],
                f"{v['human_avg']:.4f}",
                f"{v['proxy_avg']:.4f}",
                f"{diff_color}{diff_str}{RESET}",
            ])
        print(tabulate(
            hp_rows,
            headers=["Metric", "Human Avg", "Proxy Avg", "Diff"],
            tablefmt="grid",
        ))

    # --- Hallucination audit ---
    print(f"\n{banner}")
    if flagged:
        print(f"{BOLD}{RED}HALLUCINATION AUDIT -- {len(flagged)} questions with avg Faithfulness < 3.0{RESET}")
        print(banner)
        for f in flagged:
            print(f"\n  [{f['category']}] Q{f['question_idx']} | Faith={f['faithfulness']:.2f} | Total={f['total_avg']:.2f}")
            print(f"  Rated by: {', '.join(f['evaluators'])}")
            print(f"  {f['question'][:100]}{'…' if len(f['question']) > 100 else ''}")
    else:
        print(f"{BOLD}{GREEN}HALLUCINATION AUDIT -- No questions flagged (all avg Faithfulness ≥ 3.0){RESET}")
        print(banner)

    # --- Per-question detail (bottom, long) ---
    print(f"\n{banner}")
    print(f"{BOLD}PER-QUESTION HUMAN SCORES (avg across evaluators){RESET}")
    print(banner)
    pq_rows = []
    for qidx in sorted(q_stats.keys()):
        stat = q_stats[qidx]
        metric_cols = [
            f"{stat['metric_avgs'].get(m):.2f}" if stat["metric_avgs"].get(m) else "N/A"
            for m in METRICS
        ]
        flag = f" {RED}⚑{RESET}" if (stat["metric_avgs"].get("faithfulness") or 5) < 3.0 else ""
        pq_rows.append([
            f"Q{qidx}",
            stat["category"][:8],
            stat["evaluator_count"],
            *metric_cols,
            f"{stat['total_avg']:.2f}" if stat["total_avg"] else "N/A",
            flag.strip(),
        ])
    print(tabulate(
        pq_rows,
        headers=["Q", "Cat", "Evals"] + [METRIC_LABELS[m][:8] for m in METRICS] + ["Total", "Flag"],
        tablefmt="grid",
    ))


# ----------------------------------------------------------------------------
# Write reports
# ----------------------------------------------------------------------------

def write_reports(
    q_stats,
    macro_avgs,
    eval_stats,
    iaa,
    iaa_multi_count,
    human_proxy_macro,
    flagged,
    proxy_label,
    total_responses,
    total_questions_rated,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_vals = [v for v in macro_avgs.values() if v is not None]
    combined_avg  = round(sum(combined_vals) / len(combined_vals), 4) if combined_vals else 0.0

    payload = {
        "generated_at":          int(time.time()),
        "proxy_model":           proxy_label,
        "total_evaluators":      len(eval_stats),
        "total_responses":       total_responses,
        "questions_rated":       total_questions_rated,
        "multi_rated_questions": iaa_multi_count,
        "macro_avgs":            macro_avgs,
        "combined_avg":          combined_avg,
        "inter_annotator_agreement": iaa,
        "per_evaluator":         eval_stats,
        "human_vs_proxy":        human_proxy_macro,
        "hallucination_audit":   flagged,
        "per_question":          {
            str(qidx): stat for qidx, stat in q_stats.items()
        },
    }

    json_path = REPORTS_DIR / "human_answer_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Markdown
    md = ["# RAG Human Evaluation Report\n\n"]
    md.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.append(f"**Proxy model (script 10):** {proxy_label or 'not found'}  \n")
    md.append(f"**Evaluators:** {len(eval_stats)}  \n")
    md.append(f"**Responses:** {total_responses}  \n")
    md.append(f"**Questions rated:** {total_questions_rated}  \n\n")

    md.append("## Macro-Avg Human Scores\n\n")
    rows = [[METRIC_LABELS[m], f"{macro_avgs.get(m):.4f}" if macro_avgs.get(m) else "N/A"]
            for m in METRICS]
    rows.append(["── Combined ──", f"{combined_avg:.4f}"])
    md.append(tabulate(rows, headers=["Metric", "Avg (1-5)"], tablefmt="github") + "\n\n")

    md.append("## Inter-Annotator Agreement\n\n")
    iaa_rows = [
        [METRIC_LABELS[m], f"{iaa.get(m):.4f}" if iaa.get(m) else "N/A"]
        for m in METRICS
    ]
    md.append(tabulate(iaa_rows, headers=["Metric", "Avg |diff|"], tablefmt="github") + "\n\n")

    md.append("## Per-Evaluator Stats\n\n")
    eval_rows = []
    for s in eval_stats:
        eval_rows.append([
            s["name"], s["coverage"],
            *[f"{s['metric_avgs'].get(m):.2f}" if s["metric_avgs"].get(m) else "N/A" for m in METRICS],
            f"{s['total_avg']:.2f}" if s["total_avg"] else "N/A",
        ])
    md.append(tabulate(
        eval_rows,
        headers=["Evaluator", "Coverage"] + [METRIC_LABELS[m] for m in METRICS] + ["Total /20"],
        tablefmt="github",
    ) + "\n\n")

    if human_proxy_macro:
        md.append("## Human vs Automated Proxy Scores\n\n")
        hp_rows = []
        for m in METRICS:
            v = human_proxy_macro.get(m)
            if v:
                hp_rows.append([METRIC_LABELS[m], f"{v['human_avg']:.4f}",
                                 f"{v['proxy_avg']:.4f}", f"{v['diff']:+.4f}"])
        md.append(tabulate(hp_rows, headers=["Metric", "Human", "Proxy", "Diff"],
                           tablefmt="github") + "\n\n")

    md.append("## Hallucination Audit\n\n")
    if flagged:
        md.append(f"**{len(flagged)} questions flagged (avg Faithfulness < 3.0):**\n\n")
        for f in flagged:
            md.append(f"- Q{f['question_idx']} [{f['category']}] "
                      f"Faith={f['faithfulness']:.2f} | {f['question'][:80]}\n")
    else:
        md.append("No questions flagged — all avg Faithfulness ≥ 3.0.\n\n")

    md_path = REPORTS_DIR / "human_answer_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    # Hallucination audit text file (mandatory pre-demo gate per proposal §11)
    audit_path = REPORTS_DIR / "hallucination_audit.txt"
    with open(audit_path, "w", encoding="utf-8") as f:
        if flagged:
            f.write(f"HALLUCINATION AUDIT -- {len(flagged)} questions flagged\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in flagged:
                f.write(f"Q{item['question_idx']} [{item['category']}]\n")
                f.write(f"  Faithfulness avg : {item['faithfulness']:.2f}\n")
                f.write(f"  Total score avg  : {item['total_avg']:.2f}\n")
                f.write(f"  Rated by         : {', '.join(item['evaluators'])}\n")
                f.write(f"  Question         : {item['question']}\n\n")
        else:
            f.write("No questions flagged. All avg Faithfulness >= 3.0.\n")

    print(f"\nSaved JSON report          : {json_path}")
    print(f"Saved Markdown report      : {md_path}")
    print(f"Saved hallucination audit  : {audit_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    start = time.time()

    print("Loading human responses from database...")
    responses   = load_all_responses()
    evaluators  = load_evaluators()
    assignments = load_assignments()

    if not responses:
        print("No responses found in the database yet.")
        print("Run eval_ui.py and collect human scores first.")
        return

    print(f"  {len(responses)} responses from {len(evaluators)} evaluators.")

    print("Loading questions from script 10 JSONL...")
    questions    = load_questions_from_jsonl()
    proxy_scores, proxy_label = load_proxy_scores()

    if not questions:
        print("  Warning: no JSONL found -- question text will show as Q<idx>.")
    else:
        print(f"  {len(questions)} questions loaded from {proxy_label or 'unknown model'}.")

    print("Computing stats...")
    q_stats      = per_question_stats(responses, questions)
    macro_avgs   = macro_avg_per_metric(q_stats)
    eval_stats   = per_evaluator_stats(responses, assignments)
    iaa, iaa_multi = inter_annotator_from_responses(responses)
    _, hp_macro  = human_vs_proxy(q_stats, questions, proxy_scores)
    flagged      = hallucination_audit(q_stats, questions)

    total_responses       = len(responses)
    total_questions_rated = len(q_stats)

    print_report(
        q_stats, macro_avgs, eval_stats,
        iaa, iaa_multi,
        hp_macro, flagged,
        proxy_label,
        total_responses, total_questions_rated,
    )

    write_reports(
        q_stats, macro_avgs, eval_stats,
        iaa, iaa_multi,
        hp_macro, flagged,
        proxy_label,
        total_responses, total_questions_rated,
    )

    elapsed = time.time() - start
    print(f"\nTotal runtime : {elapsed:.2f} seconds")
    print("Human evaluation report completed successfully.")


if __name__ == "__main__":
    main()