import json
import sqlite3
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from tabulate import tabulate

# ----------------------------------------------------------------------------
# RAG Human Evaluation Report (Internship Proposal §11)
# ----------------------------------------------------------------------------
# PURPOSE: reads the human scores collected by script 13 (the Streamlit eval UI)
# from SQLite and merges everything into ONE unified human-verified evaluation
# report, plus a SEPARATE hallucination report.
#
# Inputs
#   reports/human_answer_evaluation/human_answer_evaluation.db   (from script 13)
#   reports/generator_model_comparison/.../generated_answer_results.jsonl (script 10)
#
# Outputs (all under reports/human_answer_evaluation/)
#   human_answer_evaluation_report.json   unified machine-readable report
#   human_answer_evaluation_report.md     unified human-readable report
#   hallucination_report.json             standalone hallucination data
#   hallucination_report.md               standalone hallucination report (Markdown)
#
# What the unified report calculates
#   1. Per-question avg human score across all evaluators (per metric + total)
#   2. Per-metric macro-avg across every rated question
#   3. Per-evaluator stats (coverage, avg scores, hallucination flags raised)
#   4. Inter-annotator agreement per metric + on the hallucination flag
#   5. Human vs automated proxy comparison (script 13 humans vs script 10 judge)
#
# What the SEPARATE hallucination report calculates
#   - Every answer a human flagged as hallucinated (real Yes/No flag, not a proxy)
#   - Severity by evaluator consensus (confirmed = majority, disputed = minority)
#   - Overall + per-category hallucination rate, cross-referenced with faithfulness
# ----------------------------------------------------------------------------

DB_PATH = Path(
    "reports/human_answer_evaluation/human_answer_evaluation.db"
)
REPORTS_DIR = Path(
    "reports/human_answer_evaluation"
)
GENERATOR_RESULTS_DIR = Path(
    "reports/generator_model_comparison/SeaLLMs-v3-7B-Chat-Q4_K_M"
)

# All five rated metrics (script 13 collects these on a 1-5 chip scale).
METRICS = [
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
    "completeness",
]

METRIC_LABELS = {
    "context_relevance":  "Context Relevance",
    "faithfulness":       "Faithfulness",
    "answer_relevance":   "Answer Relevance",
    "answer_correctness": "Answer Correctness",
    "completeness":       "Completeness",
}

MAX_TOTAL = len(METRICS) * 5   # 25 — hallucination is a separate Yes/No flag

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
            "Run scripts/13_human_answer_evaluation.py first and collect scores."
        )
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_responses():
    """Every submitted response joined with its evaluator name."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            e.name             AS evaluator_name,
            r.evaluator_id,
            r.question_idx,
            r.context_relevance,
            r.faithfulness,
            r.answer_relevance,
            r.answer_correctness,
            r.completeness,
            r.hallucination,
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
        "SELECT evaluator_id, question_idx FROM assignments "
        "ORDER BY evaluator_id, position"
    ).fetchall()
    conn.close()
    result = {}
    for r in rows:
        result.setdefault(r["evaluator_id"], []).append(r["question_idx"])
    return result


# ----------------------------------------------------------------------------
# Load automated proxy scores + questions from script 10 JSONL
# ----------------------------------------------------------------------------

def find_jsonl():
    path = GENERATOR_RESULTS_DIR / "generated_answer_results.jsonl"
    if not path.exists():
        return None, None
    return path, "SeaLLM v3 7B"


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_proxy_scores():
    """Returns {question_text: {"scores": {...}, "combined": ...}} from script 10."""
    path, label = find_jsonl()
    if not path:
        return {}, None
    proxy = {}
    for entry in _iter_jsonl(path):
        q = entry.get("question", "")
        if q and entry.get("scores"):
            proxy[q] = {
                "scores":   entry["scores"],
                "combined": entry.get("combined"),
            }
    return proxy, label


def load_questions_from_jsonl():
    """Ordered list of question dicts from script 10 JSONL (indexed by position)."""
    path, _ = find_jsonl()
    if not path:
        return []
    return list(_iter_jsonl(path))


# ----------------------------------------------------------------------------
# Core calculations
# ----------------------------------------------------------------------------

def _avg(vals, ndigits=4):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), ndigits) if vals else None


def per_question_stats(responses, questions):
    """
    Returns dict keyed by question_idx with avg metric scores, total avg,
    hallucination tally, and the list of evaluators who rated it.
    """
    by_q = defaultdict(list)
    for r in responses:
        by_q[r["question_idx"]].append(r)

    stats = {}
    for qidx, rows in by_q.items():
        q_text   = questions[qidx]["question"] if qidx < len(questions) else f"Q{qidx}"
        category = (
            questions[qidx].get("category", "unknown")
            if qidx < len(questions) else "unknown"
        )

        metric_avgs = {m: _avg([r[m] for r in rows]) for m in METRICS}
        total_avg   = _avg([r["total_score"] for r in rows])

        halluc_flags = [r["hallucination"] for r in rows if r["hallucination"] is not None]
        halluc_count = sum(1 for v in halluc_flags if v == 1)
        halluc_rate  = round(halluc_count / len(halluc_flags), 4) if halluc_flags else None

        stats[qidx] = {
            "question":            q_text,
            "category":            category,
            "evaluator_count":     len(rows),
            "evaluators":          [r["evaluator_name"] for r in rows],
            "metric_avgs":         metric_avgs,
            "total_avg":           total_avg,
            "hallucination_flags": halluc_count,
            "hallucination_rate":  halluc_rate,
        }

    return stats


def macro_avg_per_metric(q_stats):
    """Macro-avg of each metric across all rated questions."""
    return {
        m: _avg([s["metric_avgs"].get(m) for s in q_stats.values()])
        for m in METRICS
    }


def per_evaluator_stats(responses, assignments):
    """Per-evaluator coverage, avg metric scores, and hallucination flags raised."""
    by_eval = {}
    for r in responses:
        eid = r["evaluator_id"]
        by_eval.setdefault(eid, {"name": r["evaluator_name"], "responses": []})
        by_eval[eid]["responses"].append(r)

    stats = []
    for eid, data in by_eval.items():
        rows      = data["responses"]
        assigned  = len(assignments.get(eid, [])) or len(rows)
        completed = len(rows)

        metric_avgs  = {m: _avg([r[m] for r in rows], 2) for m in METRICS}
        total_avg    = _avg([r["total_score"] for r in rows], 2)
        halluc_flags = sum(1 for r in rows if r["hallucination"] == 1)

        stats.append({
            "evaluator_id":         eid,
            "name":                 data["name"],
            "assigned":             assigned,
            "completed":            completed,
            "coverage":             f"{completed}/{assigned}",
            "metric_avgs":          metric_avgs,
            "total_avg":            total_avg,
            "hallucination_flags":  halluc_flags,
        })

    return sorted(stats, key=lambda s: s["name"])


def inter_annotator_from_responses(responses):
    """
    Avg pairwise absolute score difference per metric on questions rated by
    more than one evaluator (lower = higher agreement). Also returns the
    hallucination flag disagreement rate (fraction of pairs that disagree).
    """
    by_q = defaultdict(list)
    for r in responses:
        by_q[r["question_idx"]].append(r)

    diffs = {m: [] for m in METRICS}
    halluc_pairs, halluc_disagree = 0, 0
    multi_count = 0

    for rows in by_q.values():
        if len(rows) < 2:
            continue
        multi_count += 1
        for r1, r2 in combinations(rows, 2):
            for m in METRICS:
                v1, v2 = r1.get(m), r2.get(m)
                if v1 is not None and v2 is not None:
                    diffs[m].append(abs(v1 - v2))
            h1, h2 = r1.get("hallucination"), r2.get("hallucination")
            if h1 is not None and h2 is not None:
                halluc_pairs += 1
                halluc_disagree += int(h1 != h2)

    result = {m: _avg(diffs[m]) for m in METRICS}
    halluc_disagree_rate = (
        round(halluc_disagree / halluc_pairs, 4) if halluc_pairs else None
    )
    return result, multi_count, halluc_disagree_rate


def human_vs_proxy(q_stats, questions, proxy_scores):
    """Macro-avg human vs automated proxy score per metric (questions with both)."""
    by_metric = {m: {"human": [], "proxy": []} for m in METRICS}

    for qidx, stat in q_stats.items():
        if qidx >= len(questions):
            continue
        p_entry = proxy_scores.get(questions[qidx].get("question", ""))
        if not p_entry:
            continue
        for m in METRICS:
            h_val = stat["metric_avgs"].get(m)
            p_val = p_entry["scores"].get(m)
            if h_val is None or p_val is None:
                continue
            by_metric[m]["human"].append(h_val)
            by_metric[m]["proxy"].append(p_val)

    macro = {}
    for m in METRICS:
        h_vals, p_vals = by_metric[m]["human"], by_metric[m]["proxy"]
        if h_vals and p_vals:
            h_avg = sum(h_vals) / len(h_vals)
            p_avg = sum(p_vals) / len(p_vals)
            macro[m] = {
                "human_avg": round(h_avg, 4),
                "proxy_avg": round(p_avg, 4),
                "diff":      round(h_avg - p_avg, 4),
                "n":         len(h_vals),
            }
    return macro


def hallucination_report_data(q_stats, questions, responses):
    """
    Builds the standalone hallucination report from the REAL human Yes/No flag.

    severity:
      "confirmed" -> strict majority of evaluators flagged hallucination (rate > 0.5)
      "disputed"  -> at least one flagged it, but not a strict majority (incl. ties)
    """
    # Which evaluators actually raised the hallucination flag, per question.
    flaggers = defaultdict(list)
    for r in responses:
        if r["hallucination"] == 1:
            flaggers[r["question_idx"]].append(r["evaluator_name"])

    flagged = []
    by_category = defaultdict(lambda: {"answers": 0, "halluc_flag_answers": 0})
    total_answers, total_flag_answers = 0, 0

    for qidx, stat in q_stats.items():
        rate = stat.get("hallucination_rate")
        if rate is None:
            continue
        total_answers += 1
        by_category[stat["category"]]["answers"] += 1
        if stat["hallucination_flags"] > 0:
            total_flag_answers += 1
            by_category[stat["category"]]["halluc_flag_answers"] += 1
            expected = generated = ""
            if qidx < len(questions):
                expected  = questions[qidx].get("expected_answer", "") or ""
                generated = questions[qidx].get("generated_answer", "") or ""
            flagged.append({
                "question_idx":        qidx,
                "question":            stat["question"],
                "category":            stat["category"],
                "hallucination_flags": stat["hallucination_flags"],
                "evaluator_count":     stat["evaluator_count"],
                "hallucination_rate":  rate,
                "severity":            "confirmed" if rate > 0.5 else "disputed",
                "faithfulness_avg":    stat["metric_avgs"].get("faithfulness"),
                "total_avg":           stat["total_avg"],
                "evaluators":          stat["evaluators"],
                "flagged_by":          flaggers.get(qidx, []),
                "expected_answer":     expected,
                "generated_answer":    generated,
            })

    flagged.sort(key=lambda x: (-x["hallucination_rate"], x["question_idx"]))

    category_breakdown = {
        cat: {
            "answers":             d["answers"],
            "halluc_flag_answers": d["halluc_flag_answers"],
            "rate": round(d["halluc_flag_answers"] / d["answers"], 4) if d["answers"] else None,
        }
        for cat, d in sorted(by_category.items())
    }

    return {
        "total_answers_rated":   total_answers,
        "answers_with_flag":     total_flag_answers,
        "confirmed":             sum(1 for f in flagged if f["severity"] == "confirmed"),
        "disputed":              sum(1 for f in flagged if f["severity"] == "disputed"),
        "overall_rate":          round(total_flag_answers / total_answers, 4) if total_answers else None,
        "category_breakdown":    category_breakdown,
        "flagged_answers":       flagged,
    }


# ----------------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------------

def print_report(ctx):
    banner = SEPARATOR
    macro_avgs = ctx["macro_avgs"]

    print(f"\n{banner}")
    print(f"{BOLD}RAG HUMAN EVALUATION REPORT{RESET}")
    print(banner)
    print(tabulate(
        [
            ["Total evaluators",            len(ctx["eval_stats"])],
            ["Total responses submitted",   ctx["total_responses"]],
            ["Questions rated (>=1 eval)",   ctx["total_questions_rated"]],
            ["Questions rated by >1 eval",  ctx["iaa_multi_count"]],
            ["Proxy model (script 10)",     ctx["proxy_label"] or "not found"],
        ],
        tablefmt="grid",
    ))

    print(f"\n{banner}")
    print(f"{BOLD}MACRO-AVG HUMAN SCORES (across all rated questions){RESET}")
    print(banner)
    macro_rows, combined = [], []
    for m in METRICS:
        v = macro_avgs.get(m)
        macro_rows.append([METRIC_LABELS[m], f"{v:.4f}" if v is not None else "N/A"])
        if v is not None:
            combined.append(v)
    combined_avg = sum(combined) / len(combined) if combined else 0
    macro_rows.append(["-- Combined avg --", f"{combined_avg:.4f}"])
    print(tabulate(macro_rows, headers=["Metric", "Avg (1-5)"], tablefmt="grid"))

    print(f"\n{banner}")
    print(f"{BOLD}INTER-ANNOTATOR AGREEMENT (avg pairwise |diff|, lower = better){RESET}")
    print(f"{BOLD}Based on {ctx['iaa_multi_count']} questions rated by more than one evaluator{RESET}")
    print(banner)
    iaa_rows = []
    for m in METRICS:
        v = ctx["iaa"].get(m)
        if v is None:
            iaa_rows.append([METRIC_LABELS[m], "N/A", "—"])
        else:
            agreement = (f"{GREEN}Strong{RESET}" if v <= 0.5
                         else f"{YELLOW}Moderate{RESET}" if v <= 1.0
                         else f"{RED}Weak{RESET}")
            iaa_rows.append([METRIC_LABELS[m], f"{v:.4f}", agreement])
    hdr = ctx["halluc_disagree_rate"]
    iaa_rows.append([
        "Hallucination (flag)",
        f"{hdr:.4f}" if hdr is not None else "N/A",
        "disagreement rate",
    ])
    print(tabulate(iaa_rows, headers=["Metric", "Avg |diff|", "Agreement"], tablefmt="grid"))

    print(f"\n{banner}")
    print(f"{BOLD}PER-EVALUATOR STATS{RESET}")
    print(banner)
    eval_rows = []
    for s in ctx["eval_stats"]:
        metric_cols = [
            f"{s['metric_avgs'].get(m):.2f}" if s["metric_avgs"].get(m) is not None else "N/A"
            for m in METRICS
        ]
        eval_rows.append([
            s["name"], s["coverage"], *metric_cols,
            f"{s['total_avg']:.2f}" if s["total_avg"] is not None else "N/A",
            s["hallucination_flags"],
        ])
    print(tabulate(
        eval_rows,
        headers=["Evaluator", "Coverage"] + [METRIC_LABELS[m][:9] for m in METRICS]
                + [f"Total /{MAX_TOTAL}", "Halluc"],
        tablefmt="grid",
    ))

    if ctx["human_proxy_macro"]:
        print(f"\n{banner}")
        print(f"{BOLD}HUMAN vs AUTOMATED PROXY SCORES (macro-avg){RESET}")
        print(f"{BOLD}Diff = Human - Proxy (positive = humans rated higher){RESET}")
        print(banner)
        hp_rows = []
        for m in METRICS:
            v = ctx["human_proxy_macro"].get(m)
            if not v:
                hp_rows.append([METRIC_LABELS[m], "N/A", "N/A", "N/A"])
                continue
            diff_color = (GREEN if abs(v["diff"]) <= 0.3
                          else YELLOW if abs(v["diff"]) <= 0.7 else RED)
            hp_rows.append([
                METRIC_LABELS[m], f"{v['human_avg']:.4f}", f"{v['proxy_avg']:.4f}",
                f"{diff_color}{v['diff']:+.4f}{RESET}",
            ])
        print(tabulate(hp_rows, headers=["Metric", "Human Avg", "Proxy Avg", "Diff"],
                       tablefmt="grid"))

    # --- Hallucination summary (full detail goes to the separate report) ---
    hr = ctx["halluc"]
    print(f"\n{banner}")
    if hr["answers_with_flag"]:
        print(f"{BOLD}{RED}HALLUCINATION REPORT -- {hr['answers_with_flag']} answer(s) flagged by humans"
              f" ({hr['confirmed']} confirmed, {hr['disputed']} disputed){RESET}")
        print(f"{BOLD}Overall hallucination rate: {hr['overall_rate']:.2%}{RESET}")
        print(banner)
        for f in hr["flagged_answers"]:
            tag = f"{RED}CONFIRMED{RESET}" if f["severity"] == "confirmed" else f"{YELLOW}disputed{RESET}"
            faith = f"{f['faithfulness_avg']:.2f}" if f["faithfulness_avg"] is not None else "N/A"
            print(f"\n  [{tag}] Q{f['question_idx']} [{f['category']}] "
                  f"flags={f['hallucination_flags']}/{f['evaluator_count']} | Faith={faith}")
            print(f"  {f['question'][:100]}{'…' if len(f['question']) > 100 else ''}")
    else:
        print(f"{BOLD}{GREEN}HALLUCINATION REPORT -- No answers flagged by any evaluator{RESET}")
        print(banner)

    print(f"\n{banner}")
    print(f"{BOLD}PER-QUESTION HUMAN SCORES (avg across evaluators){RESET}")
    print(banner)
    pq_rows = []
    for qidx in sorted(ctx["q_stats"].keys()):
        stat = ctx["q_stats"][qidx]
        metric_cols = [
            f"{stat['metric_avgs'].get(m):.2f}" if stat["metric_avgs"].get(m) is not None else "N/A"
            for m in METRICS
        ]
        flag = f"{RED}H{RESET}" if stat["hallucination_flags"] else ""
        pq_rows.append([
            f"Q{qidx}", stat["category"][:8], stat["evaluator_count"], *metric_cols,
            f"{stat['total_avg']:.2f}" if stat["total_avg"] is not None else "N/A", flag,
        ])
    print(tabulate(
        pq_rows,
        headers=["Q", "Cat", "Evals"] + [METRIC_LABELS[m][:6] for m in METRICS]
                + ["Total", "Halluc"],
        tablefmt="grid",
    ))


# ----------------------------------------------------------------------------
# Write the unified report (JSON + Markdown)
# ----------------------------------------------------------------------------

def write_unified_report(ctx):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    macro_avgs = ctx["macro_avgs"]

    combined_vals = [v for v in macro_avgs.values() if v is not None]
    combined_avg  = round(sum(combined_vals) / len(combined_vals), 4) if combined_vals else 0.0

    payload = {
        "generated_at":              int(time.time()),
        "proxy_model":               ctx["proxy_label"],
        "max_total_score":           MAX_TOTAL,
        "metrics":                   METRICS,
        "total_evaluators":          len(ctx["eval_stats"]),
        "total_responses":           ctx["total_responses"],
        "questions_rated":           ctx["total_questions_rated"],
        "multi_rated_questions":     ctx["iaa_multi_count"],
        "macro_avgs":                macro_avgs,
        "combined_avg":              combined_avg,
        "inter_annotator_agreement": ctx["iaa"],
        "hallucination_flag_disagreement_rate": ctx["halluc_disagree_rate"],
        "per_evaluator":             ctx["eval_stats"],
        "human_vs_proxy":            ctx["human_proxy_macro"],
        "hallucination_summary": {
            "answers_with_flag": ctx["halluc"]["answers_with_flag"],
            "confirmed":         ctx["halluc"]["confirmed"],
            "disputed":          ctx["halluc"]["disputed"],
            "overall_rate":      ctx["halluc"]["overall_rate"],
        },
        "per_question": {str(qidx): stat for qidx, stat in ctx["q_stats"].items()},
    }

    json_path = REPORTS_DIR / "human_answer_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = [
        "# RAG Human Evaluation Report\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Proxy model (script 10):** {ctx['proxy_label'] or 'not found'}  \n",
        f"**Evaluators:** {len(ctx['eval_stats'])}  \n",
        f"**Responses:** {ctx['total_responses']}  \n",
        f"**Questions rated:** {ctx['total_questions_rated']} "
        f"(multi-rated: {ctx['iaa_multi_count']})  \n",
        f"**Max total score:** {MAX_TOTAL} (5 metrics x 5)  \n\n",
    ]

    md.append("## Macro-Avg Human Scores\n\n")
    rows = [[METRIC_LABELS[m], f"{macro_avgs.get(m):.4f}" if macro_avgs.get(m) is not None else "N/A"]
            for m in METRICS]
    rows.append(["**Combined**", f"**{combined_avg:.4f}**"])
    md.append(tabulate(rows, headers=["Metric", "Avg (1-5)"], tablefmt="github") + "\n\n")

    md.append("## Inter-Annotator Agreement\n\n")
    md.append("Avg pairwise absolute difference per metric (lower = stronger agreement).\n\n")
    iaa_rows = [[METRIC_LABELS[m], f"{ctx['iaa'].get(m):.4f}" if ctx["iaa"].get(m) is not None else "N/A"]
                for m in METRICS]
    hdr = ctx["halluc_disagree_rate"]
    iaa_rows.append(["Hallucination flag (disagreement rate)",
                     f"{hdr:.4f}" if hdr is not None else "N/A"])
    md.append(tabulate(iaa_rows, headers=["Metric", "Value"], tablefmt="github") + "\n\n")

    md.append("## Per-Evaluator Stats\n\n")
    eval_rows = []
    for s in ctx["eval_stats"]:
        eval_rows.append([
            s["name"], s["coverage"],
            *[f"{s['metric_avgs'].get(m):.2f}" if s["metric_avgs"].get(m) is not None else "N/A"
              for m in METRICS],
            f"{s['total_avg']:.2f}" if s["total_avg"] is not None else "N/A",
            s["hallucination_flags"],
        ])
    md.append(tabulate(
        eval_rows,
        headers=["Evaluator", "Coverage"] + [METRIC_LABELS[m] for m in METRICS]
                + [f"Total /{MAX_TOTAL}", "Halluc flags"],
        tablefmt="github",
    ) + "\n\n")

    if ctx["human_proxy_macro"]:
        md.append("## Human vs Automated Proxy Scores\n\n")
        md.append("Diff = Human − Proxy (positive = humans rated higher than the script 10 judge).\n\n")
        hp_rows = []
        for m in METRICS:
            v = ctx["human_proxy_macro"].get(m)
            if v:
                hp_rows.append([METRIC_LABELS[m], f"{v['human_avg']:.4f}",
                                f"{v['proxy_avg']:.4f}", f"{v['diff']:+.4f}"])
        md.append(tabulate(hp_rows, headers=["Metric", "Human", "Proxy", "Diff"],
                           tablefmt="github") + "\n\n")

    hr = ctx["halluc"]
    md.append("## Hallucination Summary\n\n")
    if hr["answers_with_flag"]:
        md.append(f"- **{hr['answers_with_flag']}** answer(s) flagged by at least one evaluator "
                  f"({hr['confirmed']} confirmed, {hr['disputed']} disputed)\n")
        md.append(f"- **Overall hallucination rate:** {hr['overall_rate']:.2%}\n")
    else:
        md.append("- No answers were flagged as hallucinated by any evaluator.\n")
    md.append("\nSee **[hallucination_report.md](hallucination_report.md)** for the full breakdown.\n\n")

    md.append("## Per-Question Human Scores\n\n")
    pq_rows = []
    for qidx in sorted(ctx["q_stats"].keys()):
        stat = ctx["q_stats"][qidx]
        pq_rows.append([
            f"Q{qidx}", stat["category"], stat["evaluator_count"],
            *[f"{stat['metric_avgs'].get(m):.2f}" if stat["metric_avgs"].get(m) is not None else "N/A"
              for m in METRICS],
            f"{stat['total_avg']:.2f}" if stat["total_avg"] is not None else "N/A",
            "yes" if stat["hallucination_flags"] else "",
        ])
    md.append(tabulate(
        pq_rows,
        headers=["Q", "Category", "Evals"] + [METRIC_LABELS[m] for m in METRICS]
                + ["Total", "Halluc"],
        tablefmt="github",
    ) + "\n")

    md_path = REPORTS_DIR / "human_answer_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    return json_path, md_path


# ----------------------------------------------------------------------------
# Write the SEPARATE hallucination report (JSON + Markdown)
# ----------------------------------------------------------------------------

def write_hallucination_report(hr, proxy_label):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at":        int(time.time()),
        "proxy_model":         proxy_label,
        "source":              "human evaluator hallucination flag (script 13)",
        "total_answers_rated": hr["total_answers_rated"],
        "answers_with_flag":   hr["answers_with_flag"],
        "confirmed":           hr["confirmed"],
        "disputed":            hr["disputed"],
        "overall_rate":        hr["overall_rate"],
        "category_breakdown":  hr["category_breakdown"],
        "flagged_answers":     hr["flagged_answers"],
    }
    json_path = REPORTS_DIR / "hallucination_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = [
        "# Hallucination Report\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "Driven by the **human evaluator hallucination flag** collected in script 13 "
        "(a direct Yes/No judgement, not a proxy off another metric).\n\n",
        "- **confirmed** — a majority of the evaluators who rated the answer flagged it\n",
        "- **disputed** — at least one, but not a majority, flagged it\n\n",
    ]

    md.append("## Summary\n\n")
    summary_rows = [
        ["Answers rated",          hr["total_answers_rated"]],
        ["Answers flagged (>=1)",   hr["answers_with_flag"]],
        ["Confirmed (majority)",   hr["confirmed"]],
        ["Disputed (minority)",    hr["disputed"]],
        ["Overall rate",           f"{hr['overall_rate']:.2%}" if hr["overall_rate"] is not None else "N/A"],
    ]
    md.append(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="github") + "\n\n")

    md.append("## By Category\n\n")
    cat_rows = [
        [cat, d["answers"], d["halluc_flag_answers"],
         f"{d['rate']:.2%}" if d["rate"] is not None else "N/A"]
        for cat, d in hr["category_breakdown"].items()
    ]
    md.append(tabulate(cat_rows, headers=["Category", "Answers", "Flagged", "Rate"],
                       tablefmt="github") + "\n\n")

    md.append("## Flagged Answers\n\n")
    if not hr["flagged_answers"]:
        md.append("No answers were flagged as hallucinated by any evaluator. ✅\n")
    else:
        for f in hr["flagged_answers"]:
            faith = f"{f['faithfulness_avg']:.2f}" if f["faithfulness_avg"] is not None else "N/A"
            total = f"{f['total_avg']:.2f}" if f["total_avg"] is not None else "N/A"
            md.append(
                f"### Q{f['question_idx']} — {f['severity'].upper()} "
                f"({f['hallucination_flags']}/{f['evaluator_count']} flagged)\n\n"
            )
            md.append(f"- **Category:** {f['category']}\n")
            md.append(f"- **Hallucination rate:** {f['hallucination_rate']:.2%}\n")
            md.append(f"- **Avg faithfulness:** {faith} | **Avg total:** {total} / {MAX_TOTAL}\n")
            md.append(f"- **Flagged by:** {', '.join(f['flagged_by'])} "
                      f"(rated by: {', '.join(f['evaluators'])})\n\n")
            md.append(f"**Question**\n\n> {f['question']}\n\n")
            if f["expected_answer"]:
                md.append(f"**Expected answer**\n\n> {f['expected_answer']}\n\n")
            if f["generated_answer"]:
                md.append(f"**Generated answer**\n\n> {f['generated_answer']}\n\n")
            md.append("---\n\n")

    md_path = REPORTS_DIR / "hallucination_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    return json_path, md_path


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
        print("Run scripts/13_human_answer_evaluation.py and collect scores first.")
        return

    print(f"  {len(responses)} responses from {len(evaluators)} evaluators.")

    print("Loading questions + proxy scores from script 10 JSONL...")
    questions                 = load_questions_from_jsonl()
    proxy_scores, proxy_label = load_proxy_scores()
    if not questions:
        print("  Warning: no JSONL found -- question text will show as Q<idx>.")
    else:
        print(f"  {len(questions)} questions loaded from {proxy_label or 'unknown model'}.")

    print("Computing stats...")
    q_stats    = per_question_stats(responses, questions)
    macro_avgs = macro_avg_per_metric(q_stats)
    eval_stats = per_evaluator_stats(responses, assignments)
    iaa, iaa_multi, halluc_disagree_rate = inter_annotator_from_responses(responses)
    human_proxy_macro = human_vs_proxy(q_stats, questions, proxy_scores)
    halluc = hallucination_report_data(q_stats, questions, responses)

    ctx = {
        "q_stats":               q_stats,
        "macro_avgs":            macro_avgs,
        "eval_stats":            eval_stats,
        "iaa":                   iaa,
        "iaa_multi_count":       iaa_multi,
        "halluc_disagree_rate":  halluc_disagree_rate,
        "human_proxy_macro":     human_proxy_macro,
        "halluc":                halluc,
        "proxy_label":           proxy_label,
        "total_responses":       len(responses),
        "total_questions_rated": len(q_stats),
    }

    print_report(ctx)

    json_path, md_path = write_unified_report(ctx)
    h_json, h_md       = write_hallucination_report(halluc, proxy_label)

    print(f"\nSaved unified JSON report      : {json_path}")
    print(f"Saved unified Markdown report  : {md_path}")
    print(f"Saved hallucination JSON       : {h_json}")
    print(f"Saved hallucination Markdown   : {h_md}")
    print(f"\nTotal runtime : {time.time() - start:.2f} seconds")
    print("Human evaluation report completed successfully.")


if __name__ == "__main__":
    main()
