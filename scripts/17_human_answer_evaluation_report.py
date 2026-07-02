import json
import re
import sqlite3
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from tabulate import tabulate

# ----------------------------------------------------------------------------
# RAG Human Evaluation Report
# ----------------------------------------------------------------------------
# PURPOSE: read the human scores collected by scripts/16 (the Streamlit eval UI)
# from SQLite and merge them into ONE unified human-verified RAGAS report, plus
# a completely SEPARATE hallucination report.
#
# Hallucination is treated INDEPENDENTLY. It is a direct human Yes/No flag, not
# one of the five RAGAS metrics. It is never averaged into the RAGAS macro
# average, the combined score, or the per-evaluator metric averages — it gets
# its own standalone report (hallucination_report.{json,md}). The unified RAGAS
# report only carries a short pointer to it.
#
# Schema this matches (scripts/16, current):
#   evaluators(id, name, session_id, eval_mode, created_at)
#   responses (id, evaluator_id, question_id, context_relevance, faithfulness,
#              answer_relevance, answer_correctness, completeness, hallucination,
#              total_score, comment, submitted_at)
#   -- responses are keyed by (evaluator_id, question_id); question_id is the
#      global TEXT id (e.g. "guidance_qid_001"), NOT a positional index.
#   -- the assignments table is unused by scripts/16, so coverage is derived
#      from each evaluator's eval_mode instead.
#
# Inputs
#   reports/human_answer_evaluation/human_answer_evaluation.db          (scripts/16)
#   reports/generator_model_comparison/<winner>/generated_answer_results_full.jsonl
#       -- the SAME 400-question file scripts/16 serves answers from (scripts/13),
#          which also carries the automated proxy scores for the human-vs-proxy
#          comparison.
#
# Outputs (all under reports/human_answer_evaluation/)
#   human_answer_evaluation_report.json / .md   unified RAGAS report
#   hallucination_report.json / .md             standalone hallucination report
# ----------------------------------------------------------------------------

DB_PATH = Path("reports/human_answer_evaluation/human_answer_evaluation.db")
REPORTS_DIR = Path("reports/human_answer_evaluation")

# Must match scripts/16 JSONL_PATH exactly so question_ids line up.
JSONL_PATH = Path(
    "reports/generator_model_comparison/"
    "SeaLLMs-v3-7B-Chat-Q4_K_M/"
    "generated_answer_results_full.jsonl"
)
PROXY_LABEL = "SeaLLM v3 7B"

# question_id format: "<category>_qid_<NNN>"; ids 001-025 are human-authored.
HUMAN_QID_MAX = 25

# The five RAGAS-style metrics (scripts/16 collects these on a 1-5 chip scale).
# Hallucination is intentionally NOT in this list — it is a separate Yes/No flag.
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

# Mode -> human-readable label, used for per-evaluator coverage.
MODE_LABELS = {"human": "Human-Created", "ai": "AI-Created", "all": "All Questions"}

MAX_TOTAL = len(METRICS) * 5   # 25 — hallucination is a separate Yes/No flag

BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH


# ----------------------------------------------------------------------------
# Question-id helpers
# ----------------------------------------------------------------------------

_QID_RE = re.compile(r"^(?P<cat>.+)_qid_(?P<num>\d+)$")


def parse_qid(qid):
    """Returns (category, source) from a question_id. source is 'human' for
    ids 001-025 in each category, 'ai' above that."""
    m = _QID_RE.match(qid or "")
    if not m:
        return "unknown", "unknown"
    num = int(m.group("num"))
    return m.group("cat"), ("human" if num <= HUMAN_QID_MAX else "ai")


def qid_sort_key(qid):
    """Stable sort: by category, then numeric suffix (so guidance_qid_002 sorts
    before guidance_qid_010), falling back to the raw string."""
    m = _QID_RE.match(qid or "")
    if not m:
        return (qid or "", -1)
    return (m.group("cat"), int(m.group("num")))


# ----------------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------------

def get_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}.\n"
            "Run scripts/16_human_answer_evaluation.py first and collect scores."
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
            r.question_id,
            r.context_relevance,
            r.faithfulness,
            r.answer_relevance,
            r.answer_correctness,
            r.completeness,
            r.hallucination,
            r.total_score,
            r.comment,
            r.submitted_at
        FROM responses r
        JOIN evaluators e ON e.id = r.evaluator_id
        ORDER BY r.question_id, e.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_evaluators():
    """Returns {evaluator_id: {name, eval_mode}}."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, eval_mode, created_at FROM evaluators ORDER BY name"
    ).fetchall()
    conn.close()
    return {r["id"]: {"name": r["name"], "eval_mode": r["eval_mode"]} for r in rows}


# ----------------------------------------------------------------------------
# Load questions + automated proxy scores from the scripts/13 JSONL
# ----------------------------------------------------------------------------

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


def load_questions():
    """Returns (by_id, order, mode_sizes).

    by_id      : {question_id: entry} where entry carries question text,
                 expected/generated answers, derived category/source, and the
                 automated proxy `scores`/`combined` from scripts/13.
    order      : question_ids in file order.
    mode_sizes : {'human': n, 'ai': n, 'all': n} question counts per mode.
    """
    by_id, order = {}, []
    if not JSONL_PATH.exists():
        return by_id, order, {"human": 0, "ai": 0, "all": 0}
    for entry in _iter_jsonl(JSONL_PATH):
        qid = entry.get("question_id")
        if not qid:
            continue
        cat, source = parse_qid(qid)
        entry["category"] = entry.get("category") or cat
        entry["source"] = source
        by_id[qid] = entry
        order.append(qid)
    mode_sizes = {
        "human": sum(1 for q in order if by_id[q]["source"] == "human"),
        "ai":    sum(1 for q in order if by_id[q]["source"] == "ai"),
        "all":   len(order),
    }
    return by_id, order, mode_sizes


# ----------------------------------------------------------------------------
# Core calculations (RAGAS — hallucination handled separately below)
# ----------------------------------------------------------------------------

def _avg(vals, ndigits=4):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), ndigits) if vals else None


def per_question_stats(responses, by_id):
    """Dict keyed by question_id: avg metric scores, total avg, the evaluators
    who rated it, their comments, and the hallucination tally (kept here only so
    the standalone hallucination report can be built from the same pass — it is
    never folded into the RAGAS metric averages)."""
    grouped = defaultdict(list)
    for r in responses:
        grouped[r["question_id"]].append(r)

    stats = {}
    for qid, rows in grouped.items():
        entry = by_id.get(qid, {})
        metric_avgs = {m: _avg([r[m] for r in rows]) for m in METRICS}
        total_avg   = _avg([r["total_score"] for r in rows])

        halluc_flags = [r["hallucination"] for r in rows if r["hallucination"] is not None]
        halluc_count = sum(1 for v in halluc_flags if v == 1)
        halluc_rate  = round(halluc_count / len(halluc_flags), 4) if halluc_flags else None

        comments = [
            {"evaluator": r["evaluator_name"], "comment": (r.get("comment") or "").strip()}
            for r in rows
            if (r.get("comment") or "").strip()
        ]

        stats[qid] = {
            "question_id":         qid,
            "question":            entry.get("question", qid),
            "category":            entry.get("category", "unknown"),
            "evaluator_count":     len(rows),
            "evaluators":          [r["evaluator_name"] for r in rows],
            "metric_avgs":         metric_avgs,
            "total_avg":           total_avg,
            "comments":            comments,
            "hallucination_flags": halluc_count,
            "hallucination_rate":  halluc_rate,
        }
    return stats


def macro_avg_per_metric(q_stats):
    """Macro-avg of each RAGAS metric across all rated questions."""
    return {
        m: _avg([s["metric_avgs"].get(m) for s in q_stats.values()])
        for m in METRICS
    }


def per_evaluator_stats(responses, evaluators, mode_sizes):
    """Per-evaluator coverage (completed / questions in their chosen mode), avg
    RAGAS metric scores, and the count of hallucination flags they raised."""
    by_eval = {}
    for r in responses:
        eid = r["evaluator_id"]
        by_eval.setdefault(eid, []).append(r)

    stats = []
    for eid, rows in by_eval.items():
        meta      = evaluators.get(eid, {})
        name      = meta.get("name") or (rows[0]["evaluator_name"] if rows else f"#{eid}")
        mode      = meta.get("eval_mode") or "all"
        assigned  = mode_sizes.get(mode) or len(rows)
        completed = len(rows)

        metric_avgs  = {m: _avg([r[m] for r in rows], 2) for m in METRICS}
        total_avg    = _avg([r["total_score"] for r in rows], 2)
        halluc_flags = sum(1 for r in rows if r["hallucination"] == 1)

        stats.append({
            "evaluator_id":        eid,
            "name":                name,
            "mode":                mode,
            "mode_label":          MODE_LABELS.get(mode, mode),
            "assigned":            assigned,
            "completed":           completed,
            "coverage":            f"{completed}/{assigned}",
            "metric_avgs":         metric_avgs,
            "total_avg":           total_avg,
            "hallucination_flags": halluc_flags,
        })
    return sorted(stats, key=lambda s: s["name"].lower())


def inter_annotator_from_responses(responses):
    """Avg pairwise absolute score difference per RAGAS metric on questions
    rated by >1 evaluator (lower = higher agreement), plus the hallucination
    flag disagreement rate (fraction of evaluator pairs that disagree)."""
    grouped = defaultdict(list)
    for r in responses:
        grouped[r["question_id"]].append(r)

    diffs = {m: [] for m in METRICS}
    halluc_pairs, halluc_disagree = 0, 0
    multi_count = 0

    for rows in grouped.values():
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


def human_vs_proxy(q_stats, by_id):
    """Macro-avg human vs automated proxy score per RAGAS metric, over the
    questions that have both a human rating and a proxy score. The proxy scores
    come from the same scripts/13 JSONL (note: the proxy's answer_relevance is
    computed but UNWEIGHTED in stage 1 — it is still shown here for comparison)."""
    by_metric = {m: {"human": [], "proxy": []} for m in METRICS}

    for qid, stat in q_stats.items():
        proxy_scores = (by_id.get(qid) or {}).get("scores")
        if not proxy_scores:
            continue
        for m in METRICS:
            h_val = stat["metric_avgs"].get(m)
            p_val = proxy_scores.get(m)
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


# ----------------------------------------------------------------------------
# SEPARATE hallucination analysis (independent of the RAGAS metrics)
# ----------------------------------------------------------------------------

def hallucination_report_data(q_stats, by_id, responses):
    """Standalone hallucination report built purely from the human Yes/No flag.

    severity:
      "confirmed" -> strict majority of evaluators flagged it (rate > 0.5)
      "disputed"  -> at least one flagged it, but not a strict majority
    """
    # Evaluators who raised the flag (and their comment, if any), per question.
    flaggers = defaultdict(list)
    for r in responses:
        if r["hallucination"] == 1:
            flaggers[r["question_id"]].append({
                "evaluator": r["evaluator_name"],
                "comment":   (r.get("comment") or "").strip(),
            })

    flagged = []
    by_category = defaultdict(lambda: {"answers": 0, "halluc_flag_answers": 0})
    total_answers, total_flag_answers = 0, 0

    for qid, stat in q_stats.items():
        rate = stat.get("hallucination_rate")
        if rate is None:
            continue
        total_answers += 1
        by_category[stat["category"]]["answers"] += 1
        if stat["hallucination_flags"] > 0:
            total_flag_answers += 1
            by_category[stat["category"]]["halluc_flag_answers"] += 1
            entry = by_id.get(qid, {})
            flagged.append({
                "question_id":         qid,
                "question":            stat["question"],
                "category":            stat["category"],
                "hallucination_flags": stat["hallucination_flags"],
                "evaluator_count":     stat["evaluator_count"],
                "hallucination_rate":  rate,
                "severity":            "confirmed" if rate > 0.5 else "disputed",
                "faithfulness_avg":    stat["metric_avgs"].get("faithfulness"),
                "total_avg":           stat["total_avg"],
                "evaluators":          stat["evaluators"],
                "flagged_by":          flaggers.get(qid, []),
                "expected_answer":     entry.get("expected_answer", "") or "",
                "generated_answer":    entry.get("generated_answer", "") or "",
            })

    flagged.sort(key=lambda x: (-x["hallucination_rate"], qid_sort_key(x["question_id"])))

    category_breakdown = {
        cat: {
            "answers":             d["answers"],
            "halluc_flag_answers": d["halluc_flag_answers"],
            "rate": round(d["halluc_flag_answers"] / d["answers"], 4) if d["answers"] else None,
        }
        for cat, d in sorted(by_category.items())
    }

    return {
        "total_answers_rated": total_answers,
        "answers_with_flag":   total_flag_answers,
        "confirmed":           sum(1 for f in flagged if f["severity"] == "confirmed"),
        "disputed":            sum(1 for f in flagged if f["severity"] == "disputed"),
        "overall_rate":        round(total_flag_answers / total_answers, 4) if total_answers else None,
        "category_breakdown":  category_breakdown,
        "flagged_answers":     flagged,
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
            ["Questions rated (>=1 eval)",  ctx["total_questions_rated"]],
            ["Questions rated by >1 eval",  ctx["iaa_multi_count"]],
            ["Proxy model (scripts/13)",    ctx["proxy_label"] or "not found"],
        ],
        tablefmt="grid",
    ))

    print(f"\n{banner}")
    print(f"{BOLD}MACRO-AVG HUMAN RAGAS SCORES (across all rated questions){RESET}")
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
            s["name"], s["mode_label"], s["coverage"], *metric_cols,
            f"{s['total_avg']:.2f}" if s["total_avg"] is not None else "N/A",
            s["hallucination_flags"],
        ])
    print(tabulate(
        eval_rows,
        headers=["Evaluator", "Mode", "Coverage"] + [METRIC_LABELS[m][:9] for m in METRICS]
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

    # --- Hallucination SUMMARY only (full detail in the separate report) ------
    hr = ctx["halluc"]
    print(f"\n{banner}")
    if hr["answers_with_flag"]:
        print(f"{BOLD}{RED}HALLUCINATION (separate report) -- {hr['answers_with_flag']} answer(s) flagged"
              f" ({hr['confirmed']} confirmed, {hr['disputed']} disputed){RESET}")
        print(f"{BOLD}Overall hallucination rate: {hr['overall_rate']:.2%}{RESET}")
        print(banner)
        for f in hr["flagged_answers"]:
            tag = f"{RED}CONFIRMED{RESET}" if f["severity"] == "confirmed" else f"{YELLOW}disputed{RESET}"
            faith = f"{f['faithfulness_avg']:.2f}" if f["faithfulness_avg"] is not None else "N/A"
            print(f"\n  [{tag}] {f['question_id']} [{f['category']}] "
                  f"flags={f['hallucination_flags']}/{f['evaluator_count']} | Faith={faith}")
            print(f"  {f['question'][:100]}{'…' if len(f['question']) > 100 else ''}")
    else:
        print(f"{BOLD}{GREEN}HALLUCINATION (separate report) -- No answers flagged by any evaluator{RESET}")
        print(banner)

    print(f"\n{banner}")
    print(f"{BOLD}PER-QUESTION HUMAN RAGAS SCORES (avg across evaluators){RESET}")
    print(banner)
    pq_rows = []
    for qid in sorted(ctx["q_stats"].keys(), key=qid_sort_key):
        stat = ctx["q_stats"][qid]
        metric_cols = [
            f"{stat['metric_avgs'].get(m):.2f}" if stat["metric_avgs"].get(m) is not None else "N/A"
            for m in METRICS
        ]
        flag = f"{RED}H{RESET}" if stat["hallucination_flags"] else ""
        pq_rows.append([
            qid, stat["category"][:8], stat["evaluator_count"], *metric_cols,
            f"{stat['total_avg']:.2f}" if stat["total_avg"] is not None else "N/A", flag,
        ])
    print(tabulate(
        pq_rows,
        headers=["Question ID", "Cat", "Evals"] + [METRIC_LABELS[m][:6] for m in METRICS]
                + ["Total", "Halluc"],
        tablefmt="grid",
    ))


# ----------------------------------------------------------------------------
# Write the unified RAGAS report (JSON + Markdown)
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
        # Pointer only -- the full hallucination analysis is its own report.
        "hallucination_summary": {
            "answers_with_flag": ctx["halluc"]["answers_with_flag"],
            "confirmed":         ctx["halluc"]["confirmed"],
            "disputed":          ctx["halluc"]["disputed"],
            "overall_rate":      ctx["halluc"]["overall_rate"],
            "see":               "hallucination_report.json / hallucination_report.md",
        },
        "per_question": {qid: stat for qid, stat in ctx["q_stats"].items()},
    }

    json_path = REPORTS_DIR / "human_answer_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = [
        "# RAG Human Evaluation Report\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Proxy model (scripts/13):** {ctx['proxy_label'] or 'not found'}  \n",
        f"**Evaluators:** {len(ctx['eval_stats'])}  \n",
        f"**Responses:** {ctx['total_responses']}  \n",
        f"**Questions rated:** {ctx['total_questions_rated']} "
        f"(multi-rated: {ctx['iaa_multi_count']})  \n",
        f"**Max total score:** {MAX_TOTAL} (5 RAGAS metrics x 5)  \n\n",
        "> Hallucination is reported **separately** as a human Yes/No flag and is "
        "not part of any RAGAS average below. See "
        "**[hallucination_report.md](hallucination_report.md)**.\n\n",
    ]

    md.append("## Macro-Avg Human RAGAS Scores\n\n")
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
            s["name"], s["mode_label"], s["coverage"],
            *[f"{s['metric_avgs'].get(m):.2f}" if s["metric_avgs"].get(m) is not None else "N/A"
              for m in METRICS],
            f"{s['total_avg']:.2f}" if s["total_avg"] is not None else "N/A",
            s["hallucination_flags"],
        ])
    md.append(tabulate(
        eval_rows,
        headers=["Evaluator", "Mode", "Coverage"] + [METRIC_LABELS[m] for m in METRICS]
                + [f"Total /{MAX_TOTAL}", "Halluc flags"],
        tablefmt="github",
    ) + "\n\n")

    if ctx["human_proxy_macro"]:
        md.append("## Human vs Automated Proxy Scores\n\n")
        md.append("Diff = Human − Proxy (positive = humans rated higher than the scripts/13 judge). "
                  "Proxy `answer_relevance` is computed but unweighted in stage 1.\n\n")
        hp_rows = []
        for m in METRICS:
            v = ctx["human_proxy_macro"].get(m)
            if v:
                hp_rows.append([METRIC_LABELS[m], f"{v['human_avg']:.4f}",
                                f"{v['proxy_avg']:.4f}", f"{v['diff']:+.4f}"])
        md.append(tabulate(hp_rows, headers=["Metric", "Human", "Proxy", "Diff"],
                           tablefmt="github") + "\n\n")

    hr = ctx["halluc"]
    md.append("## Hallucination Summary (separate report)\n\n")
    if hr["answers_with_flag"]:
        md.append(f"- **{hr['answers_with_flag']}** answer(s) flagged by at least one evaluator "
                  f"({hr['confirmed']} confirmed, {hr['disputed']} disputed)\n")
        md.append(f"- **Overall hallucination rate:** {hr['overall_rate']:.2%}\n")
    else:
        md.append("- No answers were flagged as hallucinated by any evaluator.\n")
    md.append("\nSee **[hallucination_report.md](hallucination_report.md)** for the full breakdown.\n\n")

    md.append("## Per-Question Human RAGAS Scores\n\n")
    pq_rows = []
    for qid in sorted(ctx["q_stats"].keys(), key=qid_sort_key):
        stat = ctx["q_stats"][qid]
        pq_rows.append([
            qid, stat["category"], stat["evaluator_count"],
            *[f"{stat['metric_avgs'].get(m):.2f}" if stat["metric_avgs"].get(m) is not None else "N/A"
              for m in METRICS],
            f"{stat['total_avg']:.2f}" if stat["total_avg"] is not None else "N/A",
            "yes" if stat["hallucination_flags"] else "",
        ])
    md.append(tabulate(
        pq_rows,
        headers=["Question ID", "Category", "Evals"] + [METRIC_LABELS[m] for m in METRICS]
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
        "source":              "human evaluator hallucination flag (scripts/16)",
        "independent_of_ragas": True,
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
        "Driven solely by the **human evaluator hallucination flag** collected in "
        "scripts/16 — a direct Yes/No judgement, independent of the five RAGAS "
        "metrics (it is never averaged into them).\n\n",
        "- **confirmed** — a majority of the evaluators who rated the answer flagged it\n",
        "- **disputed** — at least one, but not a majority, flagged it\n\n",
    ]

    md.append("## Summary\n\n")
    summary_rows = [
        ["Answers rated",         hr["total_answers_rated"]],
        ["Answers flagged (>=1)", hr["answers_with_flag"]],
        ["Confirmed (majority)",  hr["confirmed"]],
        ["Disputed (minority)",   hr["disputed"]],
        ["Overall rate",          f"{hr['overall_rate']:.2%}" if hr["overall_rate"] is not None else "N/A"],
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
                f"### {f['question_id']} — {f['severity'].upper()} "
                f"({f['hallucination_flags']}/{f['evaluator_count']} flagged)\n\n"
            )
            md.append(f"- **Category:** {f['category']}\n")
            md.append(f"- **Hallucination rate:** {f['hallucination_rate']:.2%}\n")
            md.append(f"- **Avg faithfulness:** {faith} | **Avg total:** {total} / {MAX_TOTAL}\n")
            flagged_names = ", ".join(b["evaluator"] for b in f["flagged_by"]) or "—"
            md.append(f"- **Flagged by:** {flagged_names} "
                      f"(rated by: {', '.join(f['evaluators'])})\n\n")
            md.append(f"**Question**\n\n> {f['question']}\n\n")
            if f["expected_answer"]:
                md.append(f"**Expected answer**\n\n> {f['expected_answer']}\n\n")
            if f["generated_answer"]:
                md.append(f"**Generated answer**\n\n> {f['generated_answer']}\n\n")
            comments = [b for b in f["flagged_by"] if b["comment"]]
            if comments:
                md.append("**Evaluator comments**\n\n")
                for b in comments:
                    md.append(f"> *{b['evaluator']}:* {b['comment']}\n\n")
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
    responses  = load_all_responses()
    evaluators = load_evaluators()

    if not responses:
        print("No responses found in the database yet.")
        print("Run scripts/16_human_answer_evaluation.py and collect scores first.")
        return

    print(f"  {len(responses)} responses from {len(evaluators)} evaluators.")

    print("Loading questions + proxy scores from the scripts/13 JSONL...")
    by_id, order, mode_sizes = load_questions()
    if not by_id:
        print(f"  Warning: no JSONL found at {JSONL_PATH} -- question text/proxy "
              "comparison will be unavailable.")
    else:
        print(f"  {len(by_id)} questions loaded from {PROXY_LABEL} "
              f"(human={mode_sizes['human']}, ai={mode_sizes['ai']}).")

    print("Computing stats...")
    q_stats           = per_question_stats(responses, by_id)
    macro_avgs        = macro_avg_per_metric(q_stats)
    eval_stats        = per_evaluator_stats(responses, evaluators, mode_sizes)
    iaa, iaa_multi, halluc_disagree_rate = inter_annotator_from_responses(responses)
    human_proxy_macro = human_vs_proxy(q_stats, by_id)
    halluc            = hallucination_report_data(q_stats, by_id, responses)

    ctx = {
        "q_stats":               q_stats,
        "macro_avgs":            macro_avgs,
        "eval_stats":            eval_stats,
        "iaa":                   iaa,
        "iaa_multi_count":       iaa_multi,
        "halluc_disagree_rate":  halluc_disagree_rate,
        "human_proxy_macro":     human_proxy_macro,
        "halluc":                halluc,
        "proxy_label":           PROXY_LABEL if by_id else None,
        "total_responses":       len(responses),
        "total_questions_rated": len(q_stats),
    }

    print_report(ctx)

    json_path, md_path = write_unified_report(ctx)
    h_json, h_md       = write_hallucination_report(halluc, ctx["proxy_label"])

    print(f"\nSaved unified JSON report      : {json_path}")
    print(f"Saved unified Markdown report  : {md_path}")
    print(f"Saved hallucination JSON       : {h_json}")
    print(f"Saved hallucination Markdown   : {h_md}")
    print(f"\nTotal runtime : {time.time() - start:.2f} seconds")
    print("Human evaluation report completed successfully.")


if __name__ == "__main__":
    main()
