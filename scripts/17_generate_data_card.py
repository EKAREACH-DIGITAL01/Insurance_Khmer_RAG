import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from tabulate import tabulate

# ----------------------------------------------------------------------------
# Data Card Generator (Internship Proposal §11)
# ----------------------------------------------------------------------------
# PURPOSE: roll up everything the pipeline produced -- the source corpus, the
# chunked knowledge base, the evaluation sets, the model-selection experiments
# and the quality evaluations -- into ONE data card, written as both JSON and
# Markdown.
#
# Inputs (all optional -- missing pieces degrade to "not available")
#   data/document_metadata.json                     (source documents)
#   data/08_merged_chunked_documents/*_chunks.json  (chunked KB)
#   data/09_evaluation/*/gold_questions.json        (gold eval sets)
#   data/10_adversarial/*_questions.json            (safety eval sets)
#   reports/embedding_model_comparison/*.json       (script 09)
#   reports/retrieval_mode_comparison/*.json        (script 10)
#   reports/generator_model_comparison/*.json       (script 15)
#   reports/human_golden_question_evaluation/*.json (script 12)
#   reports/safety_gate_evaluation/*.json           (script 16)
#   vector_db/*/embedding_model_info.json           (script 08)
#
# Outputs (under reports/data_card/)
#   data_card.json   machine-readable
#   data_card.md     human-readable
# ----------------------------------------------------------------------------

PROJECT_NAME = "Khmer Insurance RAG Chatbot"

DOC_METADATA_PATH = Path("data/document_metadata.json")
CHUNKS_DIR        = Path("data/08_merged_chunked_documents")
GOLD_DIR          = Path("data/09_evaluation")
ADVERSARIAL_DIR   = Path("data/10_adversarial")
VECTOR_DB_DIR     = Path("vector_db")

EMBEDDING_REPORT  = Path("reports/embedding_model_comparison/embedding_model_comparison.json")
RETRIEVAL_REPORT  = Path("reports/retrieval_mode_comparison/retrieval_mode_comparison.json")
GENERATOR_REPORT  = Path("reports/generator_model_comparison/generator_model_comparison.json")
SAFETY_REPORT     = Path("reports/safety_gate_evaluation/safety_gate_evaluation.json")
GOLD_VALIDATION   = Path("reports/human_golden_question_evaluation/"
                         "human_golden_question_evaluation_report.json")

OUTPUT_DIR = Path("reports/data_card")

DOC_TYPE_LABELS = {
    "law":        "Laws (ច្បាប់)",
    "sub_decree": "Sub-decrees (អនុក្រឹត្យ)",
    "prakas":     "Prakas (ប្រកាស)",
    "guidance":   "Guidance (គោលការណ៍ណែនាំ)",
}


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _round(x, n=4):
    return round(x, n) if isinstance(x, (int, float)) else x


# ----------------------------------------------------------------------------
# 1. Source corpus
# ----------------------------------------------------------------------------

def build_source_corpus():
    meta = load_json(DOC_METADATA_PATH)
    if not meta:
        return None

    by_type = defaultdict(list)
    for doc in meta.values():
        by_type[doc.get("document_type", "unknown")].append(doc)

    types = {}
    for dtype, docs in by_type.items():
        dates = sorted(d["release_date"] for d in docs if d.get("release_date"))
        types[dtype] = {
            "documents": len(docs),
            "earliest":  dates[0] if dates else None,
            "latest":    dates[-1] if dates else None,
        }

    all_dates = sorted(d["release_date"] for d in meta.values() if d.get("release_date"))
    return {
        "total_documents": len(meta),
        "by_type":         types,
        "date_range":      [all_dates[0], all_dates[-1]] if all_dates else None,
        "language":        "Khmer (km)",
        "domain":          "Cambodian insurance law & regulation",
    }


# ----------------------------------------------------------------------------
# 2. Chunked knowledge base
# ----------------------------------------------------------------------------

def build_knowledge_base():
    if not CHUNKS_DIR.exists():
        return None

    categories = {}
    total_chunks = 0
    all_sizes = []
    chunking_methods = set()

    for path in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        chunks = load_json(path)
        if not chunks:
            continue
        category = path.stem.replace("_chunks", "")
        docs = set()
        sizes = []
        for c in chunks:
            md = c.get("metadata", {})
            docs.add(md.get("document_id"))
            size = md.get("chunk_size") or len(c.get("content", ""))
            sizes.append(size)
            if md.get("chunking_method"):
                chunking_methods.add(md["chunking_method"])
        all_sizes.extend(sizes)
        total_chunks += len(chunks)
        categories[category] = {
            "documents":      len([d for d in docs if d]),
            "chunks":         len(chunks),
            "avg_chunk_size": round(mean(sizes), 1) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
        }

    return {
        "total_chunks":     total_chunks,
        "avg_chunk_size":   round(mean(all_sizes), 1) if all_sizes else 0,
        "chunking_methods": sorted(chunking_methods),
        "by_category":      categories,
    }


# ----------------------------------------------------------------------------
# 3. Embedding / vector index
# ----------------------------------------------------------------------------

def build_vector_index(winner_label):
    if not VECTOR_DB_DIR.exists():
        return None

    by_label = {}
    for info_path in VECTOR_DB_DIR.glob("*/embedding_model_info.json"):
        info = load_json(info_path)
        if info and info.get("label"):
            by_label[info["label"]] = info

    indexed_models = [
        {
            "label":     label,
            "dimension": info.get("embedding_dimension"),
            "chunks":    info.get("chunk_count"),
        }
        for label, info in sorted(by_label.items())
    ]

    active = by_label.get(winner_label)
    return {
        "active_model":       winner_label,
        "active_dimension":   active.get("embedding_dimension") if active else None,
        "active_chunk_count": active.get("chunk_count") if active else None,
        "indexed_models":     indexed_models,
    }


# ----------------------------------------------------------------------------
# 4. Evaluation sets
# ----------------------------------------------------------------------------

def build_eval_sets():
    gold = {}
    gold_total = 0
    if GOLD_DIR.exists():
        for path in sorted(GOLD_DIR.glob("*/gold_questions.json")):
            qs = load_json(path) or []
            gold[path.parent.name] = len(qs)
            gold_total += len(qs)

    adversarial = {}
    adv_total = 0
    if ADVERSARIAL_DIR.exists():
        for path in sorted(ADVERSARIAL_DIR.glob("*_questions.json")):
            qs = load_json(path) or []
            cat = path.stem.replace("_questions", "")
            behaviors = Counter(q.get("expected_behavior") for q in qs)
            adversarial[cat] = {
                "total":  len(qs),
                "refuse": behaviors.get("refuse", 0),
                "answer": behaviors.get("answer", 0),
            }
            adv_total += len(qs)

    # The gold set is itself validated by humans (scripts/11 -> 12). Without
    # this the card describes the yardstick but not whether the yardstick was
    # checked.
    validation = None
    val = load_json(GOLD_VALIDATION)
    if val:
        cov = val.get("coverage", {})
        overall = val.get("overall_means") or {}
        agree = val.get("inter_evaluator_agreement") or {}
        validation = {
            "evaluators":       cov.get("evaluators"),
            "responses":        cov.get("total_responses"),
            "questions_rated":  cov.get("questions_evaluated"),
            "responses_per_question_mean": cov.get("responses_per_question_mean"),
            "combined_mean":    overall.get("combined"),
            "curation_flags":   len(val.get("curation_flags") or []),
            "questions_with_two_or_more_raters": agree.get("_questions_with_overlap"),
        }

    return {
        "gold_questions": {"by_category": gold, "total": gold_total,
                           "human_validation": validation},
        "adversarial":    {"by_category": adversarial, "total": adv_total},
    }


# ----------------------------------------------------------------------------
# 5. Model selection (winners from the comparison experiments)
# ----------------------------------------------------------------------------

def build_model_selection():
    selection = {"embedding": None, "retrieval": None, "generator": None}

    emb = load_json(EMBEDDING_REPORT)
    if emb and emb.get("ranking"):
        ranking = emb["ranking"]
        winner = ranking[0]
        selection["embedding"] = {
            "winner":            winner.get("label"),
            "macro_hit_at_5":    winner.get("macro_hit_at_5"),
            "macro_recall_at_5": winner.get("macro_recall_at_5"),
            "macro_mrr_at_5":    _round(winner.get("macro_mrr_at_5")),
            "candidates":        [r.get("label") for r in ranking],
        }

    ret = load_json(RETRIEVAL_REPORT)
    if ret and ret.get("comparison", {}).get("rows"):
        rows = ret["comparison"]["rows"]
        hit = next((r for r in rows if "Hit Rate" in r.get("metric", "")), None)
        winner_mode = "hybrid"
        if hit and hit.get("dense") is not None and hit.get("hybrid") is not None:
            winner_mode = "hybrid" if hit["hybrid"] >= hit["dense"] else "dense"
        selection["retrieval"] = {
            "winner": winner_mode,
            "metrics_at_5": {
                r["metric"]: {"dense": r.get("dense"), "hybrid": r.get("hybrid"),
                              "delta": r.get("delta")}
                for r in rows
            },
        }

    gen = load_json(GENERATOR_REPORT)
    if gen and gen.get("ranking"):
        winner_label = gen.get("recommendation")
        winner = next(
            (s for s in gen["ranking"] if s.get("label") == winner_label),
            gen["ranking"][0],
        )
        selection["generator"] = {
            "winner":       winner.get("label"),
            "combined_avg": winner.get("combined_avg"),
            "scores_avg":   winner.get("scores_avg"),
            "weights":      gen.get("weights"),
            "candidates":   [s.get("label") for s in gen["ranking"]],
        }

    return selection


# ----------------------------------------------------------------------------
# 6. Quality evaluation (automated + safety)
# ----------------------------------------------------------------------------

def build_quality():
    quality = {"automated_judge": None, "safety_gates": None}

    gen = load_json(GENERATOR_REPORT)
    if gen and gen.get("ranking"):
        winner_label = gen.get("recommendation")
        winner = next(
            (s for s in gen["ranking"] if s.get("label") == winner_label),
            gen["ranking"][0],
        )
        quality["automated_judge"] = {
            "model":             winner.get("label"),
            "total_questions":   winner.get("total_questions"),
            "combined_avg":      winner.get("combined_avg"),
            "scores_avg":        winner.get("scores_avg"),
            "truncated_answers": winner.get("truncated_answers"),
            "blank_answers":     winner.get("blank_answers"),
        }

    safety = load_json(SAFETY_REPORT)
    if safety:
        quality["safety_gates"] = {
            "model":     safety.get("model"),
            "total":     safety.get("total"),
            "correct":   safety.get("correct"),
            "accuracy":  safety.get("accuracy"),
            "confusion": safety.get("confusion"),
        }

    return quality


# ----------------------------------------------------------------------------
# Assemble the full data card
# ----------------------------------------------------------------------------

def build_data_card():
    selection = build_model_selection()
    emb_winner = (selection.get("embedding") or {}).get("winner")
    return {
        "project":         PROJECT_NAME,
        "generated_at":    int(time.time()),
        "generated_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_corpus":   build_source_corpus(),
        "knowledge_base":  build_knowledge_base(),
        "vector_index":    build_vector_index(emb_winner),
        "evaluation_sets": build_eval_sets(),
        "model_selection": selection,
        "quality":         build_quality(),
        "intended_use": (
            "Retrieval-augmented question answering over Cambodian insurance "
            "laws, sub-decrees, prakas and guidance, in Khmer. The assistant "
            "answers strictly from the retrieved source documents and refuses "
            "out-of-scope, advisory, pricing and recommendation requests."
        ),
        # "Coverage" is avoided here: it means three different things in this
        # project (the corpus, the gold-question evaluation, and insurance
        # coverage itself), so each limitation names what it actually refers to.
        "limitations": [
            "The assistant only knows the documents in this corpus, which is a "
            "fixed snapshot -- regulations issued or amended after the listed "
            "date range are absent, and it cannot say so when asked.",
            "Answers are only as accurate as the underlying documents and the "
            "retrieval step; low-similarity queries are refused by design.",
            "Khmer OCR/extraction and chunking can introduce minor text errors.",
            "Generated answers are scored by an LLM judge, not by humans. The "
            "judge was measured for self-consistency (85-96% exact agreement on "
            "a repeat pass) but was never validated against human judgement of "
            "the answers, so the generator ranking is a defensible automated "
            "comparison rather than a measured answer-quality result.",
            "Hallucination counts are an upper bound: they are judged against "
            "the gold answer, so a detail correctly taken from the source "
            "document but omitted from the gold answer is still counted. "
            "Verbose models are the most affected.",
            "Not legal advice -- users are directed to a licensed agent or the "
            "Insurance Regulator of Cambodia (IRC) for binding guidance.",
        ],
    }


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------

def _na(x):
    return x if x is not None else "N/A"


def render_markdown(card):
    md = [
        f"# Data Card — {card['project']}\n\n",
        f"Generated: {card['generated_human']}\n\n",
        f"{card['intended_use']}\n\n",
    ]

    # --- Source corpus ---
    md.append("## 1. Source Corpus\n\n")
    sc = card["source_corpus"]
    if not sc:
        md.append("_Source metadata not available._\n\n")
    else:
        md.append(f"- **Total documents:** {sc['total_documents']}\n")
        md.append(f"- **Language:** {sc['language']}\n")
        md.append(f"- **Domain:** {sc['domain']}\n")
        if sc["date_range"]:
            md.append(f"- **Release date range:** {sc['date_range'][0]} → {sc['date_range'][1]}\n")
        md.append("\n")
        rows = [
            [DOC_TYPE_LABELS.get(t, t), d["documents"], _na(d["earliest"]), _na(d["latest"])]
            for t, d in sorted(sc["by_type"].items())
        ]
        md.append(tabulate(rows, headers=["Document Type", "Documents", "Earliest", "Latest"],
                           tablefmt="github") + "\n\n")

    # --- Knowledge base ---
    md.append("## 2. Chunked Knowledge Base\n\n")
    kb = card["knowledge_base"]
    if not kb:
        md.append("_Chunked corpus not available._\n\n")
    else:
        md.append(f"- **Total chunks:** {kb['total_chunks']}\n")
        md.append(f"- **Avg chunk size:** {kb['avg_chunk_size']} chars\n")
        md.append(f"- **Chunking method(s):** {', '.join(kb['chunking_methods']) or 'N/A'}\n\n")
        rows = [
            [cat, d["documents"], d["chunks"], d["avg_chunk_size"],
             f"{d['min_chunk_size']}–{d['max_chunk_size']}"]
            for cat, d in sorted(kb["by_category"].items())
        ]
        rows.append(["**TOTAL**", "", kb["total_chunks"], kb["avg_chunk_size"], ""])
        md.append(tabulate(rows, headers=["Category", "Docs", "Chunks", "Avg Size", "Size Range"],
                           tablefmt="github") + "\n\n")

    # --- Vector index ---
    vi = card["vector_index"]
    if vi:
        md.append("## 3. Vector Index\n\n")
        md.append(f"- **Active embedding model:** {_na(vi['active_model'])}\n")
        md.append(f"- **Embedding dimension:** {_na(vi['active_dimension'])}\n")
        md.append(f"- **Indexed chunks:** {_na(vi['active_chunk_count'])}\n\n")
        if vi["indexed_models"]:
            rows = [[m["label"], _na(m["dimension"]), _na(m["chunks"])] for m in vi["indexed_models"]]
            md.append(tabulate(rows, headers=["Model", "Dim", "Chunks"], tablefmt="github") + "\n\n")

    # --- Evaluation sets ---
    md.append("## 4. Evaluation Sets\n\n")
    es = card["evaluation_sets"]
    md.append(f"**Gold question set** — {es['gold_questions']['total']} questions\n\n")
    if es["gold_questions"]["by_category"]:
        rows = [[c, n] for c, n in sorted(es["gold_questions"]["by_category"].items())]
        md.append(tabulate(rows, headers=["Category", "Questions"], tablefmt="github") + "\n\n")
    hv = es["gold_questions"].get("human_validation")
    if hv:
        md.append("*Human validation of this set* — the gold questions were "
                  "themselves rated by people, so the yardstick is checked and "
                  "not merely assumed:\n\n")
        md.append(tabulate([
            ["Evaluators",                    hv["evaluators"]],
            ["Ratings collected",             hv["responses"]],
            ["Questions rated",               hv["questions_rated"]],
            ["Questions rated by 2+ people",  hv["questions_with_two_or_more_raters"]],
            ["Mean rating (1-5)",             hv["combined_mean"]],
            ["Questions flagged for curation", hv["curation_flags"]],
        ], headers=["Metric", "Value"], tablefmt="github") + "\n\n")

    md.append(f"**Adversarial / safety set** — {es['adversarial']['total']} questions\n\n")
    if es["adversarial"]["by_category"]:
        rows = [[c, d["total"], d["refuse"], d["answer"]]
                for c, d in sorted(es["adversarial"]["by_category"].items())]
        md.append(tabulate(rows, headers=["Category", "Total", "Must Refuse", "Must Answer"],
                           tablefmt="github") + "\n\n")

    # --- Model selection ---
    md.append("## 5. Model Selection\n\n")
    ms = card["model_selection"]
    emb = ms.get("embedding")
    if emb:
        md.append(f"- **Embedding model:** `{emb['winner']}` "
                  f"(Hit@5={_na(emb['macro_hit_at_5'])}, Recall@5={_na(emb['macro_recall_at_5'])}, "
                  f"MRR@5={_na(emb['macro_mrr_at_5'])}); "
                  f"compared {len(emb['candidates'])} models\n")
    ret = ms.get("retrieval")
    if ret:
        md.append(f"- **Retrieval mode:** `{ret['winner']}`\n")
    gen = ms.get("generator")
    if gen:
        md.append(f"- **Generator model:** `{gen['winner']}` "
                  f"(combined judge score {_na(gen['combined_avg'])}/5); "
                  f"compared {len(gen['candidates'])} models\n")
    md.append("\n")
    if ret and ret.get("metrics_at_5"):
        md.append("Retrieval mode comparison (macro-avg @5):\n\n")
        rows = [[m, v.get("dense"), v.get("hybrid"), v.get("delta")]
                for m, v in ret["metrics_at_5"].items()]
        md.append(tabulate(rows, headers=["Metric", "Dense", "Hybrid", "Δ"],
                           tablefmt="github") + "\n\n")

    # --- Quality evaluation ---
    md.append("## 6. Quality Evaluation\n\n")
    q = card["quality"]
    aj = q.get("automated_judge")
    if aj:
        md.append(f"### Automated LLM-judge ({aj['model']}, {aj['total_questions']} questions)\n\n")
        if aj.get("scores_avg"):
            rows = [[k.replace("_", " ").title(), v] for k, v in aj["scores_avg"].items()]
            rows.append(["**Combined**", aj["combined_avg"]])
            md.append(tabulate(rows, headers=["Metric", "Avg (1-5)"], tablefmt="github") + "\n\n")
    sg = q.get("safety_gates")
    if sg:
        md.append("### Safety gates\n\n")
        md.append(f"- **Model:** {_na(sg['model'])}\n")
        if sg.get("accuracy") is not None:
            md.append(f"- **Accuracy:** {_na(sg['correct'])}/{_na(sg['total'])} "
                      f"({sg['accuracy']:.0%})\n")
        else:
            md.append("- **Accuracy:** N/A\n")
        if sg.get("confusion"):
            c = sg["confusion"]
            md.append(f"- **Confusion:** TR={c.get('TR')}, TA={c.get('TA')}, "
                      f"FA={c.get('FA')}, FR={c.get('FR')}\n")
        md.append("\n")
    if not (aj or sg):
        md.append("_No quality evaluation reports available yet._\n\n")

    # --- Limitations ---
    md.append("## 7. Limitations\n\n")
    for lim in card["limitations"]:
        md.append(f"- {lim}\n")
    md.append("\n")

    return "".join(md)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    start = time.time()
    print("Building data card from pipeline artifacts...")
    card = build_data_card()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "data_card.json"
    md_path   = OUTPUT_DIR / "data_card.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(card))

    # console summary of what was found vs missing
    def status(x):
        return "ok" if x else "missing"
    print(f"  source_corpus   : {status(card['source_corpus'])}")
    print(f"  knowledge_base  : {status(card['knowledge_base'])}")
    print(f"  vector_index    : {status(card['vector_index'])}")
    print(f"  model_selection : emb={status(card['model_selection']['embedding'])} "
          f"ret={status(card['model_selection']['retrieval'])} "
          f"gen={status(card['model_selection']['generator'])}")
    print(f"  quality         : judge={status(card['quality']['automated_judge'])} "
          f"safety={status(card['quality']['safety_gates'])}")

    print(f"\nSaved JSON data card     : {json_path}")
    print(f"Saved Markdown data card : {md_path}")
    print(f"Total runtime : {time.time() - start:.2f}s")
    print("Data card generated successfully.")


if __name__ == "__main__":
    main()
