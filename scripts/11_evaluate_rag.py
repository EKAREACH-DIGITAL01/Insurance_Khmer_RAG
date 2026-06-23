import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tabulate import tabulate

# Set HF_HOME before any HF import -- models live in retriever/.cache/,
# loaded by Hub ID. See scripts/13_install_models.py for the full explanation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_DIR = _PROJECT_ROOT / "models" / "retriever" / ".cache"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# RAGAS-style End-to-End RAG Evaluation (Internship Proposal §9.3, §11)
# ----------------------------------------------------------------------------
# This script has TWO PHASES, run as separate invocations:
#
#   1. GENERATE -- sweeps every installed generator model, runs every
#      gold question (that HAS an expected_answer -- questions without
#      one are skipped, same as the README always specified) through the
#      exact same hybrid retrieval + prompt pipeline app.py uses, and
#      writes one human-scoring .txt template PER MODEL to
#      reports/generator/generator_eval_template_<model>.txt, plus a
#      matching generator_eval_raw_<model>.json with the same data
#      machine-readable.
#
#   2. AGGREGATE -- run after a human has manually filled in the four
#      1-5 scores (Context Relevance, Faithfulness, Answer Relevance,
#      Answer Correctness) in each template. Parses the filled scores
#      back out, computes per-model averages, builds a cross-model
#      comparison table, and -- exactly as the README specifies --
#      pulls any entry with Faithfulness < 3 OR Answer Correctness < 3
#      into reports/generator/hallucination_audit.txt, the mandatory
#      pre-demo review gate.
#
# Why two phases in one file rather than two files: every other script
# in this project's scripts/ is a single linear pass with no branching
# mode, so two separate files would actually be MORE consistent with
# that convention -- but generate/aggregate here share the entire
# model-registry/path-constants header and the human edits the SAME file
# between phase 1 and phase 2, so keeping them as one file's two
# mutually-exclusive code paths (chosen via the same input()-driven menu
# style 13_install_models.py already uses) avoids two files drifting out
# of sync with each other's constants.

EVALUATION_ROOT = Path("data/08_evaluation")
REPORTS_DIR = Path("reports/generator")
LLM_DIR = Path("models/generator")

EMBEDDING_HF_ID = "BAAI/bge-m3"
ACTIVE_MODEL_DIR = "BAAI-BGE-M3"
INDEX_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance.index"
METADATA_PATH = f"vector_db/{ACTIVE_MODEL_DIR}/khmer_insurance_metadata.json"

LLAMA_SERVER_URL = "http://localhost:8080/v1"
LLAMA_SERVER_ROOT = "http://localhost:8080"
LLAMA_SERVER_STARTUP_TIMEOUT = 120

# Mirrors app.py exactly.
TOP_K = 10
CONTEXT_K = 5
MAX_CHUNKS_PER_DOC = 1
BM25_TOP_K = TOP_K
RRF_K = 60
NGRAM_SIZE = 3
MAX_TOKENS = 1024

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")

# Score fields every template entry must have, in this exact order, and
# the exact label text the aggregator regex below expects to find.
SCORE_FIELDS = [
    "Context Relevance",
    "Faithfulness",
    "Answer Relevance",
    "Answer Correctness",
]

# Duplicated from scripts/13_install_models.py / start_chatbot.py /
# scripts/10_compare_generator_models.py -- see scripts/10's identical
# block for the full explanation of extra_args.
LLM_MODELS = {
    "1": {
        "local_filename": "SeaLLMs-v3-7B-Chat-Q4_K_M.gguf",
        "label": "SeaLLM v3 7B Q4_K_M",
        "extra_args": ["--jinja"],
    },
    "2": {
        "local_filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "label": "Qwen 3.5 9B Q4_K_M",
        "extra_args": ["--jinja", "--reasoning", "off"],
    },
    "3": {
        "local_filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "label": "Gemma 4 E4B Q4_K_M",
        "extra_args": ["--jinja"],
    },
    "4": {
        "local_filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "label": "Llama 3.1 8B Instruct Q4_K_M",
        "extra_args": ["--jinja"],
    },
}

BOLD = "\033[1m"
RESET = "\033[0m"
SEPARATOR_WIDTH = 105
SEPARATOR = "=" * SEPARATOR_WIDTH


def label_to_filename_part(label):
    """SeaLLM v3 7B Q4_K_M -> SeaLLM_v3_7B_Q4_K_M -- used to build each
    model's own template/report filenames."""
    return re.sub(r"\s+", "_", label)


def khmer_aware_tokenize(text):
    """Identical to app.py / scripts/09 / scripts/10."""
    tokens = []
    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            chars = list(word)
            if len(chars) < NGRAM_SIZE:
                tokens.append(word)
            else:
                for i in range(len(chars) - NGRAM_SIZE + 1):
                    tokens.append("".join(chars[i : i + NGRAM_SIZE]))
        else:
            tokens.append(word.lower())
    return tokens


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Identical to app.py / scripts/09 / scripts/10."""
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[cid], chunk_lookup[cid]) for cid in ranked_chunk_ids]


def select_context_chunks(results):
    """Identical to app.py / scripts/10."""
    selected = []
    per_doc_count = {}
    for score, chunk in results:
        doc_name = chunk.get("metadata", {}).get("document_name", "unknown")
        if per_doc_count.get(doc_name, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        selected.append((score, chunk))
        per_doc_count[doc_name] = per_doc_count.get(doc_name, 0) + 1
        if len(selected) >= CONTEXT_K:
            break
    return selected


def build_context(results):
    """Identical to app.py / scripts/10."""
    context_parts = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = metadata.get("document_name", "N/A")
        doc_type = metadata.get(
            "document_type_kh", metadata.get("document_type", "N/A")
        )
        chapter = metadata.get("chapter_title", metadata.get("chapter", "N/A"))
        section = metadata.get("section_title", metadata.get("section", "N/A"))
        legal_unit = metadata.get("legal_unit", "N/A")
        release_date = metadata.get("release_date", "N/A")
        header = (
            f"[Source {rank}] {doc_name} ({doc_type}, {release_date}) | "
            f"{chapter} > {section} | {legal_unit} | {chunk['chunk_id']}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(context_parts)


def build_prompt(question, context):
    """Identical core prompt to app.py / scripts/10 (see scripts/10 for
    why the low_confidence relevance-check branch is omitted here)."""
    not_found_message = "រកមិនឃើញព័ត៌មាននេះនៅក្នុងឯកសារដែលបានផ្តល់។"
    return f"""
You are a Khmer legal assistant.

Answer ONLY from the provided context.

Rules:

- Answer in Khmer.
- Do not invent facts.
- Preserve dates, fees, deadlines and requirements exactly.
- Provide a complete and detailed answer. If the context lists multiple conditions, sub-points, exceptions, or requirements, enumerate ALL of them.
- Only combine information from multiple sources if EACH of them directly addresses the question asked. Do not stitch together unrelated facts from different sources just because they were retrieved -- if most of the retrieved sources don't actually answer the question, say so instead of forcing a combined answer out of all of them.
- Do not reinterpret a number, rate, or statistic for a purpose other than what it was stated for in the source.
- Do not recommend, endorse, or compare any specific insurance product, company, or policy, even if the context mentions one and the question asks for a recommendation.
- Do not calculate, estimate, or infer a premium, payout, deductible, or any monetary amount that is not stated explicitly and directly in the context for that exact purpose.
- Do not give personalized financial or legal advice. If the question asks for a personal recommendation, a calculation, or advice tailored to the user's situation, decline and reply with the message below instead of attempting an answer.
- If the answer is not found in the context, OR the question asks for a product recommendation, a premium/payout calculation, or personalized advice, reply EXACTLY with this message and nothing else:

{not_found_message}

Context:
{context}

Question:
{question}

Answer:
"""


def is_installed(info):
    return (LLM_DIR / info["local_filename"]).exists()


def start_llama_server(info):
    """Identical to scripts/10's start_llama_server() -- see there for
    the full explanation."""
    llama_server_bin = shutil.which("llama-server")
    if llama_server_bin is None:
        raise SystemExit(
            "llama-server not found. Please install llama.cpp and add it to PATH."
        )

    model_path = LLM_DIR / info["local_filename"]

    process = subprocess.Popen(
        [
            llama_server_bin,
            "-m",
            str(model_path),
            "-ngl",
            "999",
            "--ctx-size",
            "16384",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]
        + info["extra_args"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start_time = time.time()
    while True:
        if time.time() - start_time > LLAMA_SERVER_STARTUP_TIMEOUT:
            process.kill()
            raise SystemExit(
                f"llama-server failed to start within "
                f"{LLAMA_SERVER_STARTUP_TIMEOUT}s for {info['label']}."
            )
        try:
            response = requests.get(f"{LLAMA_SERVER_ROOT}/v1/models", timeout=5)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)

    return process


def stop_llama_server(process):
    """Identical to scripts/10's stop_llama_server()."""
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    time.sleep(2)


# ----------------------------------------------------------------------------
# PHASE 1: GENERATE
# ----------------------------------------------------------------------------


def load_retrieval_stack():
    """Loads embedding model, FAISS index, BM25 index -- ONCE, shared
    across all generator models in the sweep (retrieval is
    generator-independent)."""
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run: python scripts/07_generate_embeddings.py"
        )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_HF_ID)

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Building BM25 index...")
    bm25_corpus_tokens = [
        khmer_aware_tokenize(chunk.get("embedding_text") or chunk["text"])
        for chunk in chunks
    ]
    bm25_index = BM25Okapi(bm25_corpus_tokens)

    return embedding_model, faiss_index, chunks, bm25_index


def collect_gold_entries():
    """Loads every gold question that HAS an expected_answer, across all
    categories -- questions without one are skipped here, exactly as the
    README has always specified for this script. Returns a flat list with
    category attached, since the human template groups by category."""
    gold_files = sorted(EVALUATION_ROOT.rglob("gold_questions.json"))
    if not gold_files:
        raise ValueError("No gold_questions.json files found.")

    entries = []
    skipped_no_answer = 0
    for gold_file in gold_files:
        category = gold_file.parent.name
        with open(gold_file, "r", encoding="utf-8") as f:
            gold_questions = json.load(f)

        if not gold_questions:
            raise ValueError(f"{gold_file} is empty.")

        for item in gold_questions:
            if not item.get("expected_answer"):
                skipped_no_answer += 1
                continue
            entries.append(
                {
                    "category": category,
                    "question": item["question"],
                    "expected_chunks": item.get("expected_chunks", []),
                    "expected_answer": item["expected_answer"],
                }
            )

    if not entries:
        raise ValueError(
            "No gold questions with an expected_answer field were found -- "
            "nothing for this script to generate."
        )

    print(
        f"Loaded {len(entries)} gold questions with expected_answer "
        f"({skipped_no_answer} skipped -- no expected_answer field)."
    )
    return entries


def generate_for_model(client, model_name, entries, embedding_model, faiss_index, chunks, bm25_index):
    """Runs every gold entry through hybrid retrieval + the live prompt,
    against the currently-running model. Returns a list of result dicts
    ready to be written into both the human template and the raw JSON."""

    def retrieve_dense(question):
        query_embedding = embedding_model.encode([question], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        scores, indices = faiss_index.search(query_embedding, k=TOP_K)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            results.append((float(score), chunks[idx]))
        return results

    def retrieve_bm25(question):
        tokenized_query = khmer_aware_tokenize(question)
        scores = bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
        return [(float(scores[i]), chunks[i]) for i in top_indices]

    results = []
    for i, entry in enumerate(entries, start=1):
        print(f"  [{i}/{len(entries)}] {entry['question'][:60]}...")

        fused_results = reciprocal_rank_fusion(
            [retrieve_dense(entry["question"]), retrieve_bm25(entry["question"])]
        )
        context_results = select_context_chunks(fused_results)
        context = build_context(context_results)
        prompt = build_prompt(entry["question"], context)

        sources = [
            {
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk.get("metadata", {}).get("document_name", "N/A"),
            }
            for _, chunk in context_results
        ]

        answer_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            for chunk_resp in response:
                delta = chunk_resp.choices[0].delta.content
                if delta:
                    answer_text += delta
            try:
                response.close()
            except Exception:
                pass
        except Exception as e:
            answer_text = f"[GENERATION FAILED: {e}]"

        results.append(
            {
                "category": entry["category"],
                "question": entry["question"],
                "expected_chunks": entry["expected_chunks"],
                "expected_answer": entry["expected_answer"],
                "retrieved_sources": sources,
                "generated_answer": answer_text,
            }
        )

    return results


def write_template(label, results):
    """Writes the human-scoring .txt template for one model. Score lines
    use a plain 'Label: ' prefix with no value -- the human types the
    1-5 number directly after the colon. This exact 'Label: ' prefix is
    what the AGGREGATE phase's regex looks for, so the format here and
    the parser below must stay in sync."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename_part = label_to_filename_part(label)
    template_path = REPORTS_DIR / f"generator_eval_template_{filename_part}.txt"
    raw_path = REPORTS_DIR / f"generator_eval_raw_{filename_part}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("=" * SEPARATOR_WIDTH)
    lines.append(f"GENERATOR EVALUATION TEMPLATE -- {label}")
    lines.append("=" * SEPARATOR_WIDTH)
    lines.append("")
    lines.append(
        "Score each entry 1-5 on the four lines below its answer. Save this "
        "file, then run this script again and choose 'Aggregate' to compute "
        "results and the hallucination audit."
    )
    lines.append("")
    lines.append("Scoring guide (1=worst, 5=best):")
    lines.append("  Context Relevance  -- did the retrieved sources actually relate to the question?")
    lines.append("  Faithfulness       -- does the answer ONLY state facts present in the context (no invented facts)?")
    lines.append("  Answer Relevance   -- does the answer actually address what was asked?")
    lines.append("  Answer Correctness -- does the answer match expected_answer's substance?")
    lines.append("")

    for i, r in enumerate(results, start=1):
        lines.append("-" * SEPARATOR_WIDTH)
        lines.append(f"ENTRY {i}  [{r['category']}]")
        lines.append("-" * SEPARATOR_WIDTH)
        lines.append(f"Question: {r['question']}")
        lines.append("")
        lines.append(f"Expected Answer: {r['expected_answer']}")
        lines.append("")
        source_str = ", ".join(
            f"{s['chunk_id']} ({s['document_name']})" for s in r["retrieved_sources"]
        ) or "none"
        lines.append(f"Retrieved Sources: {source_str}")
        lines.append("")
        lines.append(f"Generated Answer: {r['generated_answer']}")
        lines.append("")
        for field in SCORE_FIELDS:
            lines.append(f"{field}: ")
        lines.append("")

    with open(template_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved template : {template_path}")
    print(f"  Saved raw data : {raw_path}")


def run_generate_phase():
    embedding_model, faiss_index, chunks, bm25_index = load_retrieval_stack()
    entries = collect_gold_entries()

    installed_models = {
        key: info for key, info in LLM_MODELS.items() if is_installed(info)
    }
    if not installed_models:
        raise SystemExit(
            "No generator models installed -- run: python scripts/13_install_models.py"
        )

    for key, info in installed_models.items():
        label = info["label"]
        print(f"\n{'=' * SEPARATOR_WIDTH}\nStarting llama-server for: {label}\n{'=' * SEPARATOR_WIDTH}")

        process = start_llama_server(info)
        try:
            client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
            model_response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=5)
            model_name = model_response.json()["data"][0]["id"]

            print(f"Generating answers for {label}...")
            results = generate_for_model(
                client, model_name, entries, embedding_model, faiss_index, chunks, bm25_index
            )
            write_template(label, results)
        finally:
            print(f"Stopping llama-server for: {label}")
            stop_llama_server(process)

    print(
        "\nAll templates written to reports/generator/. Fill in the four "
        "score lines under each entry, save, then re-run this script and "
        "choose 'Aggregate'."
    )


# ----------------------------------------------------------------------------
# PHASE 2: AGGREGATE
# ----------------------------------------------------------------------------

# Matches "Faithfulness: 4" (any whitespace around the number) at the
# start of a line. Must match write_template()'s "Label: " prefix exactly.
SCORE_LINE_PATTERN = {
    field: re.compile(rf"^{re.escape(field)}:\s*([1-5])\s*$") for field in SCORE_FIELDS
}
ENTRY_HEADER_PATTERN = re.compile(r"^ENTRY\s+(\d+)\s+\[(.+?)\]$")
QUESTION_PATTERN = re.compile(r"^Question:\s*(.*)$")


def parse_filled_template(template_path):
    """Re-parses a filled-in template back into structured entries. Each
    entry's scores dict will be missing any field the human left blank --
    those are reported, not silently defaulted, since a blank score is
    very different from an intentional middle score."""
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    entries = []
    current = None

    for line in lines:
        header_match = ENTRY_HEADER_PATTERN.match(line.strip())
        if header_match:
            if current is not None:
                entries.append(current)
            current = {
                "entry_number": int(header_match.group(1)),
                "category": header_match.group(2),
                "question": None,
                "scores": {},
            }
            continue

        if current is None:
            continue

        question_match = QUESTION_PATTERN.match(line)
        if question_match:
            current["question"] = question_match.group(1)
            continue

        for field, pattern in SCORE_LINE_PATTERN.items():
            score_match = pattern.match(line.strip())
            if score_match:
                current["scores"][field] = int(score_match.group(1))
                break

    if current is not None:
        entries.append(current)

    return entries


def run_aggregate_phase():
    template_paths = sorted(REPORTS_DIR.glob("generator_eval_template_*.txt"))
    if not template_paths:
        raise SystemExit(
            "No templates found in reports/generator/ -- run the 'Generate' "
            "phase first."
        )

    model_summaries = {}
    hallucination_entries = []
    incomplete_entries = []

    for template_path in template_paths:
        label_part = template_path.stem.replace("generator_eval_template_", "")
        label = label_part.replace("_", " ")
        entries = parse_filled_template(template_path)

        scored_entries = [e for e in entries if len(e["scores"]) == len(SCORE_FIELDS)]
        unscored_count = len(entries) - len(scored_entries)
        if unscored_count:
            incomplete_entries.append((label, unscored_count, len(entries)))

        if not scored_entries:
            print(f"{label}: no fully-scored entries yet -- skipping in comparison.")
            continue

        averages = {
            field: sum(e["scores"][field] for e in scored_entries) / len(scored_entries)
            for field in SCORE_FIELDS
        }
        model_summaries[label] = {
            "averages": averages,
            "scored_count": len(scored_entries),
            "total_count": len(entries),
        }

        for e in scored_entries:
            if e["scores"]["Faithfulness"] < 3 or e["scores"]["Answer Correctness"] < 3:
                hallucination_entries.append(
                    {
                        "model": label,
                        "entry_number": e["entry_number"],
                        "category": e["category"],
                        "question": e["question"],
                        "scores": e["scores"],
                    }
                )

    if incomplete_entries:
        print("\nIncomplete templates (some entries missing one or more scores):")
        for label, unscored, total in incomplete_entries:
            print(f"  - {label}: {unscored}/{total} entries not fully scored")

    if not model_summaries:
        raise SystemExit(
            "\nNo template has any fully-scored entries yet -- fill in the "
            "four score lines under at least one entry and re-run."
        )

    comparison_rows = []
    for label, summary in model_summaries.items():
        avg = summary["averages"]
        comparison_rows.append(
            [
                label,
                f"{avg['Context Relevance']:.2f}",
                f"{avg['Faithfulness']:.2f}",
                f"{avg['Answer Relevance']:.2f}",
                f"{avg['Answer Correctness']:.2f}",
                f"{summary['scored_count']}/{summary['total_count']}",
            ]
        )
    comparison_headers = [
        "Generator",
        "Context Relevance",
        "Faithfulness",
        "Answer Relevance",
        "Answer Correctness",
        "Scored / Total",
    ]

    banner = "=" * SEPARATOR_WIDTH
    print(f"\n{banner}")
    print(f"{BOLD}HUMAN EVALUATION -- ALL GENERATORS (avg of 4 RAGAS-style metrics){RESET}")
    print(banner)
    print(
        tabulate(
            comparison_rows,
            headers=comparison_headers,
            tablefmt="grid",
            colalign=("center",) * len(comparison_headers),
        )
    )

    # Hallucination audit -- exactly the README's existing rule:
    # Faithfulness < 3 OR Answer Correctness < 3.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = REPORTS_DIR / "hallucination_audit.txt"
    audit_lines = [
        "=" * SEPARATOR_WIDTH,
        "HALLUCINATION AUDIT -- entries with Faithfulness < 3 OR Answer Correctness < 3",
        "Mandatory pre-demo review gate (Internship Proposal §11)",
        "=" * SEPARATOR_WIDTH,
        "",
    ]
    if not hallucination_entries:
        audit_lines.append("No flagged entries -- every scored entry met the threshold.")
    else:
        for h in hallucination_entries:
            audit_lines.append("-" * SEPARATOR_WIDTH)
            audit_lines.append(f"Model: {h['model']}  |  Entry {h['entry_number']}  |  [{h['category']}]")
            audit_lines.append(f"Question: {h['question']}")
            audit_lines.append(
                f"Scores: Faithfulness={h['scores']['Faithfulness']}, "
                f"Answer Correctness={h['scores']['Answer Correctness']}"
            )
            audit_lines.append("")

    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))

    print(f"\n{len(hallucination_entries)} entries flagged in hallucination audit.")
    print(f"Saved : {audit_path}")

    json_path = REPORTS_DIR / "generator_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": model_summaries,
                "comparison": {"headers": comparison_headers, "rows": comparison_rows},
                "hallucination_flagged_count": len(hallucination_entries),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved : {json_path}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

print(
    """
========================================
RAG Generator Evaluation (RAGAS-style)
========================================

1. Generate  -- sweep all installed generators, write one human-scoring
                template per model to reports/generator/
2. Aggregate -- parse filled-in templates, compute averages, write the
                hallucination audit (run this AFTER scoring the templates)
"""
)

choice = input("Choice: ").strip()

if choice == "1":
    run_generate_phase()
elif choice == "2":
    run_aggregate_phase()
else:
    raise SystemExit("Invalid choice.")