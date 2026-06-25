# Khmer Insurance Law Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about
Cambodian insurance law in Khmer, grounded entirely in a curated corpus of
laws, sub-decrees, prakas (ministerial regulations), and regulatory
guidance documents.

Built with a local embedding model (BGE-M3), a local FAISS vector index,
hybrid retrieval (FAISS + BM25, Reciprocal Rank Fusion), and a local
quantized generator LLM (choice of SeaLLM v3 7B, Qwen 3.5 9B, Gemma 4 E4B,
or Llama 3.1 8B Instruct) served via `llama.cpp` — no external API calls
happen at chat time.

---

## Features

- **Khmer-first RAG** over four document categories: `laws`, `prakas`,
  `sub_decrees`, and `guidance`.
- **Fully local inference** — embedding, retrieval, and generation all run
  on your machine via `llama-server`; only the one-time data-build pipeline
  uses an external API (Gemini, for OCR).
- **Hybrid retrieval** (FAISS dense search + BM25 lexical search, fused
  via Reciprocal Rank Fusion) for improved recall on Khmer insurance
  terminology that semantic search alone misses.
- **Layered domain/relevance gating** before generation ever runs:
  - Non-Khmer input is rejected instantly (no LLM call).
  - Degenerate/too-short input is rejected instantly.
  - Deterministic scope-violation gate blocks recommendation, pricing, and
    personal-advice questions before retrieval.
  - Low-similarity retrieval is rejected before an expensive generation
    cycle starts, with the threshold calibrated against real test queries
    (see [Configuration](#configuration) below).
  - Borderline-confidence retrieval still generates, but with an extra
    prompt instruction asking the model to double-check topical relevance
    before answering, plus a visible "low confidence" warning in the UI.
- **Live progress ladder** showing each pipeline phase (embedding, FAISS
  search, BM25 search, RRF fusion, context building, generation) with
  timing, plus a **Stop** button that can interrupt generation mid-stream.
- **Per-answer diagnostics page** — every answer can be inspected for
  retrieval scores (FAISS, BM25, and fused RRF per chunk), provenance
  badges showing which retrieval engine surfaced each chunk, full chunk
  text, token counts, and timing breakdown.
- **Source citations** under every answer, linking back to the specific
  document and chunk used.
- **Embedding model comparison** — three models are benchmarked
  (BGE-M3, multilingual-e5-large, MiniLM) with results saved to
  `reports/embedding_model_comparison/`.
- **Generator model comparison** — four quantized LLMs are scored with a
  RAGAS-style 5-metric LLM-judge by `scripts/11_compare_generator_models.py`,
  saved to `reports/generator_model_comparison/`.
- **Human answer evaluation** — a Streamlit rating UI
  (`scripts/14_human_answer_evaluation.py`) collects human scores, merged into
  a report (plus a separate hallucination report) by
  `scripts/15_human_answer_evaluation_report.py`.
- **Safety-gate evaluation** — the adversarial test set is replayed through
  all three gates by `scripts/16_evaluate_safety_gates.py`.
- **Data card** — `scripts/17_generate_data_card.py` rolls up the whole
  pipeline into one JSON + Markdown card.

---

## Architecture

![Khmer Insurance Law Assistant — development flow: data-build pipeline (scripts 01–08), evaluation & reporting (scripts 09–17), and the live local RAG chatbot (start_chatbot.py → app.py)](assets/Development%20Flow.png)

The diagram above reads in three stages:

- **Stage A — Data-Build Pipeline (`01`–`08`):** raw source PDFs are OCR'd,
  cleaned, merged, chunked, and embedded into the FAISS indices under
  `vector_db/`. Run once, or whenever the corpus changes.
- **Stage B — Evaluation & Reporting (`09`–`17`):** benchmarks the corpus and
  models (embedding comparison, retrieval modes, generator LLM-judge scoring,
  human evaluation, safety gates) and rolls everything into a data card,
  writing JSON + Markdown to `reports/`. This is where **BGE-M3 is selected
  as the production retriever**; it does not affect the live app.
- **Stage C — Live Chatbot (`start_chatbot.py` → `app.py`):** the fully-local
  chat-time path — input gates, BGE-M3 embedding, hybrid FAISS + BM25 / RRF
  retrieval, a score gate, context building, and streamed generation via
  `llama-server`.

`start_chatbot.py` is the entry point for actually *using* the assistant:
it launches `llama-server` with the chosen generator model, waits for it to
report healthy, patches the browser tab's title/favicon, and then launches
the Streamlit app.

The numbered scripts in `scripts/` are the **one-time (or as-needed) data
pipeline** used to go from raw source PDFs to the `vector_db/` the app
actually queries. You only need to re-run any of it if you're adding new
source documents or rebuilding the corpus.

> **Note on step `[07]` (merge chunked documents):** this step is mainly a
> convenience, not a functional requirement. Chunking (`[06]`) writes one
> JSON file *per document*, so the chunks for a single category are spread
> across many files. Merging them into one file per category
> (`data/08_merged_chunked_documents/<category>_chunks.json`) makes it easy
> to (1) eyeball every chunk in a category at a glance — far more readable
> than the `vector_db/` metadata JSON — and (2) author the per-category gold
> questions in `data/09_evaluation/`, which need to reference chunk IDs.
> The embedding step `[08]` reads these merged files, but it would produce a
> byte-identical index reading the per-document `data/07_chunked_documents/`
> instead — the merge changes nothing about the chunks themselves, only how
> they're laid out on disk.

---

## Project Structure

```
Khmer-Rag-Chatbot/
├── .env                             # GEMINI_API_KEY (only for scripts/02)
├── .gitignore
├── README.md
├── app.py                           # Streamlit chatbot
├── start_chatbot.py                 # Launches llama-server + app
├── install_models.py                # Download/install model files
├── requirements.txt
│
├── assets/
│   ├── Development Flow.png
│   ├── favicon.png
│   └── logo.png
│
├── docs/
│   ├── SCORING_CALIBRATION.md
│   └── adversarial_test_set.md
│
├── data/
│   ├── 01_source_documents/         # Raw PDFs, by category
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   │
│   ├── 02_page_images/              # PDF pages rendered to PNG
│   ├── 03_raw_pages/                # Gemini OCR output
│   ├── 04_cleaned_pages/            # Page-level cleanup
│   ├── 05_merged_documents/         # Pages stitched into documents
│   ├── 06_cleaned_documents/        # Document-level cleanup
│   ├── 07_chunked_documents/        # Per-category chunks
│   ├── 08_merged_chunked_documents/ # Merged chunk files for embedding
│   ├── 09_evaluation/               # Gold questions by category
│   ├── 10_adversarial/              # Safety/refusal test cases
│   │   ├── recommendation_questions.json
│   │   ├── pricing_questions.json
│   │   ├── personal_advice_questions.json
│   │   └── out_of_domain_questions.json
│   └── document_metadata.json
│
├── reports/
│   ├── embedding_model_comparison/
│   │   ├── embedding_model_comparison.json
│   │   └── embedding_model_comparison.md
│   ├── retrieval_mode_comparison/
│   │   ├── retrieval_mode_comparison.json
│   │   └── retrieval_mode_comparison.md
│   ├── generator_model_comparison/
│   │   ├── SeaLLMs-v3-7B-Chat-Q4_K_M/   # per-model JSONL + summaries
│   │   ├── Qwen3.5-9B-Q4_K_M/
│   │   ├── gemma-4-E4B-it-Q4_K_M/
│   │   ├── Llama-3.1-8B-Instruct-Q4_K_M/
│   │   ├── generator_model_comparison.{json,md}
│   │   └── generator_score_diagnostics.{json,md}
│   ├── human_answer_evaluation/
│   │   ├── human_answer_evaluation.db          # gitignored (local scores)
│   │   ├── human_answer_evaluation_report.{json,md}
│   │   └── hallucination_report.{json,md}
│   ├── safety_gate_evaluation/
│   │   └── safety_gate_evaluation.{json,md}
│   └── data_card/
│       └── data_card.{json,md}
│
├── models/
│   ├── generator/
│   │   ├── .cache/
│   │   ├── SeaLLMs-v3-7B-Chat-Q4_K_M.gguf
│   │   ├── Qwen3.5-9B-Q4_K_M.gguf
│   │   ├── gemma-4-E4B-it-Q4_K_M.gguf
│   │   └── Llama-3.1-8B-Instruct-Q4_K_M.gguf
│   └── retriever/
│       ├── .cache/                  # Hugging Face cache
│       ├── BAAI-BGE-M3/
│       ├── multilingual-e5-large/
│       └── MiniLM-paraphrase-multilingual-L12-v2/
│
├── scripts/
│   ├── 01_convert_pdf_to_png.py
│   ├── 02_extract_text_gemini.py
│   ├── 03_clean_pages.py
│   ├── 04_merge_documents.py
│   ├── 05_clean_documents.py
│   ├── 06_chunk_documents.py
│   ├── 07_merge_chunked_documents.py
│   ├── 08_generate_embeddings.py
│   ├── 09_compare_embedding_models.py
│   ├── 10_compare_retrieval_modes.py
│   ├── 11_compare_generator_models.py
│   ├── 12_diagnose_generator_scores.py
│   ├── 13_rescore_generator_scores.py
│   ├── 14_human_answer_evaluation.py
│   ├── 15_human_answer_evaluation_report.py
│   ├── 16_evaluate_safety_gates.py
│   └── 17_generate_data_card.py
│
└── vector_db/
    ├── BAAI-BGE-M3/                 # Production index (default)
    │   ├── khmer_insurance.index
    │   ├── khmer_insurance_metadata.json
    │   └── embedding_model_info.json
    ├── multilingual-e5-large/       # Alternative index (comparison only)
    │   ├── khmer_insurance.index
    │   ├── khmer_insurance_metadata.json
    │   └── embedding_model_info.json
    └── MiniLM-paraphrase-multilingual-L12-v2/ # Alternative index (comparison only)
        ├── khmer_insurance.index
        ├── khmer_insurance_metadata.json
        └── embedding_model_info.json
```

**What's committed vs. generated locally:**

- **`data/01_source_documents/` and `data/03_raw_pages/` are committed.**
  The raw source PDFs and the Gemini-generated OCR text are kept in
  version control deliberately — OCR costs real money per page, so once
  it's done it's checked in rather than left to be re-paid-for.
- **The intermediate pipeline stages `data/02_page_images/`,
  `04_cleaned_pages/`, `05_merged_documents/`, `06_cleaned_documents/`, and
  `07_chunked_documents/` are gitignored** (kept as `.gitkeep` placeholders).
  These are fast, fully-deterministic derivatives of what *is* committed.
- **`data/08_merged_chunked_documents/`, `09_evaluation/`, and
  `10_adversarial/` are committed** — the embedded chunk corpus, the gold
  questions, and the adversarial test set are all inputs the evaluation
  scripts need.
- **`models/` is entirely gitignored.** Run `python install_models.py`
  on a fresh clone to download models.
- **`vector_db/` is committed.** The FAISS indices for all three embedding
  models ship with the repo so the chatbot works immediately after cloning
  + installing models.
- **`reports/` are committed** as the evaluation deliverables (JSON +
  Markdown), except the locally-generated human-evaluation SQLite database
  `reports/human_answer_evaluation/*.db`, which is gitignored.

---

## Prerequisites

- **Python 3.11**
- **`llama.cpp`** with `llama-server` on your `PATH`
  (`brew install llama.cpp` on macOS, or build from source)
- **At least 16 GB RAM** — 8 GB machines are extremely slow or may fail
- **~15 GB free disk space** for the embedding models + one generator
  model (more space if you install more than one generator model)
- A **Gemini API key** — only needed for `scripts/02` to OCR new source
  documents; not required to run the chatbot
- (Optional but recommended) Apple Silicon or a CUDA GPU — `start_chatbot.py`
  launches `llama-server` with `-ngl 999` for full GPU offload

---

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/sorphea7/Khmer-Rag-Chatbot.git
cd Khmer-Rag-Chatbot

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only if rebuilding corpus from new source PDFs) Add your Gemini key
echo "GEMINI_API_KEY=your-key-here" > .env

# 5. Download models
python install_models.py
```

`install_models.py` will ask which embedding model(s) and
generator model to download:

**Embedding models** (choose one or more; all three are needed for the
embedding comparison report):

| Choice | Model | Size | Use |
|---|---|---|---|
| 1 | BAAI/bge-m3 | ~2.1 GB | Production retriever (default) |
| 2 | intfloat/multilingual-e5-large | ~1.1 GB | Comparison only |
| 3 | paraphrase-multilingual-MiniLM-L12-v2 | ~470 MB | Comparison only |

**Generator models** (choose one; more can be added later):

| Choice | Model | Size |
|---|---|---|
| 1 | SeaLLM v3 7B Q4_K_M | 4.68 GB |
| 2 | Qwen 3.5 9B Q4_K_M | 5.68 GB |
| 3 | Gemma 4 E4B Q4_K_M | 4.98 GB |
| 4 | Llama 3.1 8B Instruct Q4_K_M | 4.92 GB |

Embedding models are downloaded into `models/retriever/.cache/` (one copy,
no duplication) and generator GGUFs into `models/generator/`. Re-running the
installer safely skips anything already downloaded.

---

## Running the Chatbot

```bash
python start_chatbot.py
```

This will:
1. Ask which generator model to load.
2. Start `llama-server` on `localhost:8080` and wait for it to report
   healthy (up to 120 seconds).
3. Launch the Streamlit app on `localhost:8501` and open it in your
   browser.

The vector database in `vector_db/` is already committed — you do **not**
need to run the data pipeline to use the chatbot.

The active embedding model is controlled by `ACTIVE_EMBEDDING_MODEL_DIR`
at the top of `app.py` (default: `"BAAI-BGE-M3"`). To switch to a
different model's index, change this constant to the corresponding folder
name and restart the app — no pipeline re-run needed since all three
indices are already in `vector_db/`.

---

## Evaluation Scripts

These scripts are run once after the pipeline completes. They do not
affect the live chatbot. Each writes both JSON (machine-readable) and
Markdown (human-readable) into its folder under `reports/`.

```bash
# Embedding model comparison (BGE-M3 vs. e5-large vs. MiniLM)
# Requires all three models installed and all three indices to exist
python scripts/09_compare_embedding_models.py

# Retrieval quality (hybrid vs. dense, all categories)
python scripts/10_compare_retrieval_modes.py

# Generator model comparison (requires llama-server; LLM-judge scoring)
python scripts/11_compare_generator_models.py
# optional follow-ups: 12 diagnoses, 13 rescores the generator scores

# Human answer evaluation — Streamlit rating UI on the winning model
python scripts/14_human_answer_evaluation.py          # collect scores
python scripts/15_human_answer_evaluation_report.py   # merge into a report

# Safety-gate evaluation against the adversarial set (requires llama-server)
python scripts/16_evaluate_safety_gates.py

# Data card — rolls up every artifact above into one card
python scripts/17_generate_data_card.py
```

`scripts/15_human_answer_evaluation_report.py` also writes a **separate
hallucination report** (`reports/human_answer_evaluation/hallucination_report.{json,md}`)
driven by the human Yes/No hallucination flag — the mandatory pre-demo
review gate (Internship Proposal §11).

---

## Embedding Model Comparison — Key Finding

`scripts/09_compare_embedding_models.py` benchmarks three multilingual
embedding models on the full gold question set (25 questions × 4
categories = 100 questions total, dense-only retrieval, no BM25/RRF).
The macro-averaged @5 results across all four document categories are:

| Metric | BAAI/bge-m3 | intfloat/multilingual-e5-large | paraphrase-multilingual-MiniLM-L12-v2 |
|---|---|---|---|
| Hit Rate @5 | **0.9800** | 0.6600 | 0.0100 |
| Recall @5 | **0.9600** | 0.6400 | 0.0100 |
| MRR @5 | **0.8837** | 0.5035 | 0.0100 |
| File Hit Rate @5 | **1.0000** | 0.9600 | 0.1400 |

**Why MiniLM scores near-zero on Hit Rate, Recall, and MRR:**
MiniLM-L12-v2 (118M parameters, 384-dim embeddings) is too small and
insufficiently trained on Khmer script to produce semantically
discriminative embeddings for Khmer legal text. In practice it assigns
near-identical cosine similarity scores to every chunk in the corpus
regardless of relevance — empirically confirmed: all top-10 retrieved
chunks score between **0.9587 and 0.9706** for a typical Khmer insurance
law question, a spread of only ~0.012. This score compression makes the
ranking essentially random, so the correct chunks are almost never ranked
in the top 5.

By contrast, BGE-M3 produces well-spread scores (**0.6509–0.7718** for
the same question), reflecting genuine semantic discrimination that
enables reliable retrieval.

The non-zero File Hit Rate @5 for MiniLM (0.14 macro-avg) confirms the
model is not completely blind — it occasionally retrieves chunks from the
right *document* by chance — but chunk-level precision is zero because
random ranking within a document is not useful for RAG.

**BGE-M3 is selected as the production retriever** based on these
results. Its substantially higher Hit Rate, Recall, and MRR across all
four document categories — combined with its purpose-built multilingual
training that explicitly covers Southeast Asian languages — make it the
clear choice for Khmer insurance law retrieval.

---

## Rebuilding the Corpus (optional)

Since `data/01_source_documents/` and `data/03_raw_pages/` are already
committed, a typical rebuild starts at step 3 — the paid OCR step only
runs again when adding genuinely new source PDFs.

| Script | What it does | When needed |
|---|---|---|
| `01_convert_pdf_to_png.py` | Renders PDF pages to PNG | New source PDFs only |
| `02_extract_text_gemini.py` | OCRs each PNG via Gemini (paid API) | New source PDFs only |
| `03_clean_pages.py` | Strips OCR artifacts per category | Any rebuild |
| `04_merge_documents.py` | Stitches cleaned pages into full documents | Any rebuild |
| `05_clean_documents.py` | Document-level cleanup, abbreviation expansion | Any rebuild |
| `06_chunk_documents.py` | Legal-unit / heading-hierarchy chunking | Any rebuild |
| `07_merge_chunked_documents.py` | Merges per-category chunks into unified files | Any rebuild |
| `08_generate_embeddings.py` | Builds FAISS indices for all installed models | Any rebuild |
| `10_compare_retrieval_modes.py` | Measures retrieval quality on gold questions | Any rebuild |

```bash
# Typical rebuild (starting from committed OCR output):
python scripts/03_clean_pages.py
python scripts/04_merge_documents.py
python scripts/05_clean_documents.py
python scripts/06_chunk_documents.py
python scripts/07_merge_chunked_documents.py

# Delete existing indices first -- 08 skips models that already have one
rm -rf vector_db/BAAI-BGE-M3 vector_db/multilingual-e5-large vector_db/MiniLM-paraphrase-multilingual-L12-v2
python scripts/08_generate_embeddings.py

python scripts/10_compare_retrieval_modes.py
```

**Note on `02_extract_text_gemini.py`:** OCR runs page-by-page against a
paid API. Any failed page is skipped immediately rather than retried —
re-running the script later will pick up exactly the gaps that failed.

---

## Configuration

The main tunable constants live at the top of `app.py`:

| Constant | Default | What it controls |
|---|---|---|
| `ACTIVE_EMBEDDING_MODEL_DIR` | `"BAAI-BGE-M3"` | Which vector_db subfolder the app loads at startup |
| `TOP_K` | 10 | How many chunks FAISS retrieves; also shown in the diagnostics page |
| `CONTEXT_K` | 5 | How many of those actually go into the LLM prompt (<= TOP_K) |
| `MAX_CHUNKS_PER_DOC` | 1 | Caps near-duplicate chunks from the same document in context |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant (Cormack et al. 2009) |
| `MIN_SCORE` | 0.50 | Below this, answers are flagged "low confidence" in the UI |
| `OUT_OF_DOMAIN_THRESHOLD` | 0.49 | Below this, generation is skipped and the user gets an instant refusal |
| `MIN_QUESTION_LENGTH` | 8 | Questions shorter than this are rejected without embedding |

`OUT_OF_DOMAIN_THRESHOLD` was tuned empirically against real chit-chat
and off-topic test queries — if you change the corpus, re-validate it
against `data/10_adversarial/` before trusting it.

`start_chatbot.py` configures `llama-server` with:
- `--ctx-size 16384` — must exceed your typical prompt size; raise this
  if you increase `CONTEXT_K` significantly
- `--flash-attn on` and `--cache-type-k/v q8_0` — performance tuning for
  the 8–15k token prompts this app typically sends

---

## Gold Questions Schema

`data/09_evaluation/*/gold_questions.json` uses the following schema:

```json
{
  "question": "...",
  "expected_chunks": ["chunk_id_1", "chunk_id_2"],
  "expected_answer": "..."
}
```

- `expected_chunks` is used by `09_compare_embedding_models.py` and
  `10_compare_retrieval_modes.py` to measure whether the retriever surfaces
  the right source material.
- `expected_answer` is used by `11_compare_generator_models.py` to score the
  generated answers — questions without this field are skipped.

---

## Troubleshooting

- **`llama-server not found`** — install `llama.cpp` and make sure
  `llama-server` is on your `PATH`.
- **First response is much slower than later ones** — expected.
  `llama-server`'s prompt cache warms up on shared prefixes; subsequent
  questions in the same session are typically much faster.
- **VSCode shows "Import could not be resolved" for everything despite
  the app running fine in the terminal** — Pylance is analyzing the wrong
  Python interpreter. `Cmd/Ctrl+Shift+P` → `Python: Select Interpreter`
  → choose the one under `.venv/bin/python`, then reload the window.
- **A brief "Streamlit" tab title flash on first launch only** — a
  Chromium browsing-history quirk, not something the app controls.
- **`No precomputed index` error when running `09_compare_embedding_models.py`**
  — run `python scripts/08_generate_embeddings.py` first to build the
  indices for all three models, then retry.
- **Changing the active embedding model** — edit `ACTIVE_EMBEDDING_MODEL_DIR`
  in `app.py` to one of `"BAAI-BGE-M3"`, `"multilingual-e5-large"`, or
  `"MiniLM-paraphrase-multilingual-L12-v2"`, then restart the app.
  The corresponding index must already exist in `vector_db/`.
- **MiniLM scores 0.0000 on Hit Rate / Recall / MRR in the embedding
  comparison** — this is not a bug. See
  [Embedding Model Comparison — Key Finding](#embedding-model-comparison--key-finding)
  above for the full explanation. In short: MiniLM cannot discriminate
  between Khmer legal texts and produces near-identical similarity scores
  for all chunks (~0.958–0.971), making its ranking essentially random.