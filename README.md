# Khmer Insurance Law Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions about
Cambodian insurance law in Khmer, grounded entirely in a curated corpus of
laws, sub-decrees, prakas (ministerial regulations), and regulatory
guidance documents. Built with a local embedding model (BGE-M3), a local
FAISS vector index, and a local quantized LLM (SeaLLMs-v3-7B) served via
`llama.cpp` — no external API calls happen at chat time.

---

## Features

- **Khmer-first RAG** over four document categories: `laws`, `prakas`,
  `sub_decrees`, and `guidance`.
- **Fully local inference** — embedding, retrieval, and generation all run
  on your machine via `llama-server`; only the one-time data-build pipeline
  uses an external API (Gemini, for OCR).
- **Layered domain/relevance gating** before generation ever runs:
  - Non-Khmer input is rejected instantly (no LLM call).
  - Degenerate/too-short input is rejected instantly.
  - Low-similarity retrieval is rejected before an expensive generation
    cycle starts, with the threshold calibrated against real test queries
    (see [Configuration](#configuration) below).
  - Borderline-confidence retrieval still generates, but with an extra
    prompt instruction asking the model to double-check topical relevance
    before answering, and a visible "low confidence" warning in the UI.
- **Live progress ladder** showing each pipeline phase (embedding, FAISS
  search, context building, prompt construction, generation) with timing,
  plus a **Stop** button that can interrupt generation mid-stream — even
  surviving navigation to the diagnostics page and back.
- **Per-answer diagnostics page** — every answer can be inspected for its
  retrieval scores, full retrieved chunk text, token counts, and timing
  breakdown.
- **Source citations** under every answer, linking back to the specific
  document and chunk used.

---

## Architecture

```
PDF source documents
        |
        v
 [01] Convert pages to PNG images
        |
        v
 [02] OCR each page with Gemini  ---------------> raw page text (JSON)
        |
        v
 [03] Clean OCR artifacts per category (laws / prakas / sub_decrees / guidance)
        |
        v
 [04] Merge pages back into full documents
        |
        v
 [05] Document-level cleaning (headers, signatures, appendices, abbreviations)
        |
        v
 [06] Chunk documents (legal-unit chunking for laws/prakas/sub_decrees,
      heading-hierarchy chunking for guidance)
        |
        v
 [07] Embed chunks with BGE-M3 -> FAISS index + metadata
        |
        v
 [08] Evaluate retrieval quality against a hand-labeled gold question set
        |
        v
   vector_db/ (FAISS index + metadata, used by the live app)
        |
        v
+------------------------------------------------------------+
|                      app.py (Streamlit)                    |
|                                                              |
|  question -> Khmer-script check -> length check              |
|           -> BGE-M3 embed -> FAISS search -> score gate       |
|           -> build context (capped, deduped per document)     |
|           -> prompt -> llama-server (SeaLLMs-v3-7B, GGUF)     |
|           -> streamed answer + sources + diagnostics          |
+------------------------------------------------------------+
```

`start_chatbot.py` is the entry point for actually *using* the assistant:
it launches `llama-server` with the chosen quantization, waits for it to
report healthy, patches the browser tab's title/favicon, and then launches
the Streamlit app.

The numbered scripts in `scripts/` are the **one-time (or as-needed) data
pipeline** used to go from raw source PDFs to the `vector_db/` the app
actually queries. You only need to re-run any of it if you're adding new
source documents or rebuilding the corpus — the app itself never touches
`scripts/`.

---

## Project Structure

```
Khmer-Rag-Chatbot/
├── .env                           # GEMINI_API_KEY (only needed for scripts/02)
├── .gitignore
├── README.md
├── app.py                         # The Streamlit chatbot
├── start_chatbot.py               # Entry point: launches llama-server + the app
├── requirements.txt
├── assets/
│   └── favicon.png
├── data/
│   ├── 01_source_documents/       # Raw input PDFs, by category
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 02_page_images/            # PDF pages rendered to PNG (scripts/01)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 03_raw_pages/              # Gemini OCR output, one JSON/page (scripts/02)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 04_cleaned_pages/          # OCR-artifact cleanup, per page (scripts/03)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 05_merged_documents/       # Pages stitched into full documents (scripts/04)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 06_cleaned_documents/      # Document-level cleanup (scripts/05)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 07_chunked_documents/      # Final chunks ready for embedding (scripts/06)
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   ├── 08_evaluation/             # Hand-labeled gold questions per category
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   └── document_metadata.json     # document_id -> name/type/release_date lookup
├── models/
│   ├── .hf_cache/                 # Hugging Face download cache (created by scripts/09)
│   ├── generator/
│   │   ├── SeaLLMs-v3-7B-Chat-Q4_K_M.gguf
│   └── retriever/
│       └── BAAI-BGE-M3/
├── scripts/
│   ├── 01_convert_pdf_to_png.py
│   ├── 02_extract_text_gemini.py
│   ├── 03_clean_pages.py
│   ├── 04_merge_documents.py
│   ├── 05_clean_documents.py
│   ├── 06_chunk_documents.py
│   ├── 07_generate_embeddings.py
│   ├── 08_evaluate_retrieval.py
│   └── 09_install_model.py
└── vector_db/
    ├── khmer_insurance.index       # FAISS index (built by scripts/07)
    └── khmer_insurance_metadata.json
```

**What's actually in the repo vs. generated locally:**

- **`data/01_source_documents/` and `data/03_raw_pages/` are committed.**
  The raw source PDFs and the (paid, Gemini-generated) OCR text are kept in
  version control deliberately — OCR costs real money per page, so once
  it's done it's checked in rather than left to be silently regenerated
  (and re-paid-for) by anyone who clones the repo.
- **`data/02_page_images/`, `04_cleaned_pages/`, `05_merged_documents/`,
  `06_cleaned_documents/`, and `07_chunked_documents/` are gitignored**
  (each keeps only a `.gitkeep` placeholder so the folder itself still
  exists after cloning). These are cheap, fast, fully-deterministic
  derivatives of what *is* committed, so there's no value in tracking
  them — running the corresponding script regenerates them in seconds to
  minutes, not dollars.
- **`models/` is entirely gitignored.** A fresh clone has no models at
  all until you run `scripts/09_install_model.py`.
- **`vector_db/` is committed.** The FAISS index and metadata ship with
  the repo, so the chatbot works immediately after cloning + installing
  models — no pipeline run required unless you're changing the corpus.
- **`.env` and `.venv/` are gitignored**, as usual — create your own
  `.env` locally (see [Setup](#setup)).

One practical consequence: since `01_source_documents/` and
`03_raw_pages/` already come with the repo, a typical contributor
rebuilding the corpus **doesn't need to run `scripts/01` or `scripts/02`
at all** (PDF→PNG conversion and the paid OCR step) — only when adding
genuinely new source PDFs that haven't been OCR'd yet. Otherwise, start
straight from `scripts/03` onward using the already-committed OCR output.
See [Rebuilding the Corpus](#rebuilding-the-corpus-optional) below.

---

## Prerequisites

- **Python 3.11** (the project venv is built against this version)
- **`llama.cpp`** with `llama-server` available on your `PATH`
  (e.g. `brew install llama.cpp` on macOS, or build from source)
- **At least 16 GB RAM** (required). Systems with 8 GB RAM are extremely slow or may fail to run reliably.
- **~10-15 GB free disk space** for the embedding model + one or more
  generator quantizations
- A **Gemini API key** — only required if you're running `scripts/02` to
  OCR new source documents; not needed to run the chatbot itself
- (Optional but recommended) A GPU/Metal-capable machine — `start_chatbot.py`
  launches `llama-server` with `-ngl 999` (full GPU offload). On Apple
  Silicon this uses Metal automatically; on CPU-only machines it'll still
  work, just slower

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

# 4. (Only if rebuilding the corpus from source PDFs) Add your Gemini key
echo "GEMINI_API_KEY=your-key-here" > .env

# 5. Download the embedding model and a generator quantization
python scripts/09_install_model.py
```

`scripts/09_install_model.py` will prompt you to choose a generator
quantization:

| Choice | Quantization | Size | Notes |
|---|---|---|---|
| 1 | Q4_K_M | ~4.8 GB | Fastest, recommended starting point |
| 2 | Q5_K_M | ~5.5 GB | Best speed/quality balance |
| 3 | Q6_K   | ~6.3 GB | Higher quality |
| 4 | Q8_0   | ~8.1 GB | Highest quality, slowest |

You can run it multiple times to download more than one quantization —
`start_chatbot.py` lets you pick which one to load at launch time.

---

## Running the Chatbot

```bash
python start_chatbot.py
```

This will:
1. Ask which generator quantization to load.
2. Start `llama-server` on `localhost:8080` with that model and wait for
   it to report healthy (up to 120s).
3. Launch the Streamlit app on `localhost:8501` and open it in your
   browser.

The vector database in `vector_db/` is already built and committed — you
do **not** need to run the data pipeline below to use the chatbot. The
pipeline is only for rebuilding the corpus from scratch or adding new
source documents.

---

## Rebuilding the Corpus (optional)

Each script picks up where the previous one left off, reading from
`data/0N_.../` and writing to `data/0(N+1)_.../`. All of them are
re-runnable and skip work that's already done, so you can stop and resume
freely.

**Since `data/01_source_documents/` and `data/03_raw_pages/` are already
committed to the repo (see [Project Structure](#project-structure)), a
typical rebuild — re-tuning the cleaning/chunking logic, adjusting
abbreviation expansion, regenerating embeddings, etc. — only needs steps
3 onward.** Steps 1–2 (PDF→PNG conversion and the paid Gemini OCR call)
are only needed when adding genuinely **new** source PDFs that don't have
OCR output yet.

| Step | Script | What it does | Needed for... |
|---|---|---|---|
| 1 | `01_convert_pdf_to_png.py` | Renders each source PDF page to a 400 DPI PNG | New source PDFs only |
| 2 | `02_extract_text_gemini.py` | OCRs each PNG via Gemini. Runs one page at a time against a paid API — see note below | New source PDFs only |
| 3 | `03_clean_pages.py` | Strips OCR artifacts (page numbers, footers, contact info) per category | Any rebuild |
| 4 | `04_merge_documents.py` | Stitches cleaned pages back into one full document per `document_id` | Any rebuild |
| 5 | `05_clean_documents.py` | Document-level cleanup: strips headers/signatures/appendices, normalizes headings, expands abbreviations | Any rebuild |
| 6 | `06_chunk_documents.py` | Chunks each document — legal-unit chunking (មាត្រា/ប្រការ) for laws/prakas/sub_decrees, heading-hierarchy chunking for guidance | Any rebuild |
| 7 | `07_generate_embeddings.py` | Embeds all chunks with BGE-M3 and builds the FAISS index + metadata in `vector_db/` | Any rebuild |
| 8 | `08_evaluate_retrieval.py` | Reports Hit Rate / Recall / MRR / File-Hit-Rate @K against `data/08_evaluation/*/gold_questions.json` | Any rebuild |

```bash
# Only when adding new source PDFs that haven't been OCR'd yet:
python scripts/01_convert_pdf_to_png.py
python scripts/02_extract_text_gemini.py

# A typical rebuild starts here, using the already-committed OCR output:
python scripts/03_clean_pages.py
python scripts/04_merge_documents.py
python scripts/05_clean_documents.py
python scripts/06_chunk_documents.py
python scripts/07_generate_embeddings.py   # refuses to run if vector_db/ already exists -- delete it first to regenerate

python scripts/08_evaluate_retrieval.py
```

**Note on `02_extract_text_gemini.py`:** OCR runs page-by-page against a
paid API and is *not* designed to guarantee every page succeeds in a
single pass. Each page gets one attempt; any failure (timeout or
otherwise) skips that page immediately rather than retrying in a loop, and
since no output is written for a failed page, simply re-running the script
later will pick up exactly the gaps that failed before.

---

## Configuration

The main tunable constants live at the top of `app.py`:

| Constant | Default | What it controls |
|---|---|---|
| `TOP_K` | 10 | How many chunks FAISS retrieves and shows in diagnostics |
| `CONTEXT_K` | 5 | How many of those actually go into the LLM prompt |
| `MAX_CHUNKS_PER_DOC` | 1 | Caps near-duplicate chunks from the same document in context, favoring spread across documents |
| `MIN_SCORE` | 0.50 | Below this, answers are flagged "low confidence" in the UI |
| `OUT_OF_DOMAIN_THRESHOLD` | 0.49 | Below this, generation is skipped entirely and the user gets an instant "not insurance-law-related" response |
| `MIN_QUESTION_LENGTH` | 8 | Questions shorter than this are rejected without even running the embedding model |

`OUT_OF_DOMAIN_THRESHOLD` in particular was tuned empirically against real
chit-chat and off-topic test questions (see the comment above its
definition in `app.py` for the calibration log) — if you change the
corpus, re-validate it against your own out-of-domain test questions
before trusting it.

`start_chatbot.py` configures `llama-server` with:
- `--ctx-size 16384` — must exceed your typical prompt size; raise this if
  you increase `CONTEXT_K` and see prefill slow down disproportionately
  (a sign of context-window overflow forcing context-shifting)
- `--flash-attn on` and `--cache-type-k/v q8_0` — performance/memory
  tuning for the ~5-16k token prompts this app typically sends

---

## Troubleshooting

- **`llama-server not found`** — install `llama.cpp` and make sure
  `llama-server` is on your `PATH`.
- **First response is much slower than later ones** — this is expected.
  `llama-server`'s prompt cache needs to "warm up" on shared prompt
  prefixes; subsequent questions in the same session are typically much
  faster than the first.
- **VSCode shows "Import could not be resolved" for everything despite
  the app running fine in the terminal** — Pylance is analyzing the wrong
  Python interpreter. Cmd/Ctrl+Shift+P -> `Python: Select Interpreter` ->
  choose the one under `.venv/bin/python`, then reload the window.
- **A brief "Streamlit" tab title flash on first launch only (never on
  reload)** — this is a Chromium tab-title-from-browsing-history quirk,
  not something the app controls; it disappears immediately on any
  subsequent reload of the same tab.