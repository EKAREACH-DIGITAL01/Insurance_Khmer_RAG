# Khmer Insurance Assistant

A Khmer RAG chatbot for Cambodian insurance information, built on the student's
49 legal PDFs and an expanded Insurance Regulator of Cambodia website corpus.
It uses BGE-M3 embeddings, FAISS and BM25 retrieval, and a local llama-server or
an OpenAI-compatible generator endpoint.

## Run Locally

Use Python 3.12 and the existing project environment:

```sh
source .venv/bin/activate
python start_chatbot.py
```

The launcher checks the local models and index, starts llama-server, and opens
the Streamlit application. With a generator already running:

```sh
streamlit run app.py
```

For a fresh checkout, create a virtual environment and install
`requirements.txt`. Run `python install_models.py` to install the required
local models. The active index lives in `vector_db/BAAI-BGE-M3/`; the embedding
model must match the model recorded in `embedding_model_info.json`.

## Project Layout

```text
app.py                    Streamlit entry point
start_chatbot.py          Local model/server launcher
install_models.py        Model installation
khmer_rag/
  bootstrap.py           Environment and logging setup
  config.py              Runtime settings and project paths
  language.py            Khmer greetings and tokenization
  policy.py              Scope checks
  text.py                Source and answer text cleanup
  answers.py             Source-specific website answer rules
  provisions.py          Verbatim legal provision lookup
  pipeline.py            Retrieval-to-answer orchestration
  storage.py             SQLite conversations and feedback
  categories.py          Shared IRC category definitions
  sources.py             Document IDs and source labels
  retrieval/             Search, ranking, and context selection
  generation/            Prompts, token limits, and streaming client
  ui/                    Streamlit pages, session state, and resource caches
assets/ui/               CSS and browser scripts
scripts/                 Numbered data and evaluation commands
tests/                   Offline regression tests
data/                    Source documents and processing outputs
vector_db/               Generated retrieval indexes
reports/                 Recorded evaluation results
docs/                    Architecture, research, and operations notes
```

`history.py` and `scripts/irc_website_categories.py` preserve the old import
paths. New code imports `khmer_rag.storage` and `khmer_rag.categories` directly.
The existing data folders, script names, SQLite schema, and model paths remain
compatible with the student's workflow.

## Configuration

Settings live in `khmer_rag/config.py`. The app loads `.env` at startup; it is
not committed. Environment variables take precedence over `.env` values.

| Setting | Purpose |
| --- | --- |
| `GENERATOR_BASE_URL` | OpenAI-compatible endpoint; local default is `http://localhost:8080/v1` |
| `GENERATOR_MODEL` | Model ID served by that endpoint |
| `GENERATOR_API_KEY` | Hosted generator credential |
| `EMBEDDING_LOCAL_FILES_ONLY` | Require embeddings to load from the local cache |
| `HF_HOME` | Optional model cache location |
| `LOG_LEVEL` | Console logging level, normally `INFO` |

Hosted deployments should set the model ID explicitly. Gemini credentials used
for PDF extraction are separate from the chat generator configuration. The
runtime also accepts the existing Gemini and OpenAI environment aliases.

## Development Checks

The fast tests use fake generators and temporary databases. They do not start
the chatbot server, call external APIs, download models, or alter your corpus.

```sh
python -m pip install -r requirements-test.txt -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

Tests cover module boundaries, Khmer greetings, exact provision lookup,
retrieval ranking, queue events, streaming cancellation, provider options,
feedback persistence, and Streamlit rendering. CI runs the same fast checks.

For a model-backed corpus check, run this separately with BGE-M3 installed:

```sh
python scripts/19_rag_smoke_tests.py
```

See [architecture and debugging](docs/architecture.md) for the request flow,
module ownership, troubleshooting, and known limitations. See the
[script guide](scripts/README.md) for dataset and evaluation commands.

## Evaluation Status

The original experiments are preserved in `reports/` and the
[experiment guide](docs/experiment-guide.md). Those scores describe their
recorded corpus, models, and configuration; they are not new quality measurements
for this refactor. Evaluation commands now import shared runtime helpers, but
their model comparisons and gold-answer token budgets remain experiment-specific.

Code restructuring does not certify factual accuracy. Corpus coverage, OCR
quality, year-specific statistics, and generated answers still require evaluation
against verified source documents.
