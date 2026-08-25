# RAG Project Audit And Handoff

This note applies the senior RAG-audit checklist from
`rag-chatbot-agent-prompt.md` to the current Khmer Insurance Assistant codebase.
The attached prompt is project guidance, not runtime instructions for the
chatbot.

## Current Stack

- Language/UI: Python 3.11 + Streamlit.
- Generator: local `llama.cpp` server, with SeaLLM v3 7B as the production
  model.
- Retriever: BGE-M3 sentence embeddings, committed FAISS vector indexes, and
  in-memory BM25 lexical search.
- Fusion: Reciprocal Rank Fusion over FAISS + BM25 candidate lists.
- Storage: JSON chunk metadata under `vector_db/`, source/derived corpus files
  under `data/`, and SQLite conversation/feedback history in `chat_history.db`.
- Chat-time privacy: no external API call is made when the chatbot answers.
  Gemini is only used in the offline OCR pipeline.

## Data Flow

1. Original source PDFs are stored in `data/01_source_documents/`.
2. IRC website pages and linked documents are captured by
   `scripts/00_scrape_irc_website.py`.
3. Original PDF pages are OCRed with `scripts/02_extract_text_gemini.py`;
   website PDFs are OCRed with `scripts/00c_extract_irc_pdf_text_gemini.py`.
4. Website-derived documents are classified into IRC section folders with
   `scripts/20_reclassify_website_categories.py` when using an existing
   crawl/OCR cache.
5. Text is cleaned, merged, chunked, and merged for embedding through
   `scripts/03` to `scripts/07`.
6. `scripts/08_generate_embeddings.py` creates per-model FAISS indexes and
   metadata under `vector_db/`.
7. `app.py` loads the BGE-M3 index, builds BM25 in memory, retrieves with hybrid
   search, gates weak/off-topic questions, builds prompt context, and streams
   the answer from the local LLM.
8. Each answer stores sources, diagnostics, timing, and optional user feedback
   through `history.py`.

## What Is Strong

- The app is local at chat time, which is good for demo reliability and avoids
  live API cost.
- BGE-M3 was benchmarked against alternatives and is a sensible Khmer-first
  production retriever.
- Hybrid retrieval is already implemented and is better suited to Khmer legal
  text than dense-only retrieval.
- The app has practical guardrails: Khmer-only input, short-input rejection,
  scope/recommendation refusal, low-score refusal, and low-confidence prompt
  handling.
- The UI includes source citations, per-answer diagnostics, copy/like/dislike
  actions, persisted chat history, and feedback export.
- Website-specific high-risk demo facts now have deterministic source-backed
  answer extractors, reducing local-model drift for management structure,
  gross premium, publication catalog, management profile, and policy-count
  questions.
- IRC website-derived data is no longer a single generic `website` category:
  it is split into `home`, `statistics`, `publication`, `public_service`,
  `insurance_institutions`, `knowledge`, `news`, `events`, `social_programs`,
  `announcement`, and `law_regulation` for section-specific chunking.

## Current Gaps And Risks

- `app.py` is still large. It is demo-ready, but a future production version
  should split retrieval, prompting, UI, diagnostics, and curated extractors
  into separate modules.
- Runtime constants are duplicated in evaluation scripts. This pass keeps the
  obvious values in parity, but a shared config module would be cleaner.
- The formal gold/evaluation set still focuses mainly on the original 49 PDFs.
  Website coverage now exists in the vector DB, but it needs a dedicated
  website gold set for repeatable quality scoring.
- There is no fast unit-test suite. The new smoke test fills the most important
  demo gap by verifying corpus/index shape and critical IRC website retrieval.
- No cross-encoder reranker is used. That is acceptable for the local demo
  because BGE-M3 + BM25 + curated extractors gives good latency and reliability;
  a reranker can be evaluated later if answer quality still depends on weak
  top-k ordering.
- The app is designed for local use. If exposed on a public network, add
  authentication, rate limiting, and deployment-specific logging before use.

## Target Architecture For This Project

- Keep the current local-first architecture for the demo: Streamlit,
  llama.cpp, BGE-M3, FAISS, BM25, and SQLite.
- Keep FAISS rather than moving to a hosted vector DB. The corpus is small
  enough for local search, and a paid vector service would add operational
  risk without improving the boss-demo goal.
- Keep Gemini OCR offline only. Do not call Gemini at chat time.
- Treat `scripts/00`, `00c`, and `01`-`08` as the offline indexing pipeline.
  Rebuild only when source text or chunking changes.
- Add website-specific gold questions next, especially for statistics,
  publications, management profiles, and annual reports.
- Use `scripts/19_rag_smoke_tests.py` after any corpus, chunking, or retrieval
  change before demoing the chatbot.

## Verification Command

Run this after installing dependencies and models, without starting the chatbot:

```bash
python scripts/19_rag_smoke_tests.py
```

The smoke test checks that the committed BGE-M3 index matches its metadata,
contains the expanded section-aware IRC website corpus, avoids the known
generated typo `ធានរបង`, retrieves the exact top-k sources for the current demo
questions, and verifies the curated policy-count source used by the fallback
extractor.
