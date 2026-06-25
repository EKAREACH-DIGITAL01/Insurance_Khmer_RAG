# Data Card — Khmer Insurance RAG Chatbot

Generated: 2026-06-25 21:25:36

Retrieval-augmented question answering over Cambodian insurance laws, sub-decrees, prakas and guidance, in Khmer. The assistant answers strictly from the retrieved source documents and refuses out-of-scope, advisory, pricing and recommendation requests.

## 1. Source Corpus

- **Total documents:** 49
- **Language:** Khmer (km)
- **Domain:** Cambodian insurance law & regulation
- **Release date range:** 2021-01-08 → 2026-02-25

| Document Type            |   Documents | Earliest   | Latest     |
|--------------------------|-------------|------------|------------|
| Guidance (គោលការណ៍ណែនាំ) |          16 | 2022-01-01 | 2026-01-14 |
| Laws (ច្បាប់)            |           4 | 2021-11-02 | 2022-10-30 |
| Prakas (ប្រកាស)          |          25 | 2021-11-02 | 2026-02-25 |
| Sub-decrees (អនុក្រឹត្យ) |           4 | 2021-01-08 | 2025-01-10 |

## 2. Chunked Knowledge Base

- **Total chunks:** 1435
- **Avg chunk size:** 511.6 chars
- **Chunking method(s):** guidance_heading, legal_unit

| Category    |   Docs |   Chunks |   Avg Size | Size Range   |
|-------------|--------|----------|------------|--------------|
| guidance    |     16 |      146 |      741.9 | 53–3409      |
| laws        |      4 |      255 |      473.1 | 64–4747      |
| prakas      |     25 |      783 |      484   | 62–4470      |
| sub_decrees |      4 |      251 |      502.5 | 61–3068      |
| **TOTAL**   |        |     1435 |      511.6 |              |

## 3. Vector Index

- **Active embedding model:** BAAI/bge-m3
- **Embedding dimension:** 1024
- **Indexed chunks:** 1435

| Model                                 |   Dim |   Chunks |
|---------------------------------------|-------|----------|
| BAAI/bge-m3                           |  1024 |     1435 |
| intfloat/multilingual-e5-large        |  1024 |     1435 |
| paraphrase-multilingual-MiniLM-L12-v2 |   384 |     1435 |

## 4. Evaluation Sets

**Gold question set** — 100 questions

| Category    |   Questions |
|-------------|-------------|
| guidance    |          25 |
| laws        |          25 |
| prakas      |          25 |
| sub_decrees |          25 |

**Adversarial / safety set** — 43 questions

| Category        |   Total |   Must Refuse |   Must Answer |
|-----------------|---------|---------------|---------------|
| out_of_domain   |      14 |            11 |             3 |
| personal_advice |      10 |             7 |             3 |
| pricing         |       8 |             5 |             3 |
| recommendation  |      11 |             7 |             4 |

## 5. Model Selection

- **Embedding model:** `BAAI/bge-m3` (Hit@5=0.98, Recall@5=0.96, MRR@5=0.8837); compared 3 models
- **Retrieval mode:** `hybrid`
- **Generator model:** `SeaLLM v3 7B` (combined judge score 4.476/5); compared 4 models

Retrieval mode comparison (macro-avg @5):

| Metric           |   Dense |   Hybrid |      Δ |
|------------------|---------|----------|--------|
| Hit Rate @5      |  0.98   |   1      | 0.02   |
| Recall @5        |  0.96   |   0.995  | 0.035  |
| MRR @5           |  0.8837 |   0.8882 | 0.0045 |
| File Hit Rate @5 |  1      |   1      | 0      |

## 6. Quality Evaluation

### Automated LLM-judge (SeaLLM v3 7B, 100 questions)

| Metric             |   Avg (1-5) |
|--------------------|-------------|
| Context Relevance  |       4.48  |
| Faithfulness       |       4.52  |
| Answer Correctness |       4.64  |
| Answer Relevance   |       4.01  |
| Completeness       |       4.59  |
| **Combined**       |       4.476 |

### Human evaluation

- **Evaluators:** 1 | **Responses:** 25 | **Questions rated:** 25
- **Combined avg:** 4.07/5
- **Hallucinations flagged:** 0

| Metric             |   Human Avg (1-5) |
|--------------------|-------------------|
| Context Relevance  |              3.8  |
| Faithfulness       |              4.08 |
| Answer Relevance   |              4.2  |
| Answer Correctness |              4.2  |

## 7. Limitations

- Coverage is limited to the documents listed in the source corpus; newer or amended regulations may be missing.
- Answers are only as accurate as the underlying documents and the retrieval step; low-similarity queries are refused by design.
- Khmer OCR/extraction and chunking can introduce minor text errors.
- Not legal advice -- users are directed to a licensed agent or the Insurance Regulator of Cambodia (IRC) for binding guidance.

