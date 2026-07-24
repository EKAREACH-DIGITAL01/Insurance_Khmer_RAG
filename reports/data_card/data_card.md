# Data Card — Khmer Insurance RAG Chatbot

Generated: 2026-07-24 18:00:59

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
- **Avg chunk size:** 512.7 chars
- **Chunking method(s):** guidance_heading, legal_unit

| Category    |   Docs |   Chunks |   Avg Size | Size Range   |
|-------------|--------|----------|------------|--------------|
| guidance    |     16 |      146 |      743.1 | 53–3409      |
| laws        |      4 |      255 |      473.1 | 64–4747      |
| prakas      |     25 |      783 |      485.6 | 62–4470      |
| sub_decrees |      4 |      251 |      503.8 | 61–3068      |
| **TOTAL**   |        |     1435 |      512.7 |              |

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

**Gold question set** — 400 questions

| Category    |   Questions |
|-------------|-------------|
| guidance    |         100 |
| laws        |          75 |
| prakas      |         150 |
| sub_decrees |          75 |

*Human validation of this set* — the gold questions were themselves rated by people, so the yardstick is checked and not merely assumed:

| Metric                         |   Value |
|--------------------------------|---------|
| Evaluators                     |  11     |
| Ratings collected              | 550     |
| Questions rated                | 400     |
| Questions rated by 2+ people   | 150     |
| Mean rating (1-5)              |   4.009 |
| Questions flagged for curation |  62     |

**Adversarial / safety set** — 50 questions

| Category        |   Total |   Must Refuse |   Must Answer |
|-----------------|---------|---------------|---------------|
| out_of_domain   |      16 |            13 |             3 |
| personal_advice |      12 |             9 |             3 |
| pricing         |      10 |             7 |             3 |
| recommendation  |      12 |             9 |             3 |

## 5. Model Selection

- **Embedding model:** `BAAI/bge-m3` (Hit@5=0.9658333333333333, Recall@5=0.9658333333333333, MRR@5=0.8684); compared 3 models
- **Retrieval mode:** `hybrid`
- **Generator model:** `SeaLLMs-v3-7B-Chat-Q4_K_M` (combined judge score 4.8815/5); compared 4 models

Retrieval mode comparison (macro-avg @5):

| Metric           |   Dense |   Hybrid |      Δ |
|------------------|---------|----------|--------|
| Hit Rate @5      |  0.9658 |   0.9825 | 0.0167 |
| Recall @5        |  0.9658 |   0.9825 | 0.0167 |
| MRR @5           |  0.8684 |   0.8952 | 0.0267 |
| File Hit Rate @5 |  0.9825 |   0.9917 | 0.0092 |

## 6. Quality Evaluation

### Automated LLM-judge (SeaLLMs-v3-7B-Chat-Q4_K_M, 100 questions)

| Metric             |   Avg (1-5) |
|--------------------|-------------|
| Context Relevance  |      4.88   |
| Faithfulness       |      4.86   |
| Answer Relevance   |      4.97   |
| Answer Correctness |      4.84   |
| Completeness       |      4.9    |
| **Combined**       |      4.8815 |

### Safety gates

- **Model:** SeaLLM v3 7B
- **Accuracy:** 50/50 (100%)
- **Confusion:** TR=38, TA=12, FA=0, FR=0

## 7. Limitations

- The assistant only knows the documents in this corpus, which is a fixed snapshot -- regulations issued or amended after the listed date range are absent, and it cannot say so when asked.
- Answers are only as accurate as the underlying documents and the retrieval step; low-similarity queries are refused by design.
- Khmer OCR/extraction and chunking can introduce minor text errors.
- Generated answers are scored by an LLM judge, not by humans. The judge was measured for self-consistency (85-96% exact agreement on a repeat pass) but was never validated against human judgement of the answers, so the generator ranking is a defensible automated comparison rather than a measured answer-quality result.
- Hallucination counts are an upper bound: they are judged against the gold answer, so a detail correctly taken from the source document but omitted from the gold answer is still counted. Verbose models are the most affected.
- Not legal advice -- users are directed to a licensed agent or the Insurance Regulator of Cambodia (IRC) for binding guidance.

