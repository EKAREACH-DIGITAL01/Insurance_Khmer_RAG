# Generator Score Diagnostics

Evidence file for the calibration of `scripts/13_compare_generator_models.py` scoring thresholds. See `docs/SCORING_CALIBRATION.md` for the full interpretation.

> **Answer Relevance is informational only** — it is no longer part of the weighted combined score (its answer-vs-question cosine is structurally capped and penalised even gold-identical answers). Its distribution is still shown below for reference. The weighted metrics are Context Relevance, Faithfulness, Answer Correctness, and Completeness.

> **Blank answer policy**: blank answers score **1** on Answer Correctness (not excluded). `similarity = 0.0` is stored in `detail.answer_correctness_similarity` for traceability.

## Blank Answers

| Model | Blank / Total | By Category |
|-------|--------------|-------------|
| SeaLLM v3 7B | 0/100 (0.0%) | {} |
| Qwen 3.5 9B | 0/100 (0.0%) | {} |
| Sailor2-L 8B | 0/100 (0.0%) | {} |
| Llama 3.1 8B | 0/100 (0.0%) | {} |

## Context Relevance Score Distributions

Expected to be nearly identical across all models (retrieval is generator-independent).

- **SeaLLM v3 7B**: `{1: 3, 5: 97}`
- **Qwen 3.5 9B**: `{1: 5, 5: 95}`
- **Sailor2-L 8B**: `{1: 2, 5: 98}`
- **Llama 3.1 8B**: `{1: 2, 5: 98}`

## Faithfulness — Number Grounding

Fraction of Arabic/Khmer numerals in the generated answer that appear in the retrieved context. Lower fraction = higher hallucination risk.

| Model | n | Total Numbers | Grounded | Fraction | Zero-Number Answers |
|-------|---|--------------|---------|---------|--------------------|
| SeaLLM v3 7B | 100 | 45 | 45 | 1.0000 | 70 |
| Qwen 3.5 9B | 100 | 226 | 226 | 1.0000 | 35 |
| Sailor2-L 8B | 100 | 450 | 438 | 0.9733 | 1 |
| Llama 3.1 8B | 100 | 280 | 272 | 0.9714 | 23 |

## Completeness — Length Ratio Distribution

The `>2.0x` and `>3.0x` counts were the primary evidence for adding the ceiling to `score_completeness()`. Qwen's typical answer (p50=1.98x) was getting a free 5 under the old floor-only logic.

| Model | n | Ends Cleanly | p50 | p90 | max | >2.0x | >3.0x |
|-------|---|-------------|-----|-----|-----|-------|-------|
| SeaLLM v3 7B | 100 | 100/100 (100.0%) | 0.9974 | 1.0372 | 3.203 | 2 | 1 |
| Qwen 3.5 9B | 100 | 68/100 (68.0%) | 1.4899 | 5.1635 | 9.2479 | 39 | 23 |
| Sailor2-L 8B | 100 | 96/100 (96.0%) | 1.176 | 2.2186 | 5.1374 | 17 | 2 |
| Llama 3.1 8B | 100 | 74/100 (74.0%) | 1.076 | 2.9524 | 5.9829 | 19 | 10 |

## Answer Correctness — Similarity Distribution

BGE-M3 cosine similarity between generated answer and `expected_answer`. Blank answers are included at similarity=0.0 (not excluded).

| Model | n | Blank@0.0 | min | p50 | p90 | max |
|-------|---|-----------|-----|-----|-----|-----|
| SeaLLM v3 7B | 100 | 0 | 0.8848 | 0.9943 | 1.0 | 1.0 |
| Qwen 3.5 9B | 100 | 0 | 0.7824 | 0.949 | 0.9989 | 1.0 |
| Sailor2-L 8B | 100 | 0 | 0.7961 | 0.9177 | 0.9662 | 0.9991 |
| Llama 3.1 8B | 100 | 0 | 0.3736 | 0.873 | 0.9594 | 0.9991 |

## Answer Relevance — Similarity Distribution

BGE-M3 cosine similarity between generated answer and the original question. Structurally lower ceiling than Answer Correctness — these use separate threshold sets. Blank answers included at similarity=0.0.

| Model | n | Blank@0.0 | min | p50 | p90 | max |
|-------|---|-----------|-----|-----|-----|-----|
| SeaLLM v3 7B | 100 | 0 | 0.5302 | 0.7681 | 0.8136 | 0.8701 |
| Qwen 3.5 9B | 100 | 0 | 0.5302 | 0.7661 | 0.81 | 0.8529 |
| Sailor2-L 8B | 100 | 0 | 0.5209 | 0.755 | 0.8065 | 0.8379 |
| Llama 3.1 8B | 100 | 0 | 0.3385 | 0.7195 | 0.7843 | 0.8428 |

## Final Score Distributions (1–5) per Metric

### SeaLLM v3 7B

- **Context Relevance**: `{1: 3, 5: 97}`
- **Faithfulness**: `{5: 100}`
- **Answer Correctness**: `{4: 1, 5: 99}`
- **Completeness**: `{3: 1, 4: 6, 5: 93}`
- **Combined (weighted avg)**: `4.96`

### Qwen 3.5 9B

- **Context Relevance**: `{1: 5, 5: 95}`
- **Faithfulness**: `{5: 100}`
- **Answer Correctness**: `{3: 2, 4: 18, 5: 80}`
- **Completeness**: `{3: 32, 5: 68}`
- **Combined (weighted avg)**: `4.776`

### Sailor2-L 8B

- **Context Relevance**: `{1: 2, 5: 98}`
- **Faithfulness**: `{1: 1, 4: 8, 5: 91}`
- **Answer Correctness**: `{3: 1, 4: 31, 5: 68}`
- **Completeness**: `{3: 7, 4: 7, 5: 86}`
- **Combined (weighted avg)**: `4.7905`

### Llama 3.1 8B

- **Context Relevance**: `{1: 2, 5: 98}`
- **Faithfulness**: `{1: 1, 3: 2, 4: 1, 5: 96}`
- **Answer Correctness**: `{1: 1, 2: 3, 3: 15, 4: 40, 5: 41}`
- **Completeness**: `{2: 7, 3: 34, 4: 12, 5: 47}`
- **Combined (weighted avg)**: `4.478`

## Per-Category Breakdown

| Model | Category | n | Blank | Context Relevance | Faithfulness | Answer Correctness | Completeness |
|-------|----------|---|-------|-------|-------|-------|-------|
| SeaLLM v3 7B | guidance | 25 | 0 | 5.0000 | 5.0000 | 5.0000 | 4.8800 |
| SeaLLM v3 7B | laws | 25 | 0 | 5.0000 | 5.0000 | 4.9600 | 4.8400 |
| SeaLLM v3 7B | prakas | 25 | 0 | 4.6800 | 5.0000 | 5.0000 | 4.9600 |
| SeaLLM v3 7B | sub_decrees | 25 | 0 | 4.8400 | 5.0000 | 5.0000 | 5.0000 |
| Qwen 3.5 9B | guidance | 25 | 0 | 5.0000 | 5.0000 | 4.8800 | 4.2800 |
| Qwen 3.5 9B | laws | 25 | 0 | 5.0000 | 5.0000 | 4.7600 | 4.5200 |
| Qwen 3.5 9B | prakas | 25 | 0 | 4.3600 | 5.0000 | 4.7600 | 4.4400 |
| Qwen 3.5 9B | sub_decrees | 25 | 0 | 4.8400 | 5.0000 | 4.7200 | 4.2000 |
| Sailor2-L 8B | guidance | 25 | 0 | 5.0000 | 4.8400 | 4.9200 | 4.7200 |
| Sailor2-L 8B | laws | 25 | 0 | 5.0000 | 4.8800 | 4.5200 | 4.8800 |
| Sailor2-L 8B | prakas | 25 | 0 | 4.8400 | 4.9200 | 4.6800 | 4.8400 |
| Sailor2-L 8B | sub_decrees | 25 | 0 | 4.8400 | 4.8800 | 4.5600 | 4.7200 |
| Llama 3.1 8B | guidance | 25 | 0 | 5.0000 | 4.8400 | 4.2800 | 4.1200 |
| Llama 3.1 8B | laws | 25 | 0 | 5.0000 | 5.0000 | 4.3200 | 4.3200 |
| Llama 3.1 8B | prakas | 25 | 0 | 4.8400 | 4.8800 | 3.8400 | 3.6000 |
| Llama 3.1 8B | sub_decrees | 25 | 0 | 4.8400 | 4.9200 | 4.2400 | 3.9200 |

## Threshold Reference

### Answer Correctness

| Min Similarity | Score |
|---------------|-------|
| ≥ 0.90 | 5 |
| ≥ 0.80 | 4 |
| ≥ 0.70 | 3 |
| ≥ 0.60 | 2 |
| ≥ 0.50 | 1 |
| ≥ 0.00 | 1 |

### Answer Relevance (informational only — not weighted)

Kept for reference; does not contribute to the combined score.

| Min Similarity | Score |
|---------------|-------|
| ≥ 0.82 | 5 |
| ≥ 0.75 | 4 |
| ≥ 0.65 | 3 |
| ≥ 0.55 | 2 |
| ≥ 0.00 | 1 |

### Completeness (length_ratio = len(generated) / len(expected))

No over-length ceiling: a complete answer is not penalised for being more verbose than the terse gold passage (answer_correctness judges whether the extra words are right). Only truncation / under-length is penalised.

| Condition | Score |
|-----------|-------|
| Ends cleanly AND ratio ≥ 0.80 | 5 |
| Ends cleanly AND 0.60 ≤ ratio < 0.80 | 4 |
| Ends cleanly AND 0.40 ≤ ratio < 0.60 | 3 |
| Ends cleanly AND ratio < 0.40 | 2 |
| Truncated AND ratio ≥ 0.80 | 3 |
| Truncated AND ratio ≥ 0.40 | 2 |
| Truncated AND ratio < 0.40 (incl. blank) | 1 |

