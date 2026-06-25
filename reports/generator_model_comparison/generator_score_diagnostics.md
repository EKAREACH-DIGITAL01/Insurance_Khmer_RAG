# Generator Score Diagnostics

Evidence file for the calibration of `scripts/10_compare_generator_models.py` scoring thresholds. See `docs/SCORING_CALIBRATION.md` for the full interpretation.

> **Blank answer policy**: blank answers score **1** on Answer Correctness and Answer Relevance (not excluded). `similarity = 0.0` is stored in `detail.answer_correctness_similarity` and `detail.answer_relevance_similarity` for traceability.

## Blank Answers

| Model | Blank / Total | By Category |
|-------|--------------|-------------|
| SeaLLM v3 7B | 0/100 (0.0%) | {} |
| Qwen 3.5 9B | 0/100 (0.0%) | {} |
| Gemma 4 E4B | 7/100 (7.0%) | {'guidance': 2, 'laws': 2, 'prakas': 1, 'sub_decrees': 2} |
| Llama 3.1 8B | 0/100 (0.0%) | {} |

## Context Relevance Score Distributions

Expected to be nearly identical across all models (retrieval is generator-independent).

- **SeaLLM v3 7B**: `{1: 11, 3: 4, 5: 85}`
- **Qwen 3.5 9B**: `{1: 12, 3: 4, 5: 84}`
- **Gemma 4 E4B**: `{1: 12, 3: 4, 5: 84}`
- **Llama 3.1 8B**: `{1: 12, 3: 4, 5: 84}`

## Faithfulness — Number Grounding

Fraction of Arabic/Khmer numerals in the generated answer that appear in the retrieved context. Lower fraction = higher hallucination risk.

| Model | n | Total Numbers | Grounded | Fraction | Zero-Number Answers |
|-------|---|--------------|---------|---------|--------------------|
| SeaLLM v3 7B | 100 | 113 | 111 | 0.9823 | 46 |
| Qwen 3.5 9B | 100 | 418 | 301 | 0.7201 | 13 |
| Gemma 4 E4B | 100 | 116 | 107 | 0.9224 | 52 |
| Llama 3.1 8B | 100 | 145 | 145 | 1.0000 | 50 |

## Completeness — Length Ratio Distribution

The `>2.0x` and `>3.0x` counts were the primary evidence for adding the ceiling to `score_completeness()`. Qwen's typical answer (p50=1.98x) was getting a free 5 under the old floor-only logic.

| Model | n | Ends Cleanly | p50 | p90 | max | >2.0x | >3.0x |
|-------|---|-------------|-----|-----|-----|-------|-------|
| SeaLLM v3 7B | 100 | 92/100 (92.0%) | 0.9968 | 1.7362 | 5.7717 | 9 | 5 |
| Qwen 3.5 9B | 100 | 74/100 (74.0%) | 1.979 | 4.3864 | 10.0093 | 49 | 30 |
| Gemma 4 E4B | 100 | 55/100 (55.0%) | 1.0 | 1.4603 | 2.4157 | 2 | 0 |
| Llama 3.1 8B | 100 | 77/100 (77.0%) | 1.127 | 2.695 | 5.9352 | 21 | 5 |

## Answer Correctness — Similarity Distribution

BGE-M3 cosine similarity between generated answer and `expected_answer`. Blank answers are included at similarity=0.0 (not excluded).

| Model | n | Blank@0.0 | min | p50 | p90 | max |
|-------|---|-----------|-----|-----|-----|-----|
| SeaLLM v3 7B | 100 | 0 | 0.6264 | 0.9416 | 0.9947 | 1.0 |
| Qwen 3.5 9B | 100 | 0 | 0.4718 | 0.9047 | 0.9631 | 0.9861 |
| Gemma 4 E4B | 100 | 7 | 0.0 | 0.9225 | 0.9951 | 1.0 |
| Llama 3.1 8B | 100 | 0 | 0.6208 | 0.9213 | 0.9946 | 1.0 |

## Answer Relevance — Similarity Distribution

BGE-M3 cosine similarity between generated answer and the original question. Structurally lower ceiling than Answer Correctness — these use separate threshold sets. Blank answers included at similarity=0.0.

| Model | n | Blank@0.0 | min | p50 | p90 | max |
|-------|---|-----------|-----|-----|-----|-----|
| SeaLLM v3 7B | 100 | 0 | 0.6853 | 0.7767 | 0.837 | 0.9037 |
| Qwen 3.5 9B | 100 | 0 | 0.3489 | 0.7711 | 0.8236 | 0.8753 |
| Gemma 4 E4B | 100 | 7 | 0.0 | 0.7484 | 0.8158 | 0.8741 |
| Llama 3.1 8B | 100 | 0 | 0.6771 | 0.7721 | 0.838 | 0.8741 |

## Final Score Distributions (1–5) per Metric

### SeaLLM v3 7B

- **Context Relevance**: `{1: 11, 3: 4, 5: 85}`
- **Faithfulness**: `{4: 48, 5: 52}`
- **Answer Correctness**: `{2: 1, 3: 5, 4: 23, 5: 71}`
- **Answer Relevance**: `{3: 22, 4: 55, 5: 23}`
- **Completeness**: `{1: 1, 3: 12, 4: 13, 5: 74}`
- **Combined (weighted avg)**: `4.476`

### Qwen 3.5 9B

- **Context Relevance**: `{1: 12, 3: 4, 5: 84}`
- **Faithfulness**: `{1: 6, 2: 11, 3: 22, 4: 20, 5: 41}`
- **Answer Correctness**: `{1: 1, 2: 2, 3: 8, 4: 38, 5: 51}`
- **Answer Relevance**: `{1: 1, 3: 23, 4: 63, 5: 13}`
- **Completeness**: `{3: 40, 4: 14, 5: 46}`
- **Combined (weighted avg)**: `4.115`

### Gemma 4 E4B

- **Context Relevance**: `{1: 12, 3: 4, 5: 84}`
- **Faithfulness**: `{1: 4, 3: 1, 4: 52, 5: 43}`
- **Answer Correctness**: `{1: 13, 2: 5, 3: 8, 4: 16, 5: 58}`
- **Answer Relevance**: `{1: 14, 2: 5, 3: 31, 4: 41, 5: 9}`
- **Completeness**: `{1: 18, 2: 13, 3: 18, 4: 2, 5: 49}`
- **Combined (weighted avg)**: `3.981`

### Llama 3.1 8B

- **Context Relevance**: `{1: 12, 3: 4, 5: 84}`
- **Faithfulness**: `{4: 50, 5: 50}`
- **Answer Correctness**: `{2: 3, 3: 6, 4: 31, 5: 60}`
- **Answer Relevance**: `{3: 24, 4: 58, 5: 18}`
- **Completeness**: `{2: 2, 3: 28, 4: 17, 5: 53}`
- **Combined (weighted avg)**: `4.3555`

## Per-Category Breakdown

| Model | Category | n | Blank | Context Relevance | Faithfulness | Answer Correctness | Answer Relevance | Completeness |
|-------|----------|---|-------|-------|-------|-------|-------|-------|
| SeaLLM v3 7B | guidance | 25 | 0 | 4.7600 | 4.6000 | 4.7600 | 4.4000 | 4.4800 |
| SeaLLM v3 7B | laws | 25 | 0 | 4.0400 | 4.5600 | 4.3200 | 3.7200 | 4.8000 |
| SeaLLM v3 7B | prakas | 25 | 0 | 4.2800 | 4.5600 | 4.6800 | 4.1200 | 4.5200 |
| SeaLLM v3 7B | sub_decrees | 25 | 0 | 4.8400 | 4.3600 | 4.8000 | 3.8000 | 4.5600 |
| Qwen 3.5 9B | guidance | 25 | 0 | 4.7600 | 3.8400 | 4.5200 | 4.2000 | 3.7600 |
| Qwen 3.5 9B | laws | 25 | 0 | 4.0400 | 4.1600 | 4.0400 | 3.5200 | 3.9600 |
| Qwen 3.5 9B | prakas | 25 | 0 | 4.2800 | 3.9600 | 4.4400 | 4.0800 | 4.5200 |
| Qwen 3.5 9B | sub_decrees | 25 | 0 | 4.6800 | 3.2000 | 4.4400 | 3.6800 | 4.0000 |
| Gemma 4 E4B | guidance | 25 | 2 | 4.7600 | 4.4800 | 4.2400 | 3.5200 | 3.7200 |
| Gemma 4 E4B | laws | 25 | 2 | 4.0400 | 4.4800 | 3.8800 | 2.9200 | 3.7200 |
| Gemma 4 E4B | prakas | 25 | 1 | 4.2800 | 4.0800 | 4.4000 | 3.6400 | 3.9600 |
| Gemma 4 E4B | sub_decrees | 25 | 2 | 4.6800 | 4.1600 | 3.5200 | 2.9600 | 2.6400 |
| Llama 3.1 8B | guidance | 25 | 0 | 4.7600 | 4.6000 | 4.6400 | 4.2400 | 4.4400 |
| Llama 3.1 8B | laws | 25 | 0 | 4.0400 | 4.5600 | 4.2000 | 3.7600 | 4.0000 |
| Llama 3.1 8B | prakas | 25 | 0 | 4.2800 | 4.4800 | 4.5200 | 4.0800 | 4.3600 |
| Llama 3.1 8B | sub_decrees | 25 | 0 | 4.6800 | 4.3600 | 4.5600 | 3.6800 | 4.0400 |

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

### Answer Relevance

| Min Similarity | Score |
|---------------|-------|
| ≥ 0.82 | 5 |
| ≥ 0.75 | 4 |
| ≥ 0.65 | 3 |
| ≥ 0.55 | 2 |
| ≥ 0.00 | 1 |

### Completeness (length_ratio = len(generated) / len(expected))

| Condition | Score |
|-----------|-------|
| Ends cleanly AND 0.80 ≤ ratio ≤ 2.00 | 5 |
| Ends cleanly AND (0.60 ≤ ratio < 0.80 OR 2.00 < ratio ≤ 3.00) | 4 |
| Ends cleanly AND (ratio < 0.60 OR ratio > 3.00) | 3 |
| Truncated AND ratio ≥ 0.80 | 3 |
| Truncated AND ratio ≥ 0.40 | 2 |
| Truncated AND ratio < 0.40 (incl. blank) | 1 |

