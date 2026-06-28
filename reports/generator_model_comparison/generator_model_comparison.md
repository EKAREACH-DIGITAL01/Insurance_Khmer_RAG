# Generator Model Comparison Report

> Blank answers score **1** on Answer Correctness and Answer Relevance (not excluded from averages). See `docs/SCORING_CALIBRATION.md`.

## Quality Scores (macro-avg, 1–5 scale)

| Metric                  | SeaLLM v3 7B   | Qwen 3.5 9B   | Sailor2-L 8B   | Llama 3.1 8B   |
|-------------------------|----------------|---------------|----------------|----------------|
| Context Relevance       | 4.8700         | 4.8000        | 4.9200         | 4.9184         |
| Faithfulness            | 4.3800         | 4.5600        | 4.3700         | 4.1837         |
| Answer Correctness      | 4.7225         | 4.7800        | 4.6700         | 4.1633         |
| Answer Relevance        | 3.8150         | 3.7500        | 3.5500         | 3.0204         |
| Completeness            | 4.4050         | 4.1400        | 4.1500         | 3.8061         |
| **Combined (weighted)** | **4.4786**     | **4.4785**    | **4.3990**     | **4.0944**     |

## Latency & Coverage

| Metric                  |   SeaLLM v3 7B |   Qwen 3.5 9B |   Sailor2-L 8B |   Llama 3.1 8B |
|-------------------------|----------------|---------------|----------------|----------------|
| ── Avg TTFT (s) ──      |          31.26 |         27.02 |          35.9  |          52.65 |
| ── P50 TTFT (s) ──      |          28.01 |         25.51 |          32.66 |          48.76 |
| ── P95 TTFT (s) ──      |          65.31 |         42.95 |          62.2  |         101.92 |
| ── Avg gen time (s) ──  |          55.15 |         76.03 |          73.24 |         105.02 |
| ── Avg tok/s ──         |          12.73 |         12.55 |          11.01 |           8.41 |
| ── Blank answers ──     |           0    |          0    |           0    |           0    |
| ── Truncated answers ── |          20    |         32    |          31    |          37    |

## Ranking

1. **SeaLLM v3 7B** — combined=4.4786 | blank=0 | avg TTFT=31.2593s
2. **Qwen 3.5 9B** — combined=4.4785 | blank=0 | avg TTFT=27.0229s
3. **Sailor2-L 8B** — combined=4.3990 | blank=0 | avg TTFT=35.8964s
4. **Llama 3.1 8B** — combined=4.0944 | blank=0 | avg TTFT=52.6506s

**Recommendation: SeaLLM v3 7B**

Use this model for human-scored evaluation in `scripts/13_human_answer_evaluation.py`.
