# Generator Model Comparison Report

> Weighted metrics: Context Relevance, Faithfulness, Answer Correctness, Completeness. Answer Relevance is shown for reference but NOT weighted. Blank answers score **1** on Answer Correctness (not excluded from averages). See `docs/SCORING_CALIBRATION.md`.

## Quality Scores (macro-avg, 1–5 scale)

| Metric                  | SeaLLM v3 7B   | Qwen 3.5 9B   | Sailor2-L 8B   | Llama 3.1 8B   |
|-------------------------|----------------|---------------|----------------|----------------|
| Context Relevance       | 4.8800         | 4.8000        | 4.9200         | 4.9200         |
| Faithfulness            | 5.0000         | 5.0000        | 4.8800         | 4.9100         |
| Answer Correctness      | 4.9900         | 4.7800        | 4.6700         | 4.1700         |
| Completeness            | 4.9200         | 4.3600        | 4.7900         | 3.9900         |
| **Combined (weighted)** | **4.9600**     | **4.7760**    | **4.7905**     | **4.4780**     |

## Latency & Coverage

| Metric                  |   SeaLLM v3 7B |   Qwen 3.5 9B |   Sailor2-L 8B |   Llama 3.1 8B |
|-------------------------|----------------|---------------|----------------|----------------|
| ── Avg TTFT (s) ──      |          31.48 |         27.02 |          35.9  |          53.76 |
| ── P50 TTFT (s) ──      |          28.46 |         25.51 |          32.66 |          50.01 |
| ── P95 TTFT (s) ──      |          56.4  |         42.95 |          62.2  |         104.32 |
| ── Avg gen time (s) ──  |          60.77 |         76.03 |          73.24 |         106.25 |
| ── Avg tok/s ──         |          12.39 |         12.55 |          11.01 |           8.44 |
| ── Blank answers ──     |           0    |          0    |           0    |           0    |
| ── Truncated answers ── |           0    |         32    |           4    |          26    |

## Ranking

1. **SeaLLM v3 7B** — combined=4.9600 | blank=0 | avg TTFT=31.4802s
2. **Sailor2-L 8B** — combined=4.7905 | blank=0 | avg TTFT=35.8964s
3. **Qwen 3.5 9B** — combined=4.7760 | blank=0 | avg TTFT=27.0229s
4. **Llama 3.1 8B** — combined=4.4780 | blank=0 | avg TTFT=53.7557s

**Recommendation: SeaLLM v3 7B**

Use this model for human-scored evaluation in `scripts/15_human_answer_evaluation.py`.
