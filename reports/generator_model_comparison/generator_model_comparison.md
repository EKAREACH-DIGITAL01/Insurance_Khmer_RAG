# Generator Model Comparison Report

> Blank answers score **1** on Answer Correctness and Answer Relevance (not excluded from averages). See `docs/SCORING_CALIBRATION.md`.

## Quality Scores (macro-avg, 1–5 scale)

| Metric                  | SeaLLM v3 7B   | Qwen 3.5 9B   | Gemma 4 E4B   | Llama 3.1 8B   |
|-------------------------|----------------|---------------|---------------|----------------|
| Context Relevance       | 4.4800         | 4.4400        | 4.4400        | 4.4400         |
| Faithfulness            | 4.5200         | 3.7900        | 4.3000        | 4.5000         |
| Answer Correctness      | 4.6400         | 4.3600        | 4.0100        | 4.4800         |
| Answer Relevance        | 4.0100         | 3.8700        | 3.2600        | 3.9400         |
| Completeness            | 4.5900         | 4.0600        | 3.5100        | 4.2100         |
| **Combined (weighted)** | **4.4760**     | **4.1150**    | **3.9810**    | **4.3555**     |

## Latency & Coverage

| Metric                  |   SeaLLM v3 7B |   Qwen 3.5 9B |   Gemma 4 E4B |   Llama 3.1 8B |
|-------------------------|----------------|---------------|---------------|----------------|
| ── Avg TTFT (s) ──      |          23.74 |         20.15 |         37.93 |          35.56 |
| ── P50 TTFT (s) ──      |          20.23 |         18.96 |         34.75 |          31.03 |
| ── P95 TTFT (s) ──      |          59.7  |         40.44 |         68.76 |          86.81 |
| ── Avg gen time (s) ──  |          45.85 |         55.84 |         45.53 |          70.27 |
| ── Avg tok/s ──         |          12.41 |         12.6  |         23.07 |           8.95 |
| ── Blank answers ──     |           0    |          0    |          7    |           0    |
| ── Truncated answers ── |           8    |         26    |         45    |          23    |

## Ranking

1. **SeaLLM v3 7B** — combined=4.4760 | blank=0 | avg TTFT=23.7363s
2. **Llama 3.1 8B** — combined=4.3555 | blank=0 | avg TTFT=35.5572s
3. **Qwen 3.5 9B** — combined=4.1150 | blank=0 | avg TTFT=20.1488s
4. **Gemma 4 E4B** — combined=3.9810 | blank=7 | avg TTFT=37.9288s

**Recommendation: SeaLLM v3 7B**

Use this model for human-scored evaluation in `scripts/13_human_answer_evaluation.py`.
