# Generator Evaluation Summary: Qwen 3.5 9B

## Proxy Metric Averages (1–5 scale)

> Answer Relevance is computed for traceability but NOT weighted. Blank answers score 1 on Answer Correctness (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.8000      |
| Faithfulness            | 5.0000      |
| Answer Correctness      | 4.7800      |
| Completeness            | 4.3600      |
| **Combined (weighted)** | **4.7760**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  32 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   27.02 |
| P50 first-token time (s)      |   25.51 |
| P95 first-token time (s)      |   42.95 |
| Avg total generation time (s) |   76.03 |
| Avg decode speed (tok/s)      |   12.55 |
| Avg tokens out                |  614    |
| Min tokens out                |  103    |
| Max tokens out                | 2156    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Completeness |
|-------------|---------------------|----------------|----------------------|----------------|
| guidance    |                5    |              5 |                 4.88 |           4.28 |
| laws        |                5    |              5 |                 4.76 |           4.52 |
| prakas      |                4.36 |              5 |                 4.76 |           4.44 |
| sub_decrees |                4.84 |              5 |                 4.72 |           4.2  |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          28.26 |              79.46 |            649   |
| laws        |          22.45 |              68.4  |            574.7 |
| prakas      |          25.45 |              69.06 |            545.1 |
| sub_decrees |          31.95 |              87.52 |            690.2 |
