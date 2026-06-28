# Generator Evaluation Summary: Qwen 3.5 9B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.8000      |
| Faithfulness            | 4.5600      |
| Answer Correctness      | 4.7800      |
| Answer Relevance        | 3.7500      |
| Completeness            | 4.1400      |
| **Combined (weighted)** | **4.4785**  |

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

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                5    |           4.16 |                 4.88 |               3.88 |           4.04 |
| laws        |                5    |           4.68 |                 4.76 |               3.56 |           4.2  |
| prakas      |                4.36 |           4.8  |                 4.76 |               3.84 |           4.4  |
| sub_decrees |                4.84 |           4.6  |                 4.72 |               3.72 |           3.92 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          28.26 |              79.46 |            649   |
| laws        |          22.45 |              68.4  |            574.7 |
| prakas      |          25.45 |              69.06 |            545.1 |
| sub_decrees |          31.95 |              87.52 |            690.2 |
