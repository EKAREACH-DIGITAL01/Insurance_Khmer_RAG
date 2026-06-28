# Generator Evaluation Summary: SeaLLM v3 7B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.8800      |
| Faithfulness            | 4.4000      |
| Answer Correctness      | 4.9900      |
| Answer Relevance        | 3.7600      |
| Completeness            | 4.8900      |
| **Combined (weighted)** | **4.6050**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |   0 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   31.48 |
| P50 first-token time (s)      |   28.46 |
| P95 first-token time (s)      |   56.4  |
| Avg total generation time (s) |   60.77 |
| Avg decode speed (tok/s)      |   12.39 |
| Avg tokens out                |  357.8  |
| Min tokens out                |   91    |
| Max tokens out                | 1491    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                5    |           4.44 |                 5    |               4.12 |           4.88 |
| laws        |                5    |           4.36 |                 4.96 |               3.48 |           4.8  |
| prakas      |                4.68 |           4.4  |                 5    |               3.72 |           4.96 |
| sub_decrees |                4.84 |           4.4  |                 5    |               3.72 |           4.92 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          34.95 |              63.91 |            366.6 |
| laws        |          28.53 |              58.21 |            323.4 |
| prakas      |          26.62 |              54.69 |            360.8 |
| sub_decrees |          35.94 |              66.53 |            379.2 |
