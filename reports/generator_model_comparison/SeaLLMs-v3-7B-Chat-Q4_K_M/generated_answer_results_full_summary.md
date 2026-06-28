# Generator Evaluation Summary: SeaLLM v3 7B

## Proxy Metric Averages (1–5 scale)

> Answer Relevance is computed for traceability but NOT weighted. Blank answers score 1 on Answer Correctness (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.8700      |
| Faithfulness            | 5.0000      |
| Answer Correctness      | 4.7200      |
| Completeness            | 4.4300      |
| **Combined (weighted)** | **4.7765**  |

## Coverage

|-------------------|-----|
| Total questions   | 400 |
| Valid (scored)    | 400 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  15 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   31.45 |
| P50 first-token time (s)      |   28.11 |
| P95 first-token time (s)      |   65.31 |
| Avg total generation time (s) |   55.24 |
| Avg decode speed (tok/s)      |   12.67 |
| Avg tokens out                |  296.7  |
| Min tokens out                |   55    |
| Max tokens out                | 1903    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Completeness |
|-------------|---------------------|----------------|----------------------|----------------|
| guidance    |              4.92   |              5 |               4.76   |         4.36   |
| laws        |              4.9467 |              5 |               4.7733 |         4.4667 |
| prakas      |              4.76   |              5 |               4.6933 |         4.4467 |
| sub_decrees |              4.9467 |              5 |               4.6667 |         4.4533 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          35.15 |              64.54 |            366.9 |
| laws        |          30.03 |              52.13 |            257.6 |
| prakas      |          28.82 |              49.23 |            263.2 |
| sub_decrees |          33.15 |              57.85 |            307.9 |
