# Generator Evaluation Summary: SeaLLM v3 7B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.8700      |
| Faithfulness            | 4.3800      |
| Answer Correctness      | 4.7225      |
| Answer Relevance        | 3.8150      |
| Completeness            | 4.4050      |
| **Combined (weighted)** | **4.4786**  |

## Coverage

|-------------------|-----|
| Total questions   | 400 |
| Valid (scored)    | 400 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  20 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   31.26 |
| P50 first-token time (s)      |   28.01 |
| P95 first-token time (s)      |   65.31 |
| Avg total generation time (s) |   55.15 |
| Avg decode speed (tok/s)      |   12.73 |
| Avg tokens out                |  300.4  |
| Min tokens out                |   55    |
| Max tokens out                | 1903    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |              4.92   |         4.27   |               4.77   |             3.9    |         4.3    |
| laws        |              4.9467 |         4.28   |               4.7733 |             3.68   |         4.4    |
| prakas      |              4.76   |         4.4867 |               4.6933 |             3.84   |         4.4867 |
| sub_decrees |              4.9467 |         4.4133 |               4.6667 |             3.7867 |         4.3867 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          34.73 |              64.18 |            371.3 |
| laws        |          29.57 |              52.13 |            271.4 |
| prakas      |          28.82 |              49.23 |            263.2 |
| sub_decrees |          33.15 |              57.85 |            307.9 |
