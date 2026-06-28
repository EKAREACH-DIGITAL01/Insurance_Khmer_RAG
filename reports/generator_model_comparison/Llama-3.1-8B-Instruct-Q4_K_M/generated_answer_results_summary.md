# Generator Evaluation Summary: Llama 3.1 8B

## Proxy Metric Averages (1–5 scale)

> Answer Relevance is computed for traceability but NOT weighted. Blank answers score 1 on Answer Correctness (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.9200      |
| Faithfulness            | 4.9100      |
| Answer Correctness      | 4.1700      |
| Completeness            | 3.9900      |
| **Combined (weighted)** | **4.4780**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  26 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   53.76 |
| P50 first-token time (s)      |   50.01 |
| P95 first-token time (s)      |  104.32 |
| Avg total generation time (s) |  106.25 |
| Avg decode speed (tok/s)      |    8.44 |
| Avg tokens out                |  420.5  |
| Min tokens out                |   27    |
| Max tokens out                | 1260    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Completeness |
|-------------|---------------------|----------------|----------------------|----------------|
| guidance    |                5    |           4.84 |                 4.28 |           4.12 |
| laws        |                5    |           5    |                 4.32 |           4.32 |
| prakas      |                4.84 |           4.88 |                 3.84 |           3.6  |
| sub_decrees |                4.84 |           4.92 |                 4.24 |           3.92 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          76.07 |             148.13 |            404.8 |
| laws        |          35.74 |              72.28 |            365.6 |
| prakas      |          43.81 |              94.83 |            476.8 |
| sub_decrees |          58.33 |             107.12 |            431.3 |
