# Generator Evaluation Summary: Llama 3.1 8B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.9184      |
| Faithfulness            | 4.1837      |
| Answer Correctness      | 4.1633      |
| Answer Relevance        | 3.0204      |
| Completeness            | 3.8061      |
| **Combined (weighted)** | **4.0944**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    |  98 |
| Failed (error)    |   2 |
| Blank answers     |   0 |
| Truncated answers |  37 |

## Latency & Throughput

|-------------------------------|---------|
| Avg first-token time (s)      |   52.65 |
| P50 first-token time (s)      |   48.76 |
| P95 first-token time (s)      |  101.92 |
| Avg total generation time (s) |  105.02 |
| Avg decode speed (tok/s)      |    8.41 |
| Avg tokens out                |  417.1  |
| Min tokens out                |   27    |
| Max tokens out                | 1260    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |              5      |          3.625 |                 4.25 |               3.25 |         4      |
| laws        |              5      |          4.48  |                 4.32 |               2.8  |         4.08   |
| prakas      |              4.84   |          4.24  |                 3.84 |               2.8  |         3.48   |
| sub_decrees |              4.8333 |          4.375 |                 4.25 |               3.25 |         3.6667 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          74.99 |             147.52 |            396   |
| laws        |          35.74 |              72.28 |            365.6 |
| prakas      |          43.81 |              94.83 |            476.8 |
| sub_decrees |          56.01 |             104.48 |            426   |
