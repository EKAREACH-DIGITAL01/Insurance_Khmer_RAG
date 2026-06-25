# Generator Evaluation Summary: Qwen 3.5 9B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.4400      |
| Faithfulness            | 3.7900      |
| Answer Correctness      | 4.3600      |
| Answer Relevance        | 3.8700      |
| Completeness            | 4.0600      |
| **Combined (weighted)** | **4.1150**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  26 |

## Latency & Throughput

|-------------------------------|--------|
| Avg first-token time (s)      |  20.15 |
| P50 first-token time (s)      |  18.96 |
| P95 first-token time (s)      |  40.44 |
| Avg total generation time (s) |  55.84 |
| Avg decode speed (tok/s)      |  12.6  |
| Avg tokens out                | 449.6  |
| Min tokens out                |  48    |
| Max tokens out                | 829    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                4.76 |           3.84 |                 4.52 |               4.2  |           3.76 |
| laws        |                4.04 |           4.16 |                 4.04 |               3.52 |           3.96 |
| prakas      |                4.28 |           3.96 |                 4.44 |               4.08 |           4.52 |
| sub_decrees |                4.68 |           3.2  |                 4.44 |               3.68 |           4    |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          24.41 |              65.61 |            525.5 |
| laws        |          16.18 |              55.12 |            486.6 |
| prakas      |          17.05 |              41.22 |            306.8 |
| sub_decrees |          22.96 |              61.4  |            479.3 |
