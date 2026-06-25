# Generator Evaluation Summary: Llama 3.1 8B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.4400      |
| Faithfulness            | 4.5000      |
| Answer Correctness      | 4.4800      |
| Answer Relevance        | 3.9400      |
| Completeness            | 4.2100      |
| **Combined (weighted)** | **4.3555**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  23 |

## Latency & Throughput

|-------------------------------|--------|
| Avg first-token time (s)      |  35.56 |
| P50 first-token time (s)      |  31.03 |
| P95 first-token time (s)      |  86.81 |
| Avg total generation time (s) |  70.27 |
| Avg decode speed (tok/s)      |   8.95 |
| Avg tokens out                | 302.6  |
| Min tokens out                |  82    |
| Max tokens out                | 570    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                4.76 |           4.6  |                 4.64 |               4.24 |           4.44 |
| laws        |                4.04 |           4.56 |                 4.2  |               3.76 |           4    |
| prakas      |                4.28 |           4.48 |                 4.52 |               4.08 |           4.36 |
| sub_decrees |                4.68 |           4.36 |                 4.56 |               3.68 |           4.04 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          44.65 |              76.99 |            280.4 |
| laws        |          27.21 |              67.31 |            356.1 |
| prakas      |          28.8  |              56.17 |            252.7 |
| sub_decrees |          41.58 |              80.61 |            321.2 |
