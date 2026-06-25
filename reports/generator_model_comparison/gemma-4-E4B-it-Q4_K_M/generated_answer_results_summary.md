# Generator Evaluation Summary: Gemma 4 E4B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.4400      |
| Faithfulness            | 4.3000      |
| Answer Correctness      | 4.2366      |
| Answer Relevance        | 3.4301      |
| Completeness            | 3.5100      |
| **Combined (weighted)** | **4.0373**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  45 |

## Latency & Throughput

|-------------------------------|--------|
| Avg first-token time (s)      |  37.93 |
| P50 first-token time (s)      |  34.75 |
| P95 first-token time (s)      |  68.76 |
| Avg total generation time (s) |  45.53 |
| Avg decode speed (tok/s)      |  23.07 |
| Avg tokens out                | 103.4  |
| Min tokens out                |   0    |
| Max tokens out                | 308    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                4.76 |           4.48 |               4.5217 |             3.7391 |           3.72 |
| laws        |                4.04 |           4.48 |               4.1304 |             3.087  |           3.72 |
| prakas      |                4.28 |           4.08 |               4.5417 |             3.75   |           3.96 |
| sub_decrees |                4.68 |           4.16 |               3.7391 |             3.1304 |           2.64 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          39.54 |              47.99 |            105.9 |
| laws        |          34.39 |              42.61 |            106.6 |
| prakas      |          34.01 |              39.97 |            108.9 |
| sub_decrees |          43.78 |              51.54 |             92.1 |
