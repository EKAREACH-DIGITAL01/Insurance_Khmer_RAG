# Generator Evaluation Summary: Sailor2-L 8B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.9200      |
| Faithfulness            | 4.3700      |
| Answer Correctness      | 4.6700      |
| Answer Relevance        | 3.5500      |
| Completeness            | 4.1500      |
| **Combined (weighted)** | **4.3990**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |  31 |

## Latency & Throughput

|-------------------------------|--------|
| Avg first-token time (s)      |  35.9  |
| P50 first-token time (s)      |  32.66 |
| P95 first-token time (s)      |  62.2  |
| Avg total generation time (s) |  73.24 |
| Avg decode speed (tok/s)      |  11.01 |
| Avg tokens out                | 407.4  |
| Min tokens out                | 122    |
| Max tokens out                | 899    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                5    |           3.96 |                 4.92 |               3.76 |           4.04 |
| laws        |                5    |           4.56 |                 4.52 |               3.24 |           4.32 |
| prakas      |                4.84 |           4.36 |                 4.68 |               3.6  |           4.08 |
| sub_decrees |                4.84 |           4.6  |                 4.56 |               3.6  |           4.16 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          43.12 |              84.31 |            425.4 |
| laws        |          26.18 |              57.32 |            368.9 |
| prakas      |          31.76 |              69.42 |            412.3 |
| sub_decrees |          42.26 |              81.29 |            421   |
