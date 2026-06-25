# Generator Evaluation Summary: SeaLLM v3 7B

## Proxy Metric Averages (1–5 scale)

> Blank answers score 1 on Answer Correctness and Answer Relevance (not excluded from the average).

| Metric                  | Avg Score   |
|-------------------------|-------------|
| Context Relevance       | 4.4800      |
| Faithfulness            | 4.5200      |
| Answer Correctness      | 4.6400      |
| Answer Relevance        | 4.0100      |
| Completeness            | 4.5900      |
| **Combined (weighted)** | **4.4760**  |

## Coverage

|-------------------|-----|
| Total questions   | 100 |
| Valid (scored)    | 100 |
| Failed (error)    |   0 |
| Blank answers     |   0 |
| Truncated answers |   8 |

## Latency & Throughput

|-------------------------------|--------|
| Avg first-token time (s)      |  23.74 |
| P50 first-token time (s)      |  20.23 |
| P95 first-token time (s)      |  59.7  |
| Avg total generation time (s) |  45.85 |
| Avg decode speed (tok/s)      |  12.41 |
| Avg tokens out                | 262.6  |
| Min tokens out                |  81    |
| Max tokens out                | 726    |

## Per-Category Scores

| Category    |   Context Relevance |   Faithfulness |   Answer Correctness |   Answer Relevance |   Completeness |
|-------------|---------------------|----------------|----------------------|--------------------|----------------|
| guidance    |                4.76 |           4.6  |                 4.76 |               4.4  |           4.48 |
| laws        |                4.04 |           4.56 |                 4.32 |               3.72 |           4.8  |
| prakas      |                4.28 |           4.56 |                 4.68 |               4.12 |           4.52 |
| sub_decrees |                4.84 |           4.36 |                 4.8  |               3.8  |           4.56 |

## Per-Category Timing (avg, seconds)

| Category    |   Avg TTFT (s) |   Avg Gen Time (s) |   Avg Tokens Out |
|-------------|----------------|--------------------|------------------|
| guidance    |          28.69 |              47.35 |            235.9 |
| laws        |          19.63 |              44.26 |            282   |
| prakas      |          20.59 |              42.33 |            246.4 |
| sub_decrees |          26.04 |              49.45 |            286.2 |
