# Generator Model Comparison Report

> **Context Relevance** is measured deterministically by `scripts/14` as recall of `expected_chunks` in `context_chunk_ids` — it scores retrieval, not the answer. The other four metrics are assigned by an **LLM judge** comparing `generated_answer` against `expected_answer`. Hallucination is judged separately and is **never** folded into the combined score. See `docs/llm_judge_prompt.md`.

## Quality Scores (macro-avg, 1–5 scale)

| Metric                  | SeaLLMs-v3-7B-Chat-Q4_K_M   | Sailor2-L-8B-Chat-Q4_K_M   | Llama-3.1-8B-Instruct-Q4_K_M   | Qwen3.5-9B-Q4_K_M   |
|-------------------------|-----------------------------|----------------------------|--------------------------------|---------------------|
| Context Relevance       | 4.8800                      | 4.9200                     | 4.9200                         | 4.8000              |
| Faithfulness            | 4.8600                      | 4.5900                     | 4.5500                         | 3.5000              |
| Answer Relevance        | 4.9700                      | 4.8600                     | 4.5700                         | 4.5100              |
| Answer Correctness      | 4.8400                      | 4.7700                     | 4.2000                         | 4.6100              |
| Completeness            | 4.9000                      | 4.8900                     | 4.2400                         | 3.7600              |
| **Combined (weighted)** | **4.8815**                  | **4.7865**                 | **4.4930**                     | **4.2280**          |

## Hallucination (separate from the score)

A hallucination is any specific fact asserted by the answer — an article number, figure, deadline, institution or procedure — that does not appear in `expected_answer`. Judged independently of the five metrics and **never** folded into the combined score: a model can rank first and still fabricate.

> Read these as an **upper bound**. Because the reference is the gold answer rather than the retrieved context, a detail correctly taken from the source document but omitted from the gold answer is still counted. Verbose models are the most exposed. Each flag carries the quoted span in the per-model summary, so they can be checked individually.

| Metric        | SeaLLMs-v3-7B-Chat-Q4_K_M   | Sailor2-L-8B-Chat-Q4_K_M   | Llama-3.1-8B-Instruct-Q4_K_M   | Qwen3.5-9B-Q4_K_M   |
|---------------|-----------------------------|----------------------------|--------------------------------|---------------------|
| ── Flagged ── | 5                           | 16                         | 7                              | 55                  |
| ── Judged ──  | 100                         | 100                        | 100                            | 100                 |
| ── Rate ──    | 5%                          | 16%                        | 7%                             | 55%                 |

### Rate by category

| Category    | SeaLLMs-v3-7B-Chat-Q4_K_M   | Sailor2-L-8B-Chat-Q4_K_M   | Llama-3.1-8B-Instruct-Q4_K_M   | Qwen3.5-9B-Q4_K_M   |
|-------------|-----------------------------|----------------------------|--------------------------------|---------------------|
| guidance    | 4%                          | 8%                         | 20%                            | 60%                 |
| laws        | 4%                          | 32%                        | 0%                             | 48%                 |
| prakas      | 0%                          | 16%                        | 4%                             | 52%                 |
| sub_decrees | 12%                         | 8%                         | 4%                             | 60%                 |

### Questions multiple models hallucinated on

More than one model fabricating on the same question usually means the retrieved context was inadequate, not that every model failed independently. Check retrieval for these first.

| Question ID        |   Models | Which                                                                     |
|--------------------|----------|---------------------------------------------------------------------------|
| prakas_qid_002     |        3 | Sailor2-L-8B-Chat-Q4_K_M, Llama-3.1-8B-Instruct-Q4_K_M, Qwen3.5-9B-Q4_K_M |
| guidance_qid_025   |        2 | SeaLLMs-v3-7B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                              |
| law_qid_014        |        2 | SeaLLMs-v3-7B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                              |
| sub_decree_qid_010 |        2 | SeaLLMs-v3-7B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                              |
| sub_decree_qid_014 |        2 | SeaLLMs-v3-7B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                              |
| guidance_qid_001   |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| guidance_qid_020   |        2 | Sailor2-L-8B-Chat-Q4_K_M, Llama-3.1-8B-Instruct-Q4_K_M                    |
| law_qid_006        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| law_qid_010        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| law_qid_016        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| law_qid_018        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| law_qid_019        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| law_qid_020        |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| prakas_qid_005     |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| sub_decree_qid_017 |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| sub_decree_qid_025 |        2 | Sailor2-L-8B-Chat-Q4_K_M, Qwen3.5-9B-Q4_K_M                               |
| guidance_qid_005   |        2 | Llama-3.1-8B-Instruct-Q4_K_M, Qwen3.5-9B-Q4_K_M                           |
| guidance_qid_009   |        2 | Llama-3.1-8B-Instruct-Q4_K_M, Qwen3.5-9B-Q4_K_M                           |
| guidance_qid_019   |        2 | Llama-3.1-8B-Instruct-Q4_K_M, Qwen3.5-9B-Q4_K_M                           |

## Latency & Coverage

| Metric                  |   SeaLLMs-v3-7B-Chat-Q4_K_M |   Sailor2-L-8B-Chat-Q4_K_M |   Llama-3.1-8B-Instruct-Q4_K_M |   Qwen3.5-9B-Q4_K_M |
|-------------------------|-----------------------------|----------------------------|--------------------------------|---------------------|
| ── Avg TTFT (s) ──      |                     31.4802 |                    35.8964 |                        53.7557 |             27.0229 |
| ── Avg gen time (s) ──  |                     60.771  |                    73.2418 |                       106.252  |             76.0336 |
| ── Avg tok/s ──         |                     12.3873 |                    11.011  |                         8.4432 |             12.5454 |
| ── Blank answers ──     |                      0      |                     0      |                         0      |              0      |
| ── Truncated answers ── |                      0      |                     4      |                        26      |             32      |
| ── Unjudged ──          |                      0      |                     0      |                         0      |              0      |

## Ranking

1. **SeaLLMs-v3-7B-Chat-Q4_K_M** — combined=4.8815 | hallucinated=5 | blank=0
2. **Sailor2-L-8B-Chat-Q4_K_M** — combined=4.7865 | hallucinated=16 | blank=0
3. **Llama-3.1-8B-Instruct-Q4_K_M** — combined=4.4930 | hallucinated=7 | blank=0
4. **Qwen3.5-9B-Q4_K_M** — combined=4.2280 | hallucinated=55 | blank=0

**Recommendation: SeaLLMs-v3-7B-Chat-Q4_K_M**

Use this model for the live chatbot and `scripts/16_evaluate_safety_gates.py`.
