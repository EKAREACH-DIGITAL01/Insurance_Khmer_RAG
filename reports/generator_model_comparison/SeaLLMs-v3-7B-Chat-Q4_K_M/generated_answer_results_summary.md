# SeaLLMs-v3-7B-Chat-Q4_K_M — Answer Summary

Scored by an LLM judge (see `docs/llm_judge_prompt.md`). 100/100 answers judged.

## Quality (1–5)

| Metric                  |    Avg |
|-------------------------|--------|
| Context Relevance       | 4.88   |
| Faithfulness            | 4.86   |
| Answer Relevance        | 4.97   |
| Answer Correctness      | 4.84   |
| Completeness            | 4.9    |
| **Combined (weighted)** | 4.8815 |

## Hallucination

- **Flagged:** 5/100 (5%)
- Judged separately; never folded into the combined score.

### By category

| Category    |   Flagged |   Judged | Rate   |
|-------------|-----------|----------|--------|
| guidance    |         1 |       25 | 4%     |
| laws        |         1 |       25 | 4%     |
| prakas      |         0 |       25 | 0%     |
| sub_decrees |         3 |       25 | 12%    |

### Flagged answers

The span quoted is what the judge said was fabricated — check it against the retrieved context before acting on it.

| Question ID        | Category    | Question                                         | Fabricated span                                              |   Faith. |
|--------------------|-------------|--------------------------------------------------|--------------------------------------------------------------|----------|
| guidance_qid_025   | guidance    | វិញ្ញាបនបត្រធានារ៉ាប់រងជាឯកសារត្រូវភ្ជាប់ជាមួយន… | ហត្ថលេខា និងត្រារបស់ក្រុមហ៊ុនធានារ៉ាប់រង                     |        3 |
| law_qid_014        | laws        | តើបុគ្គលណាមួយដែលមិនសុខចិត្តនឹងសេចក្តីសម្រេចរបស់… | 2. ប្តឹងទាមទារឱ្យគណៈកម្មាធិការជាតិការពារអ្នកប្រើប្រាស់ ពិនិ… |        2 |
| sub_decree_qid_007 | sub_decrees | តើកំណត់សំអាង ជាឯកសារអ្វី?                        | 3. ជូនដំណឹងដល់អ្នកទទួលផល ឬសាច់ញាតិ ឬម្ចាស់បណ្ណសន្យារ៉ាប់រង … |        2 |
| sub_decree_qid_010 | sub_decrees | តើការធានារ៉ាប់រងខ្នាតតូចមានអានុភាពដូចម្តេច?      | ក្នុងករណីកិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូចមិនមានកំណត់ពីកាលបរិច្… |        3 |
| sub_decree_qid_014 | sub_decrees | តើផលិតផលធានារ៉ាប់រងខ្នាតតូច អាចត្រូវបានចែកចាយតា… | ផលិតផលធានារ៉ាប់រងខ្នាតតូចអាចត្រូវបានចែកចាយតាមប្រព័ន្ធអេឡិចត… |        3 |

## Coverage & latency

| Metric            |    Value |
|-------------------|----------|
| Blank answers     |   0      |
| Truncated answers |   0      |
| Avg TTFT (s)      |  31.4802 |
| Avg gen time (s)  |  60.771  |
| Avg tokens out    | 357.76   |
| Avg tok/s         |  12.3873 |

## Per category

| Category    |   Qs |   Context Relevance |   Faithfulness |   Answer Relevance |   Answer Correctness |   Completeness |   Combined |
|-------------|------|---------------------|----------------|--------------------|----------------------|----------------|------------|
| guidance    |   25 |                5    |           4.88 |               5    |                 4.8  |           4.84 |      4.896 |
| laws        |   25 |                5    |           4.84 |               5    |                 4.76 |           4.84 |      4.876 |
| prakas      |   25 |                4.68 |           5    |               5    |                 4.96 |           4.96 |      4.92  |
| sub_decrees |   25 |                4.84 |           4.72 |               4.88 |                 4.84 |           4.96 |      4.834 |
