# Llama-3.1-8B-Instruct-Q4_K_M — Answer Summary

Scored by an LLM judge (see `docs/llm_judge_prompt.md`). 100/100 answers judged.

## Quality (1–5)

| Metric                  |   Avg |
|-------------------------|-------|
| Context Relevance       | 4.92  |
| Faithfulness            | 4.55  |
| Answer Relevance        | 4.57  |
| Answer Correctness      | 4.2   |
| Completeness            | 4.24  |
| **Combined (weighted)** | 4.493 |

## Hallucination

- **Flagged:** 7/100 (7%)
- Judged separately; never folded into the combined score.

### By category

| Category    |   Flagged |   Judged | Rate   |
|-------------|-----------|----------|--------|
| guidance    |         5 |       25 | 20%    |
| laws        |         0 |       25 | 0%     |
| prakas      |         1 |       25 | 4%     |
| sub_decrees |         1 |       25 | 4%     |

### Flagged answers

The span quoted is what the judge said was fabricated — check it against the retrieved context before acting on it.

| Question ID        | Category    | Question                                         | Fabricated span                                              |   Faith. |
|--------------------|-------------|--------------------------------------------------|--------------------------------------------------------------|----------|
| guidance_qid_005   | guidance    | ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវឆ្លើយតបទៅអ្នក… | ១៥ (ដប់ប្រាំ) ថ្ងៃ                                           |        2 |
| guidance_qid_009   | guidance    | ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវបង់កម្រៃសេវារ… | ២ ០០០ ០០០ (ពីររយពាន់) រៀល                                    |        1 |
| guidance_qid_014   | guidance    | បុគ្គលិករបស់ក្រុមហ៊ុនជើងសាធានារ៉ាប់រងដែលមានវិជ្… | ការ ដាក់ ពាក្យ បណ្តឹង ទៅ តុលាការ                             |        2 |
| guidance_qid_019   | guidance    | តើបុគ្គលការដាក់ពាក្យស្នើសុំការអនុញ្ញាតសាកល្បងត្… | ប្រការ ៧ - ធនាគារជាតិនៃកម្ពុជានឹងជូនដំណឹងជាលាយលក្ខណ៍ អក្សរ … |        2 |
| guidance_qid_020   | guidance    | ក្នុងករណីមានតម្រូវការពន្យារយៈពេលសាកល្បង បុគ្គលប… | ១០ (ម្ភៃ) ថ្ងៃ                                               |        2 |
| prakas_qid_002     | prakas      | ក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្សព្វផ្សាយរបាយការណ៍ប… | ១៥ (ដប់ប្រាំ) ថ្ងៃ នៃថ្ងៃធ្វើការ                             |        2 |
| sub_decree_qid_005 | sub_decrees | តើការចុះកិច្ចសន្យាធានារ៉ាប់រងខ្នាតតូច ត្រូវតែធ្… | ២០ ០០០ ០០០ (ម្ភៃលាន) រៀល                                     |        2 |

## Coverage & latency

| Metric            |    Value |
|-------------------|----------|
| Blank answers     |   0      |
| Truncated answers |  26      |
| Avg TTFT (s)      |  53.7557 |
| Avg gen time (s)  | 106.252  |
| Avg tokens out    | 420.531  |
| Avg tok/s         |   8.4432 |

## Per category

| Category    |   Qs |   Context Relevance |   Faithfulness |   Answer Relevance |   Answer Correctness |   Completeness |   Combined |
|-------------|------|---------------------|----------------|--------------------|----------------------|----------------|------------|
| guidance    |   25 |                5    |           4    |               4.28 |                 3.76 |           3.88 |      4.164 |
| laws        |   25 |                5    |           4.8  |               4.64 |                 4.36 |           4.36 |      4.64  |
| prakas      |   25 |                4.84 |           4.56 |               4.44 |                 4.04 |           4.04 |      4.39  |
| sub_decrees |   25 |                4.84 |           4.84 |               4.92 |                 4.64 |           4.68 |      4.778 |
