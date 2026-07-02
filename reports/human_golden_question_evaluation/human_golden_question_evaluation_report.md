# Golden-Question Evaluation Report

Human validation of the gold question set collected by `scripts/11_human_golden_question_evaluation.py`. Measures the quality of the **questions and target answers** — the ground truth every downstream evaluation is scored against — not the RAG system's answers.

## Coverage

| Metric                              |   Value |
|-------------------------------------|---------|
| Evaluators                          |       2 |
| Gold questions                      |     400 |
| — human-authored                    |     100 |
| — AI-authored                       |     300 |
| Questions evaluated                 |       1 |
| Questions NOT yet evaluated         |     399 |
| Total responses                     |       1 |
| Mean responses / evaluated question |       1 |

## Metric means (1–5)

`combined` = mean of the five metric means. AI-authored vs human-authored is the key comparison — it shows whether the auto-generated questions hold up.

| Scope          |   Clarity |   Relevance |   Difficulty |   Answer Quality |   Uniqueness |   Combined |
|----------------|-----------|-------------|--------------|------------------|--------------|------------|
| All            |         1 |           2 |            2 |                2 |            2 |        1.8 |
| Human-authored |         1 |           2 |            2 |                2 |            2 |        1.8 |
| AI-authored    |           |             |              |                  |              |            |
| cat: prakas    |         1 |           2 |            2 |                2 |            2 |        1.8 |

## Score distribution

| Clarity   |   1 Very Poor |   2 Poor |   3 Average |   4 Good |   5 Excellent |
|-----------|---------------|----------|-------------|----------|---------------|
| count     |             1 |        0 |           0 |        0 |             0 |

| Relevance   |   1 Not Relevant |   2 Slightly |   3 Moderate |   4 Relevant |   5 Highly Relevant |
|-------------|------------------|--------------|--------------|--------------|---------------------|
| count       |                0 |            1 |            0 |            0 |                   0 |

| Difficulty   |   1 Too Easy/Hard |   2 Poor Level |   3 Adequate |   4 Good Level |   5 Perfect Level |
|--------------|-------------------|----------------|--------------|----------------|-------------------|
| count        |                 0 |              1 |            0 |              0 |                 0 |

| Answer Quality   |   1 Very Poor |   2 Poor |   3 Adequate |   4 Good |   5 Excellent |
|------------------|---------------|----------|--------------|----------|---------------|
| count            |             0 |        1 |            0 |        0 |             0 |

| Uniqueness   |   1 Very Redundant |   2 Redundant |   3 Somewhat Unique |   4 Unique |   5 Very Unique |
|--------------|--------------------|---------------|---------------------|------------|-----------------|
| count        |                  0 |             1 |                   0 |          0 |               0 |

## Inter-evaluator agreement

Population std-dev per metric across the **0** questions scored by ≥2 evaluators (lower = more agreement; blank = no overlap yet).

| Metric         | Mean std-dev   |
|----------------|----------------|
| Clarity        |                |
| Relevance      |                |
| Difficulty     |                |
| Answer Quality |                |
| Uniqueness     |                |

## Curation flags

**1** question(s) fell below a threshold. These are advisory — nothing is dropped or edited automatically.

| Question ID    | Src   |   Comb. |   #Ev | Question                                                | Flags → action                                                                                                                                                                                                                                                                                        |
|----------------|-------|---------|-------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| prakas_qid_016 | human |     1.8 |     1 | ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ មិនត្រូវជំរុញការលក់ផលិតផលធានា… | relevance=2 → DROP — off-topic / not an insurance-law question; answer_quality=2 → FIX — target answer needs legal correction; clarity=1 → REWORD — question is unclear/ambiguous; uniqueness=2 → DEDUPE — redundant with another question; difficulty=2 → REVIEW — difficulty is off (too easy/hard) |

## Target-answer corrections (captured, NOT applied)

**1** target answer(s) were corrected by evaluators in the rating UI. They are **not** written back to `data/09_evaluation/*/gold_questions.json` — review these and apply the accepted ones by hand, then re-run `scripts/13_compare_generator_models.py --rescore`.

### `prakas_qid_016`  (human, edited by sorphea)

**Gold (current):**

> ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ មិនត្រូវជំរុញការលក់ផលិតផលធានារ៉ាប់រង ដោយអះអាងដោយបំពានចំពោះព័ត៌មានដូចខាងក្រោម៖
> - ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចមានម្ចាស់ភាគហ៊ុនជាបុគ្គលពិសេស និង/ឬបុគ្គលមានកេរ្តិ៍ឈ្មោះ
>
> - បុគ្គលណាមួយបានឧបត្ថម្ភគាំទ្រ បានទទួលស្គាល់ បានបញ្ជាក់ ឬមានសម្ព័ន្ធភាពជាមួយ ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច
> - បុគ្គលពិសេស និង/ឬបុគ្គលមានកេរ្តិ៍ឈ្មោះណាមួយបានទិញផលិតផលធានារ៉ាប់រងនោះ
> - ផលិតផលធានារ៉ាប់រងនោះមានតម្រូវការច្រើន
> - ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ផ្តល់សំណងធានារ៉ាប់រងគ្រប់ករណីទាំងអស់ សម្រាប់ផលិតផលធានារ៉ាប់រងនោះ
> - លក្ខខណ្ឌ និងអត្ថប្រយោជន៍មិនពិតផ្សេងទៀតរបស់ផលិតផលធានារ៉ាប់រង។

**Corrected:**

> ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ មិនត្រូវជំរុញការលក់ផលិតផលធានារ៉ាប់រង ដោយអះអាងដោយបំពានចំពោះព័ត៌មានដូចខាងក្រោម៖
> - ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចមានម្ចាស់ភាគហ៊ុនជាបុគ្គលពិសេស និង/ឬបុគ្គលមានកេរ្តិ៍ឈ្មោះ
> - បុគ្គលណាមួយបានឧបត្ថម្ភគាំទ្រ បានទទួលស្គាល់ បានបញ្ជាក់ ឬមានសម្ព័ន្ធភាពជាមួយ ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច
> - បុគ្គលពិសេស និង/ឬបុគ្គលមានកេរ្តិ៍ឈ្មោះណាមួយបានទិញផលិតផលធានារ៉ាប់រងនោះ
> - ផលិតផលធានារ៉ាប់រងនោះមានតម្រូវការច្រើន
> - ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ផ្តល់សំណងធានារ៉ាប់រងគ្រប់ករណីទាំងអស់ សម្រាប់ផលិតផលធានារ៉ាប់រងនោះ
> - លក្ខខណ្ឌ និងអត្ថប្រយោជន៍មិនពិតផ្សេងទៀតរបស់ផលិតផលធានារ៉ាប់រង។ testingasdffffffffff
> asdf
> asdf
> asdf
> asdf
> asdf
