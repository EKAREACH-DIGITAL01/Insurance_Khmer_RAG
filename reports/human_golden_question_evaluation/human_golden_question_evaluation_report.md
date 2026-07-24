# Golden-Question Evaluation Report

Human validation of the gold question set collected by `scripts/11_human_golden_question_evaluation.py`. Measures the quality of the **questions and target answers** — the ground truth every downstream evaluation is scored against — not the RAG system's answers.

## Coverage

| Metric                              |   Value |
|-------------------------------------|---------|
| Evaluators                          |   10    |
| Gold questions                      |  400    |
| — human-authored                    |  100    |
| — AI-authored                       |  300    |
| Questions evaluated                 |  370    |
| Questions NOT yet evaluated         |   30    |
| Total responses                     |  500    |
| Mean responses / evaluated question |    1.35 |

## Metric means (1–5)

`combined` = mean of the five metric means. AI-authored vs human-authored is the key comparison — it shows whether the auto-generated questions hold up.

| Scope           |   Clarity |   Relevance |   Difficulty |   Answer Quality |   Uniqueness |   Combined |
|-----------------|-----------|-------------|--------------|------------------|--------------|------------|
| All             |     4.098 |       4.098 |        3.968 |            4.066 |        3.742 |      3.994 |
| Human-authored  |     4.076 |       4.053 |        3.924 |            4.115 |        3.664 |      3.966 |
| AI-authored     |     4.106 |       4.114 |        3.984 |            4.049 |        3.77  |      4.005 |
| cat: guidance   |     4.058 |       4.1   |        3.992 |            4.058 |        3.817 |      4.005 |
| cat: law        |     4.15  |       4     |        4.02  |            4.03  |        3.85  |      4.01  |
| cat: prakas     |     4.142 |       4.131 |        3.973 |            4.087 |        3.661 |      3.999 |
| cat: sub_decree |     4.01  |       4.134 |        3.876 |            4.072 |        3.691 |      3.957 |

## Score distribution

| Clarity   |   1 Very Poor |   2 Poor |   3 Average |   4 Good |   5 Excellent |
|-----------|---------------|----------|-------------|----------|---------------|
| count     |            11 |       12 |          63 |      245 |           169 |

| Relevance   |   1 Not Relevant |   2 Slightly |   3 Moderate |   4 Relevant |   5 Highly Relevant |
|-------------|------------------|--------------|--------------|--------------|---------------------|
| count       |                9 |            8 |           73 |          245 |                 165 |

| Difficulty   |   1 Too Easy/Hard |   2 Poor Level |   3 Adequate |   4 Good Level |   5 Perfect Level |
|--------------|-------------------|----------------|--------------|----------------|-------------------|
| count        |                 7 |             15 |          122 |            199 |               157 |

| Answer Quality   |   1 Very Poor |   2 Poor |   3 Adequate |   4 Good |   5 Excellent |
|------------------|---------------|----------|--------------|----------|---------------|
| count            |             8 |       12 |           96 |      207 |           177 |

| Uniqueness   |   1 Very Redundant |   2 Redundant |   3 Somewhat Unique |   4 Unique |   5 Very Unique |
|--------------|--------------------|---------------|---------------------|------------|-----------------|
| count        |                  8 |            48 |                 145 |        163 |             136 |

## Inter-evaluator agreement

Population std-dev per metric across the **130** questions scored by ≥2 evaluators (lower = more agreement; blank = no overlap yet).

| Metric         |   Mean std-dev |
|----------------|----------------|
| Clarity        |          0.446 |
| Relevance      |          0.442 |
| Difficulty     |          0.508 |
| Answer Quality |          0.565 |
| Uniqueness     |          0.619 |

## Curation flags

**60** question(s) fell below a threshold. These are advisory — nothing is dropped or edited automatically.

| Question ID        | Src   |   Comb. |   #Ev | Question                                                | Flags → action                                                                                                                                                                                                                                                                                        |
|--------------------|-------|---------|-------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| guidance_qid_041   | ai    |     1.8 |     1 | តើការបំពានក្រមសីលធម៌ត្រូវបានចែកជាប៉ុន្មានកម្រិត?        | relevance=2 → DROP — off-topic / not an insurance-law question; answer_quality=1 → FIX — target answer needs legal correction; clarity=2 → REWORD — question is unclear/ambiguous; uniqueness=2 → DEDUPE — redundant with another question; difficulty=2 → REVIEW — difficulty is off (too easy/hard) |
| law_qid_007        | human |     2   |     1 | តើមាត្រា៣៣នៃច្បាប់ច្បាប់ស្តីពីពាណិជ្ជកម្មតាមប្រព័ន្ធអេ… | relevance=2 → DROP — off-topic / not an insurance-law question; answer_quality=2 → FIX — target answer needs legal correction; clarity=1 → REWORD — question is unclear/ambiguous; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                          |
| prakas_qid_021     | human |     2   |     1 | ប្រកាសស្តីពីបែបបទ និងនីតិវិធីនៃការធ្វើអធិការកិច្ចធានារ… | relevance=2 → DROP — off-topic / not an insurance-law question; clarity=1 → REWORD — question is unclear/ambiguous; uniqueness=2 → DEDUPE — redundant with another question; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                |
| guidance_qid_051   | ai    |     2.2 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្ទេរហានិភ័យដែលលើសពីកម្រិត… | relevance=2 → DROP — off-topic / not an insurance-law question; answer_quality=2 → FIX — target answer needs legal correction; clarity=2 → REWORD — question is unclear/ambiguous; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                          |
| guidance_qid_052   | ai    |     2.2 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវប្រើប្រាស់ចំណាត់ថ្នាក់ឥណទា… | relevance=1 → DROP — off-topic / not an insurance-law question; clarity=1 → REWORD — question is unclear/ambiguous; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                                                                         |
| sub_decree_qid_074 | ai    |     2.2 |     1 | តើអនុក្រឹត្យនេះ មានវិសាលភាពអនុវត្តចំពោះអង្គភាពអ្វីខ្លះ? | relevance=2 → DROP — off-topic / not an insurance-law question; clarity=1 → REWORD — question is unclear/ambiguous; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                                                                         |
| guidance_qid_027   | ai    |     2.5 |     2 | តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវពន្យល់អ្នកទាមទារសំ… | uniqueness=2 → DEDUPE — redundant with another question; difficulty=1.5 → REVIEW — difficulty is off (too easy/hard)                                                                                                                                                                                  |
| law_qid_003        | human |     2.7 |     2 | តើវិសាលភាពនៃច្បាប់ស្តីពីពាណិជ្ជកម្មតាមប្រព័ន្ធអេឡិចត្រ… | relevance=2 → DROP — off-topic / not an insurance-law question                                                                                                                                                                                                                                        |
| guidance_qid_037   | ai    |     2.8 |     1 | តើបុគ្គលិករបស់ក្រុមហ៊ុនជើងសាធានារ៉ាប់រងដែលមានវិជ្ជាជីវ… | clarity=2 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                    |
| guidance_qid_094   | ai    |     2.8 |     1 | តើសិក្ខាកាមដែលចាត់ទុកថាប្រឡងជាប់វគ្គបណ្តុះបណ្តាលមូលដ្ឋ… | answer_quality=2 → FIX — target answer needs legal correction                                                                                                                                                                                                                                         |
| law_qid_033        | ai    |     2.8 |     1 | តើបុគ្គលណាមួយ មិនត្រូវបង្កើតបង្ខំដោយកម្លាំង ការរំខាន ឬ… | difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                                                                                                                                                                                             |
| law_qid_036        | ai    |     2.8 |     1 | តើច្បាប់ស្តីពីការរៀបចំ និងការប្រព្រឹត្តទៅនៃអាជ្ញាធរសេវ… | difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                                                                                                                                                                                             |
| prakas_qid_069     | ai    |     2.8 |     2 | តើអាណត្តិរបស់អភិបាលឯករាជ្យ ត្រូវមានរយៈពេលប៉ុន្មាន?      | answer_quality=2.5 → FIX — target answer needs legal correction                                                                                                                                                                                                                                       |
| prakas_qid_070     | ai    |     2.8 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រង អាចផ្តល់កម្ចីជើងសារ ឬកម្ចីទៅឱ្យ… | clarity=2 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                    |
| prakas_qid_129     | ai    |     2.8 |     1 | តើស្ថាប័នណាត្រូវបានចាត់តាំងជាអាណត្តិគាហករបស់ក្រសួងសេដ្… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_145     | ai    |     2.8 |     1 | តើគ្រប់ក្រុមហ៊ុនធានារ៉ាប់រង ត្រូវផ្ញើស្ថិតិអ្វីខ្លះមកក… | answer_quality=2 → FIX — target answer needs legal correction                                                                                                                                                                                                                                         |
| guidance_qid_092   | ai    |     2.9 |     2 | តើបុគ្គលដែលមានបំណងចូលបម្រើការងារផ្តល់សេវាលក់ផលិតផលធានា… | answer_quality=2 → FIX — target answer needs legal correction                                                                                                                                                                                                                                         |
| prakas_qid_053     | ai    |     2.9 |     2 | តើក្រុមហ៊ុនធានារ៉ាប់រង និងអន្តរការីធានារ៉ាប់រង មិនត្រូ… | uniqueness=2.5 → DEDUPE — redundant with another question; difficulty=2 → REVIEW — difficulty is off (too easy/hard)                                                                                                                                                                                  |
| sub_decree_qid_025 | human |     2.9 |     2 | ដើម្បីចុះកិច្ចសន្យាធានារ៉ាប់រង អ្នកដែលមានបំណងចុះកិច្ចស… | clarity=1.5 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                  |
| sub_decree_qid_041 | ai    |     3   |     1 | តើឯកសណ្ឋាន សញ្ញាសម្គាល់ និងសញ្ញាសក្តិ ត្រូវបានផ្តល់ជូន… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| guidance_qid_093   | ai    |     3.1 |     2 | តើសមាគមធានារ៉ាប់រងកម្ពុជា ត្រូវរៀបចំអ្វីខ្លះ ទាក់ទងនឹង… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| law_qid_030        | ai    |     3.1 |     2 | តើរាល់បុគ្គល មិនត្រូវសន្យាផ្តល់រង្វាន់អំណោយ រង្វាន់ ឬវ… | answer_quality=2.5 → FIX — target answer needs legal correction                                                                                                                                                                                                                                       |
| prakas_qid_049     | ai    |     3.1 |     2 | តើក្នុងករណីមានការស្នើសុំកែប្រែលក្ខខណ្ឌណាមួយនៃកិច្ចសន្យ… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| sub_decree_qid_016 | human |     3.1 |     2 | តើលក្ខខណ្ឌតម្រូវនៃការកំណត់ដើមទុនអប្បបរមានិងកម្រិតសធនភា… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_042 | ai    |     3.1 |     2 | តើបុគ្គលណាដែលបន្លំស្លៀកពាក់ឯកសណ្ឋាន សញ្ញាសម្គាល់ និងសញ… | answer_quality=2.5 → FIX — target answer needs legal correction                                                                                                                                                                                                                                       |
| guidance_qid_066   | ai    |     3.2 |     1 | ក្រោយពីពិនិត្យ និងវាយតម្លៃលើឯកសារតម្រូវ ន.ធ.ក. (និយ័តក… | answer_quality=2 → FIX — target answer needs legal correction; uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                |
| guidance_qid_099   | ai    |     3.2 |     2 | តើពេលបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រងមុនកាលកំណត់ ម្ចាស់បណ្ណ… | answer_quality=2.5 → FIX — target answer needs legal correction; uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                            |
| prakas_qid_004     | human |     3.2 |     2 | លិខិតស្នើសុំផ្អាកកិច្ចសន្យាធានារ៉ាប់រង ត្រូវបញ្ជាក់និទ… | clarity=2.5 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                  |
| prakas_qid_025     | human |     3.2 |     1 | អ្នកជំនាញគណនបុព្វលាភធានារ៉ាប់រង ត្រូវមានតួនាទី និងភារក… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_026     | ai    |     3.2 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត ក្រុមហ៊ុនធានារ៉ាប់រងទូ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_043     | ai    |     3.2 |     1 | តើអ្នកជម្រះបញ្ជីក្នុងវិស័យធានារ៉ាប់រង ត្រូវផ្តល់ផែនការ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_119     | ai    |     3.2 |     1 | តើអាជ្ញាបណ្ណអាជីវកម្មក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ និងអាយុជ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_054 | ai    |     3.2 |     1 | តើនៅពេលដែលក្រុមហ៊ុនធានារ៉ាប់រងបានធ្វើសំណងរួចហើយ អ្នកត្… | answer_quality=2 → FIX — target answer needs legal correction; clarity=2 → REWORD — question is unclear/ambiguous                                                                                                                                                                                     |
| sub_decree_qid_064 | ai    |     3.2 |     1 | តើអាជ្ញាបណ្ណអាជីវកម្មក្រុមហ៊ុនជើងសាធានារ៉ាប់រងមានប៉ុន្… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_075 | ai    |     3.2 |     2 | តើអគ្គលេខាធិការដ្ឋាន មាននាយកដ្ឋានចំណុះប៉ុន្មាន និងមានអ… | relevance=2.5 → DROP — off-topic / not an insurance-law question; clarity=2.5 → REWORD — question is unclear/ambiguous                                                                                                                                                                                |
| guidance_qid_008   | human |     3.3 |     2 | តើគោលការណ៍ស្តីពីការទទួល និងដោះស្រាយសំណងធានារ៉ាប់រងត្រូ… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| law_qid_027        | ai    |     3.3 |     2 | តើសំណើ ស្នើការ និងកិច្ចសន្យា អាចធ្វើឡើងតាមមធ្យោបាយអ្វី? | answer_quality=2.5 → FIX — target answer needs legal correction; clarity=2.5 → REWORD — question is unclear/ambiguous                                                                                                                                                                                 |
| prakas_qid_003     | human |     3.3 |     2 | ភាគីដែលមានបំណងបន្តកិច្ចសន្យាធានារ៉ាប់រង ត្រូវស្នើសុំដោ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| guidance_qid_100   | ai    |     3.4 |     1 | តើកំណត់សំអាង គឺជាអ្វី និងត្រូវបញ្ជាក់ខ្លឹមសារអប្បបរមាអ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_019     | human |     3.4 |     2 | ប្រកាសស្តីពីវិធាន និងគោលការណ៍ណែនាំនៃការធានារ៉ាប់រងបន្ត… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_023 | human |     3.4 |     1 | បណ្ណសន្យារ៉ាប់រង ត្រូវកំណត់លក្ខខណ្ឌទូទៅនិង/ឬលក្ខខណ្ឌពិ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_049 | ai    |     3.4 |     1 | តើការធានារ៉ាប់រងមានប្រសិទ្ធភាពចាប់ពីពេលណា?              | clarity=2 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                    |
| law_qid_034        | ai    |     3.5 |     2 | តើបុគ្គលណាមួយ មិនត្រូវលក់ទំនិញដែលមានភ្ជាប់សេចក្តីបរិយា… | relevance=2.5 → DROP — off-topic / not an insurance-law question                                                                                                                                                                                                                                      |
| law_qid_051        | ai    |     3.5 |     2 | តើនៅពេលដែលក្រុមហ៊ុនធានារ៉ាប់រងបានធ្វើសំណងរួចហើយ អ្នកត្… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| guidance_qid_004   | human |     3.6 |     2 | ក្នុងករណីដែលការស្នើសុំទាមទារសំណងធានារ៉ាប់រង មិនមានឯកសា… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| guidance_qid_053   | ai    |     3.6 |     1 | តើការក្លែងបន្លំធានារ៉ាប់រង សំដៅដល់អ្វី?                 | relevance=2 → DROP — off-topic / not an insurance-law question                                                                                                                                                                                                                                        |
| law_qid_019        | human |     3.6 |     2 | តើច្បាប់ស្តីពីការធានារ៉ាប់រងគោលដៅអ្វីខ្លះ?              | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| prakas_qid_020     | human |     3.6 |     1 | របាយការណ៍សង្ខេបអំពីការធានារ៉ាប់រងបន្ត ត្រូវមានព័ត៌មានអ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_042     | ai    |     3.6 |     2 | តើអ្នកជម្រះបញ្ជីក្នុងវិស័យធានារ៉ាប់រង ត្រូវតែជានរណា?    | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| prakas_qid_046     | ai    |     3.6 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_058     | ai    |     3.6 |     2 | តើបុគ្គលដែលមិនសុខចិត្តនឹងសេចក្តីសម្រេចដាក់ទណ្ឌកម្ម មាន… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| prakas_qid_060     | ai    |     3.6 |     1 | តើការធានារ៉ាប់រងបន្ត សំដៅដល់អ្វី?                       | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_095     | ai    |     3.6 |     1 | តើបុគ្គលដែលមានបំណងប្រកបអាជីវកម្មជើងសាធានារ៉ាប់រងជាក្រុ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| prakas_qid_149     | ai    |     3.6 |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច ត្រូវបង់កម្រៃអាជ្ញាបណ្ណ… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| sub_decree_qid_026 | ai    |     3.7 |     2 | តើនិយ័តករធានារ៉ាប់រងកម្ពុជា មានសមត្ថកិច្ចអ្វីខ្លះ ទាក់… | uniqueness=2.5 → DEDUPE — redundant with another question                                                                                                                                                                                                                                             |
| guidance_qid_047   | ai    |     3.8 |     1 | តើបណ្ណសន្យារ៉ាប់រងខ្នាតតូច ត្រូវមានមាតិកាអប្បបរមាអ្វីខ… | clarity=1 → REWORD — question is unclear/ambiguous                                                                                                                                                                                                                                                    |
| law_qid_011        | human |     3.8 |     2 | តើសមាគមអ្នកប្រើប្រាស់មានតួនាទីនិងភារកិច្ចដូចម្តេច?      | relevance=2.5 → DROP — off-topic / not an insurance-law question                                                                                                                                                                                                                                      |
| law_qid_023        | human |     3.8 |     1 | ក្រៅពីនិទ្ទេសចាំបាច់ក្នុងកិច្ចសន្យាធានារ៉ាប់រង បណ្ណសន្… | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |
| law_qid_014        | human |     4   |     2 | តើបុគ្គលណាមួយដែលមិនសុខចិត្តនឹងសេចក្តីសម្រេចរបស់គណៈកម្ម… | relevance=2.5 → DROP — off-topic / not an insurance-law question                                                                                                                                                                                                                                      |
| prakas_qid_116     | ai    |     4   |     1 | តើក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ សំដៅដល់អ្វី?                 | uniqueness=2 → DEDUPE — redundant with another question                                                                                                                                                                                                                                               |

## Target-answer corrections (captured, NOT applied)

**2** target answer(s) were corrected by evaluators in the rating UI. They are **not** written back to `data/09_evaluation/*/gold_questions.json` — review these and apply the accepted ones by hand, then re-run `scripts/13_compare_generator_models.py --rescore`.

### `guidance_qid_099`  (ai, edited by Kuysiang)

**Gold (current):**

> ការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រង៖ កំណត់លក្ខខណ្ឌ និងនីតិវិធីនៃការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រង ដោយម្ចាស់បណ្ណសន្យារ៉ាប់រង និងក្នុងករណីមានការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រងមុនកាលកំណត់ បណ្ណសន្យារ៉ាប់រងត្រូវចែងពីសិទ្ធិទទួលបានតម្លៃ៥០%នៃបុព្វលាភធានារ៉ាប់រងដែលនៅសល់ ឬតម្លៃនៃការបោះបង់កិច្ចសន្យាធានារ៉ាប់រងដែលត្រូវកំណត់នៅក្នុងវិញ្ញាបនបត្រធានារ៉ាប់រង

**Corrected:**

> ការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រង៖ កំណត់លក្ខខណ្ឌ និងនីតិវិធីនៃការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រង ដោយម្ចាស់បណ្ណសន្យារ៉ាប់រង និងក្នុងករណីមានការបញ្ចប់កិច្ចសន្យាធានារ៉ាប់រងមុនកាលកំណត់ បណ្ណសន្យារ៉ាប់រងត្រូវចែងពីសិទ្ធិទទួលបានតម្លៃ៩០%នៃបុព្វលាភធានារ៉ាប់រងដែលនៅសល់ ឬតម្លៃនៃការបោះបង់កិច្ចសន្យាធានារ៉ាប់រងដែលត្រូវកំណត់នៅក្នុងវិញ្ញាបនបត្រធានារ៉ាប់រង

### `law_qid_007`  (human, edited by HENG)

**Gold (current):**

> មាត្រា ៣៣ - ការអនុវត្តបទប្បញ្ញត្តិទាក់ទងនឹងកិច្ចការពារអ្នកប្រើប្រាស់បុគ្គលដែលប្រើប្រាស់ទំនាក់ទំនងអេឡិចត្រូនិកក្នុងសកម្មភាពពាណិជ្ជកម្មជាមួយអ្នកប្រើប្រាស់ក៏ត្រូវអនុវត្តតាមបទប្បញ្ញត្តិនានាទាក់ទងនឹងកិច្ចការពារអ្នកប្រើប្រាស់ផងដែរ។

**Corrected:**

> ការអនុវត្តបទប្បញ្ញត្តិទាក់ទងនឹងកិច្ចការពារអ្នកប្រើប្រាស់បុគ្គលដែលប្រើប្រាស់ទំនាក់ទំនងអេឡិចត្រូនិកក្នុងសកម្មភាពពាណិជ្ជកម្មជាមួយអ្នកប្រើប្រាស់ក៏ត្រូវអនុវត្តតាមបទប្បញ្ញត្តិនានាទាក់ទងនឹងកិច្ចការពារអ្នកប្រើប្រាស់ផងដែរ។
