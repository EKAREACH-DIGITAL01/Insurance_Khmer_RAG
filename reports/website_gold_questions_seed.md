# Website Gold-Question Seed Sets

These seed sets extend the student's original legal gold questions to the IRC website categories.

- Existing student categories kept unchanged: `guidance`, `laws`, `prakas`, `sub_decrees`.
- New website-category questions start at `qid_026`, so they are treated as retrieval/human-evaluation seeds rather than hand-validated generator-core questions.
- Every question points to a real current chunk ID from `vector_db/BAAI-BGE-M3/khmer_insurance_metadata.json`.

| Category | Questions | Notes |
| --- | ---: | --- |
| `announcement` | 0 | no source-rich chunks passed filters; re-scrape article bodies before adding gold questions; scrape is thin and should be human-reviewed first |
| `events` | 0 | no source-rich chunks passed filters; re-scrape article bodies before adding gold questions |
| `home` | 3 | only 3 usable chunks passed filters |
| `insurance_institutions` | 10 | source-grounded seed set |
| `knowledge` | 10 | source-grounded seed set |
| `law_regulation` | 8 | only 8 usable chunks passed filters |
| `news` | 10 | source-grounded seed set |
| `public_service` | 10 | source-grounded seed set |
| `publication` | 10 | source-grounded seed set |
| `social_programs` | 10 | source-grounded seed set |
| `statistics` | 10 | source-grounded seed set |

Before promoting any website seed question into the core generator benchmark, have a reviewer check the source chunk and move it into the `qid_001`-`qid_025` range for that category.
