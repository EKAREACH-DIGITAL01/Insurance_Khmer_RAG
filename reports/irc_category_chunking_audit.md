# IRC Website Corpus Category and Chunking Audit

Updated: 2026-08-26

## Summary

The IRC website data is not one uniform dataset. The same crawler can collect
short HTML pages, logo directories, service checklists, Excel/PDF statistics,
and long annual-report PDFs. Treating all of those as generic paragraphs makes
retrieval unstable, especially for Khmer numeric questions.

The 2025 annual report PDF
`website_file_db047ce07365__92c7b84f-41fb-4327-b8c0-f3e63d05e72a.pdf__92c7b84f-41fb-4327-b8c0-f3e63d05e72a.pdf`
was already extracted by Gemini OCR. The failure was not missing OCR; it was
weak publication chunking and category placement. The key fact now lives in a
dedicated section chunk:

- Document ID: `website_file_db047ce07365`
- Chunk ID: `website_file_db047ce07365_chunk_13`
- Heading path: `ផ្នែកទី៤៖ ទីផ្សារធានារ៉ាប់រងកម្ពុជា > ៤.១. គ្រឹះស្ថានធានារ៉ាប់រង > ៤.១.១. ក្រុមហ៊ុនធានារ៉ាប់រង`
- Fact: ឆ្នាំ២០២៥ មានក្រុមហ៊ុនធានារ៉ាប់រងសរុប ៣៩, ក្នុងនោះអាយុជីវិត ១៣, ទូទៅ ១៨, ខ្នាតតូច ៧, និងធានារ៉ាប់រងបន្ត ១។

## Category Decisions

| Category | Source shape | Chunking strategy |
| --- | --- | --- |
| `laws` | Long legal PDFs | Legal-unit chunks by `មាត្រា` |
| `sub_decrees` | Long legal PDFs | Legal-unit chunks by `មាត្រា` |
| `prakas` | Long legal PDFs | Legal-unit chunks by `ប្រការ` |
| `guidance` | Heading-heavy guidance PDFs | Heading hierarchy chunks |
| `statistics` | List/table-like market data and yearly/monthly files | Compact table windows, keep numbers close to labels |
| `publication` | Annual reports, strategic plans, glossary, publication page | Section-aware chunks using headings such as `ផ្នែកទី`, `៤.១`, `៤.១.១`; English duplicate publication PDFs are kept in raw data but skipped from the Khmer index |
| `public_service` | Public-service page and downloadable forms/procedures | Service-step/list chunks with small max size |
| `insurance_institutions` | Logo/name directory pages | Institution directory chunks, preserving company names as records |
| `home` | Homepage blocks, management structure, highlighted stats | Homepage block chunks plus curated high-value facts |
| `knowledge` | Educational articles | Article-level chunks |
| `news`, `events`, `announcement`, `social_programs` | Short article pages | One to few article chunks |
| `law_regulation` | Website law/regulation listing pages and linked website files | Website document windows; original 49 student legal PDFs remain in their own categories |

## Publication PDF Handling

There are 9 PDFs in `data/01_source_documents/publication`. The Khmer chatbot
index now uses the 6 Khmer publication PDFs and skips the 3 English duplicate
versions during chunking:

- skipped: `website_file_646db1aa65d3` - English annual report 2021
- skipped: `website_file_723cfd5541c1` - English strategic development plan 2025-2030
- skipped: `website_file_ea0a6e39a7f9` - English annual report 2025

The raw PDFs and Gemini OCR page JSONs are still kept locally for refreshes.

## Current Indexed Corpus

The active BGE-M3 vector DB now contains:

- Total chunks: 3,092
- Total documents: 854
- Website-derived chunks: 1,657
- Publication chunks: 480
- Public-service chunks: 179
- Insurance-institutions chunks: 22
- Student legal/guidance chunks preserved: 1,435

## Validation

Passed checks:

- `scripts/19_rag_smoke_tests.py`
- `scripts/22_rag_quality_audit.py`

The smoke test now includes the demo question:

`តើមានគ្រឹះស្ថានធានារ៉ាប់រងប៉ុន្មាននៅប្រទេសកម្ពុជា?`

Expected source: `website_file_db047ce07365`

Expected values: total `៣៩`, life `១៣`, general `១៨`, microinsurance `៧`,
reinsurance `១`.
