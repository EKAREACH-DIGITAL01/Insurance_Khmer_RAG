# Data and Evaluation Commands

Run these commands from the project root using the project's Python environment.
Numbered filenames are retained so the student's workflow and existing command
history continue to work. These commands may write datasets, reports, or indexes;
they are separate from the offline tests under `tests/`.

| Stage | Commands | Outputs |
| --- | --- | --- |
| IRC collection | `00_scrape_irc_website.py` | Website cache and source metadata |
| Website PDF OCR | `00c_extract_irc_pdf_text_gemini.py` | Gemini text and extraction progress |
| Original PDF processing | `01_convert_pdf_to_png.py` through `05_clean_documents.py` | Page images, extracted and cleaned documents |
| Category-aware chunking | `06_chunk_documents.py`, `07_merge_chunked_documents.py` | Chunk records and merged corpus |
| Embeddings | `08_generate_embeddings.py` | Model-specific FAISS indexes and metadata |
| Retriever comparisons | `09_compare_embedding_models.py`, `10_compare_retrieval_modes.py` | Comparison reports |
| Human golden-question review | `11_human_golden_question_evaluation.py`, `12_human_golden_question_evaluation_report.py` | Human review database and report |
| Generator evaluation | `13_generate_model_answers.py` through `16_evaluate_safety_gates.py` | Generated answers, judge input, model and safety reports |
| Dataset documentation | `17_generate_data_card.py` | Data card |
| Feedback export | `18_export_chat_feedback.py` | Saved answer reactions |
| Corpus smoke test | `19_rag_smoke_tests.py` | Retrieval and source coverage checks |
| Source organization | `20_reclassify_website_categories.py`, `21_arrange_irc_raw_sources.py` | Category assignments and raw asset layout |
| Quality audit | `22_rag_quality_audit.py` | Corpus quality report |
| Incremental embeddings | `23_rebuild_bge_incremental.py` | Updated BGE index using reusable embeddings |
| Website golden questions | `24_prepare_website_gold_questions.py` | Website-category evaluation questions |

Category definitions live in `khmer_rag/categories.py`; the old helper import is
a compatibility wrapper. Evaluation commands 10, 13, 16, and 19 import shared
runtime helpers. Their old report files remain historical results, not evidence
that the current corpus and generation settings have been reevaluated.
