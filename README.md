Khmer-Rag-Chatbot/
├── .env                             # GEMINI_API_KEY (only for scripts/02)
├── .gitignore
├── README.md
├── app.py                           # The Streamlit chatbot
├── start_chatbot.py                 # Entry point: launches llama-server + app
├── requirements.txt
│
├── assets/
│   └── favicon.png
│
├── data/
│   ├── 01_source_documents/         # Raw input PDFs, by category
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   │
│   ├── 02_page_images/              # PDF pages rendered to PNG (scripts/01)
│   ├── 03_raw_pages/                # Gemini OCR output, one JSON/page (scripts/02)
│   ├── 04_cleaned_pages/            # Page-level cleanup (scripts/03)
│   ├── 05_merged_documents/         # Pages stitched into full docs (scripts/04)
│   ├── 06_cleaned_documents/        # Document-level cleanup (scripts/05)
│   ├── 07_chunked_documents/        # Final chunks for embedding (scripts/06)
│   │
│   ├── 08_evaluation/               # Gold evaluation datasets
│   │   ├── guidance/gold_questions.json
│   │   ├── laws/gold_questions.json
│   │   ├── prakas/gold_questions.json
│   │   └── sub_decrees/gold_questions.json
│   │
│   └── 09_adversarial/              # Safety / refusal test cases
│       ├── recommendation_questions.json
│       ├── pricing_questions.json
│       ├── personal_advice_questions.json
│       └── out_of_domain_questions.json
│
├── reports/
│   ├── retrieval/                   # Output from scripts/08
│   │   ├── retrieval_eval.json
│   │   └── retrieval_eval.md
│   ├── embeddings/                  # Output from scripts/08
│   │   ├── embedding_comparison.json
│   │   └── embedding_comparison.md
│   ├── generator/                   # Output from scripts/10 & 11
│   │   ├── generator_eval_template.txt
│   │   ├── generator_eval_raw.json
│   │   └── hallucination_audit.txt
│   └── data_card/                   # Output from scripts/12
│
├── models/
│   ├── generator/
│   │   └── SeaLLMs-v3-7B-Chat-Q4_K_M.gguf   # (and other quantizations)
│   │
│   └── retriever/
│       ├── .cache/                  # Hugging Face model weights (one copy,
│       │                            # all three models live here via HF cache)
│       ├── BAAI-BGE-M3/
│       │   └── INSTALLED.json       # Marker file only (weights are in .cache/)
│       ├── multilingual-e5-large/
│       │   └── INSTALLED.json
│       └── MiniLM-paraphrase-multilingual-L12-v2/
│           └── INSTALLED.json
│
├── scripts/
│   ├── 01_convert_pdf_to_png.py
│   ├── 02_extract_text_gemini.py
│   ├── 03_clean_pages.py
│   ├── 04_merge_documents.py
│   ├── 05_clean_documents.py
│   ├── 06_chunk_documents.py
│   ├── 07_generate_embeddings.py    # Builds all installed models' indices
│   ├── 08_compare_embedding_models.py    # 3-model embedding comparison
│   ├── 09_compare_retrieval_modes.py     # Hybrid vs. dense retrieval comparison
│   ├── 10_compare_generator_models.py    # SeaLLM quantization comparison
│   ├── 11_evaluate_rag.py                # End-to-end RAG evaluation (RAGAS-style)
│   ├── 12_generate_data_card.py     # Auto-generate corpus statistics
│   └── 13_install_models.py         # Download all models
│
└── vector_db/
    ├── BAAI-BGE-M3/                 # Production index (default)
    │   ├── khmer_insurance.index
    │   ├── khmer_insurance_metadata.json
    │   └── embedding_model_info.json
    ├── multilingual-e5-large/       # Alternative index (comparison only)
    │   ├── khmer_insurance.index
    │   ├── khmer_insurance_metadata.json
    │   └── embedding_model_info.json
    └── MiniLM-paraphrase-multilingual-L12-v2/
        ├── khmer_insurance.index
        ├── khmer_insurance_metadata.json
        └── embedding_model_info.json