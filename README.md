# Khmer Insurance RAG Chatbot

Khmer legal and insurance Retrieval-Augmented Generation (RAG) chatbot using:

* Gemini Vision OCR
* Khmer semantic retrieval
* FAISS vector search
* Streamlit frontend

This project processes Khmer legal PDFs into a searchable AI knowledge system.

---

# Current Features

* PDF → image conversion
* Khmer OCR extraction using Gemini
* Khmer text cleaning
* document chunking
* multilingual embeddings
* FAISS semantic retrieval
* Streamlit chatbot UI
* grounded RAG answering
* source inspection and retrieval debugging

---

# Tech Stack

* Python 3.11
* Gemini API
* Sentence Transformers
* FAISS
* Streamlit
* Pillow
* NumPy

---

# Recommended Python Version

Python 3.11 is recommended for:
- FAISS compatibility
- stable dependency installation
- Streamlit support
- sentence-transformers compatibility

---

# Project Structure

```text
Khmer-Rag-Insurance-Chatbot/
│
├── scripts/
│   ├── 01_convert_pdf_to_images.py
│   ├── 02_extract_text_gemini.py
│   ├── 03_clean_text.py
│   ├── 04_chunk_documents.py
│   ├── 05_generate_embeddings.py
│   └── utils.py
│
├── data/
│   │
│   ├── source_documents/
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   │
│   ├── page_images/
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   │
│   ├── evaluation/
│   │
│   ├── extracted_text/
│   │   │
│   │   ├── raw/
│   │   │   ├── guidance/
│   │   │   ├── laws/
│   │   │   ├── prakas/
│   │   │   └── sub_decrees/
│   │   │
│   │   └── cleaned/
│   │       ├── guidance/
│   │       ├── laws/
│   │       ├── prakas/
│   │       └── sub_decrees/
│   │
│   ├── chunked_documents/
│   │   ├── guidance/
│   │   ├── laws/
│   │   ├── prakas/
│   │   └── sub_decrees/
│   │
│   └── document_metadata.json
│
├── vector_db/
│   ├── *.index
│   └── *_metadata.json
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

# Metadata System

The project uses a centralized metadata registry:

```text
data/document_metadata.json
```

This file stores document-level metadata such as:
- document name
- document type
- document release date
- document language

Example:

```json
{
  "law_01": {
    "document_name": "ច្បាប់ស្តីពីពាណិជ្ជកម្មតាមប្រព័ន្ធអេឡិចត្រូនិក",
    "document_type": "law",
    "release_date": "2019-11-02",
    "language": "kh-en"
  }
}
```

Metadata is later attached to chunked documents for:
- retrieval context
- source attribution
- citations
- future filtering support

---

# Folder Explanations

## `scripts/`

Contains the complete data processing pipeline.

---

## `01_convert_pdf_to_images.py`

Converts PDFs into page images (`.png`) for OCR processing.

Uses:

* `pdf2image`
* 400 DPI rendering for better Khmer OCR quality

---

## `02_extract_text_gemini.py`

Performs OCR extraction using Gemini Vision.

Features:

* Khmer + English extraction
* retry logic
* resumable processing
* automatic skip for completed pages
* strict OCR prompting

Output:

* raw JSON OCR files

---

## `03_clean_text.py`

Cleans OCR text while preserving Khmer legal structure.

Cleaning includes:

* whitespace normalization
* line break cleanup
* formatting cleanup

Minimal cleaning is intentionally used to avoid damaging Khmer legal text.

---

## `04_chunk_documents.py`

Splits cleaned text into smaller RAG chunks.

Features:

* Khmer-safe chunking
* paragraph-based splitting
* metadata preservation

Each chunk contains:

* document ID
* page number
* chunk ID
* text

---

## `05_generate_embeddings.py`

Generates multilingual embeddings and creates FAISS vector databases.

Uses:

* `paraphrase-multilingual-MiniLM-L12-v2`
* FAISS vector indexing

Output:

* `.index`
* metadata JSON

---

## `utils.py`

Stores shared helper functions used across multiple scripts.

This file is reserved for reusable utilities such as:

* retry handling
* JSON save/load helpers
* folder creation helpers
* runtime formatting
* OCR helper functions
* logging utilities

Utilities are added gradually as the project grows to avoid unnecessary early abstraction.

The goal of `utils.py` is to improve:

* code reusability
* maintainability
* cleaner pipeline scripts

---

# `data/`

Main project data directory.

---

## `data/source_documents/`

Stores original Khmer legal PDFs.

Categories:

* laws
* sub_decrees
* prakas
* guidance

These files should never be modified.

---

## `data/page_images/`

Stores generated page images.

Example:

```text
page_images/laws/law_01/page_001.png
```

Used for:

* OCR extraction
* debugging
* manual verification

---

## `data/evaluation/`

Stores evaluation datasets, testing logs, and quality validation files.

### Purpose

- OCR quality validation
- retrieval quality testing
- hallucination detection
- RAG answer evaluation
- semantic search benchmarking

This folder is used to systematically improve:

- OCR accuracy
- retrieval precision
- grounded answer quality

### Example Evaluation Areas

- Khmer OCR comparison
- expected retrieval results
- chatbot answer validation
- hallucination analysis

The evaluation system is expanded gradually as the project matures.

---

## `data/extracted_text/raw/`

Stores raw OCR output from Gemini.

No cleaning is applied here.

Purpose:

* preserve original OCR
* OCR debugging
* reprocessing

---

## `data/extracted_text/cleaned/`

Stores cleaned OCR text.

This version is used for chunking and embeddings.

---

## `data/chunked_documents/`

Stores chunked RAG-ready documents.

Each file contains structured chunks with metadata.

---

# `vector_db/`

Stores FAISS vector databases and metadata.

Example:

```text
vector_db/
├── law_01.index
└── law_01_metadata.json
```

Used for:

* semantic retrieval
* RAG search
* chatbot grounding

---

# Environment Setup

## 1. Clone Repository

Run from any directory where you want the project folder to be created.

```bash
git clone https://github.com/sorphea7/Khmer-Rag-Insurance-Chatbot.git
cd Khmer-Rag-Insurance-Chatbot
```

---

## 2. Create Virtual Environment

Run from the project root directory.

```bash
python3.11 -m venv .venv
```

---

## 3. Activate Virtual Environment

This project uses:
- VSCode
- zsh / oh-my-zsh terminal
- WSL for Windows users

Use the following command inside the VSCode terminal:

```bash
source .venv/bin/activate
```

If successful, terminal should display:

```text
(.venv)
```

---

## 4. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 5. Install Poppler

Poppler is required for:
- `pdf2image`
- PDF page conversion

### macOS

```bash
brew install poppler
```

### Ubuntu / WSL

```bash
sudo apt install poppler-utils
```

---

## 6. Install Python Dependencies

Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Create `.env`

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

---

# requirements.txt

Current dependencies:

```text
streamlit
google-genai
python-dotenv
Pillow
sentence-transformers
faiss-cpu
numpy
torch
transformers
huggingface-hub
pdf2image
```

---

# Pipeline Workflow

```text
PDF Documents
      ↓
Convert PDF Pages → Images
      ↓
Gemini OCR Extraction
      ↓
Clean Khmer Text
      ↓
Chunk Documents
      ↓
Generate Embeddings
      ↓
Store in FAISS Vector Database
      ↓
RAG Chatbot Retrieval
```

---

# Pipeline Execution Order

Run scripts in this exact order.

---

## Step 1 — Convert PDFs to Images

```bash
python scripts/01_convert_pdf_to_images.py
```

---

## Step 2 — OCR Extraction

```bash
python scripts/02_extract_text_gemini.py
```

This is the slowest stage because it uses Gemini Vision OCR.

Features:

* retry logic
* resumable processing
* skip completed pages

---

## Step 3 — Clean OCR Text

```bash
python scripts/03_clean_text.py
```

---

## Step 4 — Chunk Documents

```bash
python scripts/04_chunk_documents.py
```

---

## Step 5 — Generate Embeddings

```bash
python scripts/05_generate_embeddings.py
```

---

# Run Chatbot UI

Start the Streamlit frontend:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Features:

* Khmer question answering
* semantic retrieval
* retrieved source inspection
* grounded AI responses
* RAG debugging

---

# Current Scope

This project currently focuses only on:

* Khmer legal documents
* insurance regulations
* image-based PDFs
* Khmer semantic retrieval
* Gemini OCR pipeline

Document categories:

* Laws
* Sub Decrees
* Prakas
* Guidance

---

# Notes

* OCR quality depends heavily on PDF quality.
* 400 DPI is currently used for better Khmer OCR performance.
* Gemini Vision OCR is the primary bottleneck of the pipeline.
* Streamlit frontend replaces terminal-based retrieval testing.

---

# Future Improvements

Planned improvements:

* multi-document retrieval
* citation highlighting
* article-level chunking
* chat memory
* reranking
* OCR evaluation tools
* better Khmer embeddings
* production deployment

IF the yellow line appear under package even after installed successfully

Cmd + Shift + P

Run
Python: Restart Language Server
Pylance: Restart Language Server
Developer: Reload Window