# Khmer Insurance RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Khmer insurance and legal documents.

The system retrieves relevant information from insurance-related laws, sub-decrees, prakas, and guidance documents, then generates grounded answers in Khmer using SeaLLM-v3.

---

## Features

* Khmer-language question answering
* Retrieval-Augmented Generation (RAG)
* Local embedding generation using BGE-M3
* FAISS vector search
* SeaLLM-v3-7B local inference via llama.cpp
* Structured legal document chunking
* Retrieval evaluation framework
* Source-aware answer generation
* Retrieval inspection and debugging
* Local inference using SeaLLM-v3

---

## System Architecture (Will be replaced by Image here)

User Question

↓

BGE-M3 Query Embedding

↓

FAISS Similarity Search

↓

Top-K Relevant Chunks

↓

Context Construction

↓

SeaLLM-v3-7B

↓

Khmer Answer + Sources

---

## Dataset

The knowledge base contains Khmer insurance-related documents collected from official sources.

Document categories:

* Laws
* Prakas
* Sub-Decrees
* Guidance Documents

Total Dataset Statistics:

* Documents: 49
* Chunks: 1,435
* Embedding Model: BAAI/bge-m3
* Vector Store: FAISS IndexFlatIP

---

## Project Structure

├── data
│   ├── 01_source_documents
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 02_page_images
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 03_raw_pages
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 04_cleaned_pages
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 05_merged_documents
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 06_cleaned_documents
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 07_chunked_documents
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 08_evaluation
│   │   ├── guidance
│   │   ├── laws
│   │   ├── prakas
│   │   └── sub_decrees
│   ├── 09_merged_chunked_documents
│   │   ├── guidance_chunks.json
│   │   ├── laws_chunks.json
│   │   ├── prakas_chunks.json
│   │   └── sub_decrees_chunks.json
│   └── document_metadata.json
├── models
│   ├──.hf_cache
│   ├── retriever
│       └── BAAI-BGE-M3
│   └── generator
│       └── SeaLLMs-v3-7B-Chat
├── requirements.txt
├── scripts
│   ├── 01_convert_pdf_to_png.py
│   ├── 02_extract_text_gemini.py
│   ├── 03_clean_pages.py
│   ├── 04_merge_documents.py
│   ├── 05_clean_documents.py
│   ├── 06_chunk_documents.py
│   ├── 07_generate_embeddings.py
│   ├── 08_evaluate_retrieval.py
│   ├── 09_install_model.py
│   ├── 10_merge_chunked_documents.py
│   └── utils.py
└── vector_db
    ├── khmer_insurance.index
    └── khmer_insurance_metadata.json
├── .env
├── README.md
├── app.py
└── requirements.txt

---

## Retrieval Pipeline

### Chunking
### Guidance Documents

Chunking Strategy:

- Heading hierarchy (H1–H4)
- Parent heading inheritance
- Section-level retrieval
### Legal Documents

Chunking Strategy:

- មាត្រា (Articles)
- ប្រការ (Regulations)
- Chapter context
- Section context

### Embedding

Model:

BAAI/bge-m3

Embedding text includes:

* Document Name
* Document Type
* Chapter
* Section
* Legal Unit
* Heading Context
* Chunk Content

### Vector Search

FAISS IndexFlatIP

Cosine similarity is implemented using:

* Normalized embeddings
* Inner product search

---

## Retrieval Evaluation

Retriever performance was evaluated using manually curated gold questions.

Metrics:

- Hit Rate@K
- Recall@K
- MRR@K
- File Hit Rate@K

Evaluation Range:

K = 1–10

Results showed strong retrieval performance across all document categories, with several categories achieving over 90% Hit Rate@1 and over 96% Hit Rate@5.

---

## Models

### Retriever

BAAI/bge-m3

Purpose:

- Query embedding
- Chunk embedding
- Semantic retrieval

### Generator

SeaLLMs-v3-7B-Chat

Purpose:

- Khmer answer generation
- Context-grounded response generation

Inference Engine:

- llama.cpp

Quantization:

- GGUF

---

## Installation

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install local models:

```bash
python scripts/09_install_model.py
```

---

## Run Application

Start llama.cpp server:

```bash
llama-server \
-m models/generator/SeaLLMs-v3-7B-Chat-Q8_0.gguf \
--host 0.0.0.0 \
--port 8080
```

Launch Streamlit:

```bash
streamlit run app.py
```

Open:

http://localhost:8501

---

## Technologies

* Python
* Streamlit
* Sentence Transformers
* BGE-M3
* FAISS
* llama.cpp
* SeaLLM-v3
* NumPy
* llama.cpp
* SeaLLMs-v3
* Hugging Face
* OpenAI-Compatible API

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

# Metadata System

The project uses a centralized metadata registry:

```text
data/document_metadata.json
```

This file stores document-level metadata such as:
- document name
- document type
- release date

Example:

```json
{
  "law_01": {
    "document_name": "ច្បាប់ស្តីពីពាណិជ្ជកម្មតាមប្រព័ន្ធអេឡិចត្រូនិក",
    "document_type": "law",
    "release_date": "2019-11-02",
  }
}
```



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
pillow
pdf2image
sentence-transformers
transformers
huggingface-hub
sentencepiece
accelerate
faiss-cpu
numpy
torch
torchvision
tabulate
openai
```

Note
IF the yellow line appear under package even after installed successfully

Cmd + Shift + P

Run
Python: Restart Language Server
Pylance: Restart Language Server
Developer: Reload Window

---


# Pipeline Execution Order

Run scripts in this exact order.

---

## Step 1

```bash
python scripts/01
```

---

## Step 2

```bash
python scripts/02
```
---

## Step 3 —

```bash
python scripts/03_
```

---

## Author

Sorphea

Khmer Insurance Retrieval-Augmented Generation Chatbot