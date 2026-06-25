## 1. Background & Motivation

Cambodia's insurance sector has grown steadily in recent years, with the Insurance Regulator of
Cambodia (IRC) overseeing an expanding base of policyholders across health, life, micro, and
general insurance products. Despite this growth, a significant gap remains: most Cambodian
policyholders — especially those purchasing insurance for the first time — struggle to understand
the terms, coverage, exclusions, and claims processes written in their own policies.

Insurance documents are typically written in formal, legal-style Khmer that ordinary citizens find
difficult to interpret. No accessible, trustworthy, Khmer-language tool currently exists to help
policyholders understand what they have bought, what they are entitled to, and what steps to take
when they need to make a claim.

This project proposes building a Khmer-language chatbot that uses Retrieval-Augmented
Generation (RAG) — a technique that grounds every chatbot response in retrieved source
documents, reducing fabricated answers and making every response traceable to a real document.
The goal is not to replace professional advice, but to close the information gap for ordinary
Cambodian insurance consumers.

**Important Scope Boundary**

▸  This chatbot is an information and financial literacy tool — it does NOT provide personalized
financial advice.
▸  It will NOT recommend specific products, calculate premiums, or assist in submitting actual
claims.
▸  Every response must carry a clear disclaimer directing users to a licensed agent or the IRC for
their specific situation.


## 2. Problem Statement

The following core problems motivate this project:

•  Policy documents are written in complex legal Khmer that most policyholders cannot
interpret without professional help.
•  No Khmer-language AI assistant currently exists to explain insurance concepts and claims
procedures in plain, everyday language.
•  General AI chatbots often produce fabricated or inaccurate insurance information — a
particularly harmful failure in a legal-financial domain.
•  The IRC and insurers have limited capacity to answer individual policyholder questions at
scale.
•  Khmer language processing poses unique technical challenges that the student must learn to
navigate, including the absence of spaces between words and limited training data for AI
models.
•  No publicly available, curated Khmer insurance Q&A dataset exists for research or
evaluation purposes.

This project addresses all these problems by building a grounded, source-cited chatbot from the
ground up — starting with zero pre-existing datasets and constructing everything from publicly
available materials.


## 3. Project Objectives

### 3.1  Primary Objectives

1.  Collect and curate a Khmer-language insurance corpus from publicly available sources —
covering product descriptions, policy terms, claims procedures, regulatory guidelines, and
insurance literacy content.
2.  Design and implement a complete RAG pipeline suitable for the Khmer language, from
document ingestion through to answer generation with source citation.
3.  Build a working chatbot interface that is accessible to Khmer-speaking users, with a
mandatory disclaimer on every response.
4.  Evaluate the chatbot systematically using both automated metrics and human reviewers —
including at least one insurance domain expert.
5.  Produce a reproducible codebase and a well-written technical report that documents
findings, limitations, and recommendations.

### 3.2  Learning Objectives for the Student

•  Understand how RAG systems work conceptually and be able to explain each component.
•  Gain hands-on experience building an NLP pipeline for a low-resource, non-English
language.
•  Learn how to collect, clean, and structure domain-specific data when no dataset exists.
•  Practice responsible AI development: understanding accuracy requirements, ethical
boundaries, and disclaimer obligations in a sensitive domain.
•  Develop skills in system evaluation, technical writing, and presenting findings to non-
technical audiences.


## 4. Domain Scope

The chatbot will focus on four sub-domains within the Cambodian insurance context. These were
selected because they have the highest demand from ordinary consumers and the most accessible
public documentation:

| Sub-Domain | What the Chatbot Should Help Users Understand |
| --- | --- |
| Insurance Products & Types | The difference between life, health, micro, and general insurance; who each type is suitable for; basic coverage principles. |
| Policy Terms & Exclusions | What common policy clauses mean in plain Khmer; what is typically covered and what is excluded; how to read a policy document. |
| Claims Process | Step-by-step guidance on how to prepare and submit a claim; what documents are commonly required; general timelines and expectations. |
| Insurance Financial Literacy | Core concepts such as premium, deductible, beneficiary, sum insured, and policy renewal — explained in simple, everyday Khmer. |

Out of scope for this Project: premium calculation, personalized product recommendation, live
claim submission, actuarial analysis, and connection to any insurer's internal systems. These are
appropriate for a later production phase.


## 5. Data Collection Strategy

One of the most valuable skills this Project will develop is the ability to build a dataset from
nothing. No Khmer insurance dataset currently exists in any public repository. The student will
construct one using the following three approaches:

### Track A — Public Regulatory Sources

Begin with the Insurance Regulator of Cambodia (IRC) and equivalent authoritative bodies. These
organisations publish consumer-facing materials, guidelines, and awareness content — often in
Khmer — that form the most credible foundation for the corpus.
•  Collect all publicly available documents from regulatory websites: guidelines, circulars,
consumer protection information, and awareness campaigns.
•  Translate relevant international insurance literacy resources (e.g., from ASEAN or OECD
bodies) into Khmer where Khmer versions do not exist.
•  Record the source, publication date, and issuing authority for every document.

### Track B — Insurer Public-Facing Materials

Licensed insurance companies operating in Cambodia publish product brochures, FAQ pages, and
claims guides that are publicly available and appropriate for a research corpus.
•  Collect product brochures, FAQ pages, and claims guides from the public websites of
licensed Cambodian insurers.
•  These are educational marketing materials — not confidential contracts — and may be used
for non-commercial research.
•  Document the insurer name, product name, and access date for every item collected.

### Track C — Carefully Controlled Synthetic Data

When real documents are insufficient to cover a topic, the student may generate synthetic Q&A pairs
using a large language model — but with strict controls.
•  Every synthetically generated item must be reviewed and verified by a person with domain
knowledge before it enters the corpus.
•  Synthetic items must be clearly labeled in the dataset. They must never be used as ground
truth in the evaluation set.
•  Treat synthetic data as a supplement, not a substitute for real documents.


## 6. System Design Overview

A RAG chatbot consists of several connected components. The student is expected to understand
each component conceptually, research appropriate tools independently, and document the
technology choices with justification. The following describes what each component must
accomplish — not which specific product to use.

### 6.1  Document Ingestion & Cleaning

Before any AI processing, raw documents must be converted into clean, machine-readable text.
•  The student must handle multiple file formats: web pages, PDFs, scanned images, and plain
text files.
•  Scanned documents require Optical Character Recognition (OCR). The student should
research OCR tools that support the Khmer script.
•  Cleaning steps include removing formatting artifacts, normalising Khmer Unicode
characters, and stripping repeated boilerplate text (such as recurring legal disclaimers).
•  Each document chunk must be stored with metadata describing its origin, language, and
topic.

### 6.2  Text Chunking

Long documents must be divided into smaller pieces (chunks) that a retrieval system can match
against user queries.
•  The student must decide on an appropriate chunk size, balancing context completeness
against retrieval precision.
•  Khmer presents a particular challenge: unlike English, Khmer does not use spaces between
words, so standard text-splitting approaches may cut words in unexpected places. The student
should research Khmer-aware segmentation approaches.
•  Legal clauses must not be split mid-sentence. A numbered clause (e.g., Article 5.2) should
always remain intact within a single chunk.
•  Each chunk should inherit the metadata of its parent document.

### 6.3  Embedding & Vector Search

The core of retrieval is converting text into numerical representations (embeddings) that allow the
system to find the most relevant document chunks for any given query.
•  The student should research and compare at least two multilingual text embedding
approaches, evaluating their performance on Khmer insurance queries using the gold Q&A
pairs collected in Month 1.
•  Embeddings are stored in a vector database that enables fast similarity search at scale.
•  The student should also implement a keyword-based search method alongside vector search,
then combine both. This hybrid approach is important because Khmer insurance
terminology (product names, clause numbers, specific legal phrases) is better matched by
keyword search than by semantic similarity alone.

### 6.4  Retrieval & Context Selection

When a user asks a question, the system retrieves the most relevant chunks from the vector store
and prepares them as context for the language model.
•  The student should retrieve a candidate set of chunks, then apply a re-ranking step to improve
the quality of what is passed to the language model.
•  A confidence threshold should be implemented: if no retrieved chunk is sufficiently relevant,
the system should decline to answer and direct the user elsewhere — rather than attempt a
response based on weak context.
•  The student should document the retrieval parameters chosen and explain the reasoning.

### 6.5  Answer Generation

A large language model (LLM) reads the retrieved context and generates a natural-language
answer in Khmer.
•  The student must design a system prompt that instructs the LLM to: answer only from the
retrieved context; cite the source document; respond in the user's language; and always
append the required disclaimer.
•  The disclaimer must be injected programmatically — it should not be possible for a user to
bypass it through their query.
•  If the retrieved context does not contain sufficient information, the LLM should respond
clearly in Khmer that it cannot answer from available documents, and direct the user to a
licensed agent or the IRC.

### 6.6  User Interface

The student should build a simple, user-facing interface that makes the chatbot accessible to Khmer
speakers.
•  Begin with a simple prototype interface for internal testing (Month 2), then refine into a
cleaner web interface for the final demonstration (Month 3).
•  The interface must display: the chatbot's answer, the source document(s) cited, and the
disclaimer — prominently and on every response.
•  Design with Cambodian users in mind: clear Khmer typography, simple layout, and
accessible on common mobile browsers.


## 7. Key Technical Considerations

The following are areas where the student is expected to conduct independent research and make
justified design decisions. These are not prescriptions — they are questions the student must answer
through experimentation and literature review.

| Area | Question to Investigate | Why It Matters |
| --- | --- | --- |
| Khmer Language Processing | How should Khmer text be segmented and tokenised given the absence of word spaces? | Incorrect segmentation will produce poor chunking and retrieval results. |
| Multilingual Embeddings | Which embedding approach performs best on Khmer insurance queries? How will this be measured? | The embedding model is the foundation of retrieval quality. |
| Hybrid Retrieval | How should semantic search and keyword search be combined and weighted for insurance terminology? | Neither approach alone is sufficient for domain-specific legal vocabulary. |
| Khmer OCR Quality | How accurate is OCR on scanned Khmer insurance brochures? How should low-confidence text be handled? | Poor OCR introduces errors that propagate through the entire pipeline. |
| Hallucination Prevention | How can the system prompt and retrieval threshold prevent the LLM from generating unsupported claims? | Fabricated insurance information can directly mislead or harm users. |
| Disclaimer Enforcement | How can the disclaimer be made technically non-bypassable by user input? | Regulatory and ethical obligation; cannot be optional. |


## 8. 12-Week Work Plan

| Phase | Duration | Key Activities | Deliverables |
| --- | --- | --- | --- |
| Phase 1 — Foundation & Data Collection | Weeks 1–3 | Study RAG concepts and Khmer NLP literature; collect regulatory documents; gather insurer brochures; conduct policyholder interviews; clean and annotate corpus; build evaluation Q&A set. | Curated corpus (400+ chunks); 80 gold Q&A pairs; Data Card document describing the corpus. |
| Phase 2 — Pipeline Construction | Weeks 4–6 | Design and implement the ingestion, chunking, embedding, and retrieval pipeline; compare embedding approaches; evaluate retrieval quality on gold Q&A pairs; build prototype interface. | Working retrieval pipeline; embedding comparison report; initial prototype interface. |
| Phase 3 — Answer Generation & Integration | Weeks 7–8 | Integrate the language model with the retrieval pipeline; design and test the system prompt; implement disclaimer enforcement; tune retrieval threshold; internal testing. | End-to-end chatbot prototype; system prompt documentation; internal test results. |
| Phase 4 — Evaluation | Weeks 9–10 | Run automated evaluation metrics; conduct human evaluation panel (including domain expert); perform hallucination audit; document failure modes and limitations. | Automated evaluation report; human evaluation results; hallucination audit log; failure analysis. |
| Phase 5 — Finalisation & Presentation | Weeks 11–12 | Refine interface; fix identified issues; complete documentation; write technical report; record demo video; final presentation to supervisor and stakeholders. | Final chatbot; open codebase; technical report; demo video; presentation slides. |


## 9. Evaluation Framework

Evaluation must be planned from the beginning of the project, not added at the end. The corpus
and Q&A pairs collected in Phase 1 are specifically designed to enable rigorous testing in Phase 4.

### 9.1  Retrieval Quality

The student must measure how well the system finds the right documents for a given question.
•  Does the correct source document appear in the top results for each test question?
•  How high does the correct document rank on average?
•  Are the top retrieved chunks actually relevant to the question asked?
•  The student should define target thresholds for each metric before running experiments, then
report whether they were met.

### 9.2  Answer Quality

The student must measure whether the generated answers are accurate, grounded, and helpful.
•  Faithfulness: Is every claim in the answer supported by the retrieved context? This is the
most critical metric given the insurance domain.
•  Relevance: Does the answer actually address the question?
•  Hallucination audit: A sample of answers must be manually checked by the supervisor
against source documents before any external demonstration.
•  Refusal quality: When the system cannot answer, does it decline clearly and helpfully in
Khmer?

### 9.3  Human Evaluation

Automated metrics alone are not sufficient. The student must conduct a structured human review.
•  Panel composition: at least one insurance domain expert (an agent, an IRC staff member, or
a compliance professional), at least one Khmer language evaluator, and at least two
ordinary policyholders or potential insurance buyers.
•  Each evaluator reviews a sample of 10–15 answers using a simple scoring rubric covering:
accuracy, clarity of Khmer language, usefulness of cited sources, and appropriateness of
the disclaimer.
•  All test questions used in human evaluation must come from real policyholder interviews —
not from synthetic sources.

### 9.4  Insurance-Specific Checks

•  Coverage accuracy: Does the system correctly distinguish between what IS covered and
what is NOT, without confusing the two?
•  Terminology: Are Khmer insurance terms used correctly and consistently?
•  Mixed language: Many real Cambodian insurance documents blend Khmer and English.
Can the system handle queries that do the same?
