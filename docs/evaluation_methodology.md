# Evaluation Methodology

How the Khmer Insurance RAG chatbot is evaluated end-to-end — every evaluation
in the project, what each one measures, the metrics and frameworks used, where
the resulting data is written, and what happens to that data afterwards.

The project runs **six distinct evaluations** plus a final **data card** roll-up.
They fall into two families:

- **Automated evaluations** (scripts 09, 10, 13, 18) — deterministic or
  embedding-based proxies, no human in the loop. Used to *select components*
  (embedding model, retrieval mode, generator model) and to *stress-test the
  safety gates*.
- **Human evaluations** (scripts 11→12, 16→17) — a Streamlit rating UI backed
  by SQLite, where people score either the **gold question set** or the
  **winning model's answers**. Each has a companion report script that turns the
  raw ratings into aggregate metrics.

The guiding design is a **two-stage funnel**: cheap automated proxies rank and
narrow the field first; expensive human judgement is spent only on the single
winning model. The gold question set itself is validated by humans (stage 11/12)
*before* it is trusted as the yardstick for everything downstream.

---

## The evaluation pipeline at a glance

| # | Evaluation | Script(s) | Type | What it decides / produces | Reports folder |
|---|------------|-----------|------|----------------------------|----------------|
| 1 | Embedding Model Comparison | `09` | Automated (retrieval) | Best embedding model → **BGE-M3** | `reports/embedding_model_comparison/` |
| 2 | Retrieval Mode Comparison | `10` | Automated (retrieval) | Dense vs BM25 vs Hybrid → **Hybrid (RRF)** | `reports/retrieval_mode_comparison/` |
| 3 | Human Golden-Question Evaluation | `11` → `12` | Human | Validates **quality of the gold set** | `reports/human_golden_question_evaluation/` |
| 4 | Generator Model Comparison | `13` → `14` → `15` | Automated (proxy RAGAS) | Best generator model → **SeaLLM** | `reports/generator_model_comparison/` |
| 5 | Safety Gate Evaluation | `18` | Automated (adversarial) | Confirms the safety gates hold | `reports/safety_gate_evaluation/` |
| — | **Data Card** | `19` | Roll-up | One card summarising everything | `reports/data_card/` |

The shared yardstick for evaluations 1, 2 and 4 is the **gold question set** in
`data/09_evaluation/<category>/gold_questions.json` — **400 questions** across four
categories (`guidance`, `law`, `prakas`, `sub_decree`), of which **100 are
human-authored** (question ids `001–025` per category) and **300 are AI-authored**
(ids above `025`). Each gold entry carries:

```
question_id · question · expected_chunks · context · expected_answer
```

This **400** is the set *before* human curation. Evaluation 3 (below) is a
curation gate run by legal professionals: it **removes** weak questions and lets
experts **rewrite target answers**, so the finalised gold set used by evaluations
1, 2, 4 and 5 is **smaller than 400 and legally vetted**.

Evaluation 5 uses a separate **adversarial set** in
`data/10_adversarial/*.json` (`question_id · question · expected_behavior`).

---

## 1. Embedding Model Comparison — `scripts/09_compare_embedding_models.py`

**Purpose.** Choose the multilingual embedding model that retrieves Khmer
insurance passages best. Deliberately a **dense-only** comparison — no BM25, no
fusion — so a score difference can only come from the embedding model itself,
not a lexical layer riding alongside it.

**How it runs.** It does *not* re-encode the corpus. It loads each model's
**precomputed** FAISS index + metadata from `vector_db/<model_dir>/` (built once
by `scripts/08_generate_embeddings.py`) and only encodes the few hundred gold
**questions** fresh, as queries, at evaluation time.

**Metrics** (all computed at K = 1…10, per category, using `expected_chunks` as
ground truth):

- **Hit Rate @ K** — fraction of questions where at least one relevant chunk is
  in the top-K.
- **Recall @ K** — proportion of the relevant chunks retrieved within top-K.
- **MRR @ K** — mean reciprocal rank of the first relevant chunk (ranking
  quality).
- **File Hit Rate @ K** — at least one chunk from the *correct source document*
  was retrieved (a looser, document-level signal).

**Output → `reports/embedding_model_comparison/`**: `embedding_model_comparison.json`
(machine-readable) and `.md` (human-readable table).

**What happens next.** The winning model (**BGE-M3**, `BAAI/bge-m3`) becomes the
active embedding model wired into `app.py`, `scripts/10`, `scripts/13` and
`scripts/18`. Its index in `vector_db/BAAI-BGE-M3/` is the retrieval backbone for
every later evaluation.

---

## 2. Retrieval Mode Comparison — `scripts/10_compare_retrieval_modes.py`

**Purpose.** With the embedding model fixed, decide **how** to retrieve: pure
dense (FAISS), pure lexical (BM25), or **hybrid** fusion of the two.

**Framework / method.**
- **Dense** — BGE-M3 cosine over the FAISS index.
- **BM25** — `rank_bm25.BM25Okapi` with a **Khmer-aware tokenizer** that
  character-**n-grams** (n = 3) Khmer-script words (which have no spaces) while
  lower-casing Latin words.
- **Hybrid** — **Reciprocal Rank Fusion (RRF)**, `RRF_K = 60`, fusing the dense
  and BM25 rankings.

**Metrics.** The same retrieval family as evaluation 1 — **Hit Rate**, **Recall**,
**MRR**, **File Hit Rate**, at K = 1…10, per category, plus a macro-average at
K = 5 for the headline comparison.

**Output → `reports/retrieval_mode_comparison/`**: `retrieval_mode_comparison.json`
and `.md`.

**What happens next.** The winning mode (**Hybrid / RRF**) is the retrieval path
used by the live `app.py` and reproduced byte-for-byte inside `scripts/13` and
`scripts/18`, so downstream generator scores reflect the retrieval the client
actually gets.

---

## 3. Human Golden-Question Evaluation — `scripts/11` → `scripts/12`

**Purpose.** Validate the **quality of the gold question set itself** *before* it
is trusted as the measuring stick for everything else. This judges the questions,
their context, and their target answers — **not** any RAG output.

**Collection UI — `scripts/11_human_golden_question_evaluation.py`.** A Streamlit
app (port 8503) backed by SQLite
(`reports/human_golden_question_evaluation/human_golden_question_evaluation.db`).
Each evaluator logs in by name and is assigned a **coverage-aware slice of 50**
of the 400 questions: the app always hands out the least-assigned questions
first, so the first 8 evaluators cover all 400 with no gaps and later evaluators
add a redundancy layer. Evaluators can also **edit/correct the target answer**
(shared, versioned, full history kept).

**Metrics** (each scored 1–5):

- **Clarity** — how clear and understandable the question is.
- **Relevance** — how relevant it is to insurance.
- **Difficulty** — whether it is appropriately challenging for the RAG system.
- **Answer Quality** — how good the target answer is.
- **Uniqueness** — how non-redundant the question is.

**Report — `scripts/12_human_golden_question_evaluation_report.py`.** Reads the
SQLite ratings + target-answer corrections and produces (a) aggregate test-set
quality metrics and (b) a per-question **curation flag list** a maintainer acts
on by hand (e.g. low-scoring questions to revise or drop).

**Output → `reports/human_golden_question_evaluation/`**:
`human_golden_question_evaluation_report.json` and `.md` (the SQLite `.db` is the
raw store).

**The gold set is not final at 400.** This stage is a **curation gate**, not just
a scoring pass. Because the evaluators here include **legal professionals**, the
400-question set is expected to **shrink** and **change** as a result:

- **Questions are removed.** Questions that score poorly (unclear, irrelevant,
  redundant, or wrong difficulty) are **dropped** by hand, so the finalised gold
  set ends up **smaller than 400** — the exact count depends on how many the
  evaluators cull.
- **Target answers are corrected.** The `expected_answer` for a kept question can
  be **rewritten** in the UI (shared, versioned) when a legal-domain evaluator
  judges the original target answer inaccurate or incomplete. The corrected
  answer, not the original, becomes the ground truth.

**What happens next.** These edits are applied back into
`data/09_evaluation/*/gold_questions.json`, producing the **finalised, reduced
gold set** — fewer questions, with expert-revised target answers. Every
downstream evaluation that uses the gold set as its yardstick (evaluations 1, 2,
4 and 5) is measured against this curated set, **not** the raw 400. So the
"400 questions" figure describes the *input* to this stage; the *output* is a
trimmed, legally-vetted set.

---

## 4. Generator Model Comparison — `scripts/13` → `scripts/14` → `scripts/15`

**Purpose.** Rank all installed generator LLMs against each other on the gold
set and pick a single winner to carry into human evaluation. **Fully automated,
no LLM judge and no human scoring** — deterministic and embedding-based proxies
only. Run over the **100 human-authored gold questions** (25 × 4 categories) as
the clean, trusted subset (the shipped model, SeaLLM, is additionally scored on
the full set).

**Metrics (proxy RAGAS)** — each mapped to a 1–5 score:

| Metric | How it is computed | Weight |
|--------|--------------------|--------|
| **Context Relevance** | Exact chunk-ID overlap between `expected_chunks` and the chunks actually sent to the LLM (deterministic, not a proxy). | 0.20 |
| **Faithfulness** | Arabic + Khmer **numeral overlap** between answer and retrieved context — catches fabricated dates, fees, article numbers, the highest-risk hallucination in a legal domain. | 0.25 |
| **Answer Correctness** | **BGE-M3 cosine** between generated answer and `expected_answer` (mirrors RAGAS Answer Semantic Similarity). Own calibrated thresholds. | 0.25 |
| **Answer Relevance** | BGE-M3 cosine between generated answer and the **question**. Own threshold set (structurally lower ceiling than correctness). | 0.15 |
| **Completeness** | Ends with Khmer sentence-final punctuation (។ / ៕) **and** length ratio vs expected within a floor (truncation) and ceiling (padding). | 0.15 |

Weights encode **legal-domain priorities** (fabricated numbers are actively
harmful → faithfulness weighted high). A **blank answer is not a free pass**:
similarity is forced to 0.0 → score 1, never excluded from the average.
Thresholds are calibrated to this corpus's real score distribution — see
`docs/scoring_calibration.md`.

**Companion scripts.**
- `14_diagnose_generator_scores.py` — surfaces *why* scores came out as they did
  (distribution diagnostics), written to `generator_score_diagnostics.{json,md}`.
- `15_rescore_generator_scores.py` — re-judges answers **from disk** (`--rescore`)
  after a metric/threshold change, without re-generating (delegates to script 13's
  scorer). This is what makes calibration cheap.

**Data produced.** Every generated answer is written immediately (crash-safe,
resumable) to
`reports/generator_model_comparison/<model>/generated_answer_results.jsonl`.

**Output → `reports/generator_model_comparison/`**:
`generator_model_comparison.{json,md}` (cross-model table) and
`generator_score_diagnostics.{json,md}`.

**What happens next.** The winning model (**SeaLLM**, `SeaLLMs-v3-7B-Chat`) is the
generator shipped in `app.py`, and it is the model the safety-gate run
(script 18) replays the adversarial set against.

---

## 5. Safety Gate Evaluation — `scripts/18_evaluate_safety_gates.py`

**Purpose.** Verify that the **three layered safety gates** in `app.py` correctly
enforce scope boundaries under creative/adversarial queries — run against the
winning model.

**Test set — `data/10_adversarial/`.** Four files deliberately mixing cases the
bot **must refuse** with legitimate cases it **must not wrongly block**:
`out_of_domain_questions.json`, `personal_advice_questions.json`,
`pricing_questions.json`, `recommendation_questions.json`
(`question_id · question · expected_behavior`).

**The three gates (evaluated in order, matching `app.py`):**
1. **Gate 1** — Khmer-script check + minimum length + deterministic
   scope-violation regex. Fires before any embedding/retrieval (zero cost).
2. **Gate 2** — FAISS score gate (`OUT_OF_DOMAIN_THRESHOLD = 0.49`). Fires after
   retrieval, before generation (costs one embed + one search).
3. **Gate 3** — model behaviour: did it emit the canned refusal or a real
   answer? Only reached if Gates 1 and 2 both passed (costs a full generation).

**Metrics — a confusion matrix over the two directions**, with **which gate
fired** recorded per question:

- **True Refuse (TR)** — a "refuse" case correctly blocked.
- **False Answer (FA)** — a "refuse" case wrongly answered (a gate miss).
- **True Answer (TA)** — an "answer" case correctly answered.
- **False Refuse (FR)** — an "answer" case wrongly blocked (a false positive).

**Output → `reports/safety_gate_evaluation/`**: `safety_gate_evaluation.{json,md}`
plus per-question `safety_gate_results.jsonl` (records which gate caught each
refusal).

**What happens next.** Feeds the data card's safety section; misses/false-positives
drive gate tuning in `app.py`.

---

## The final step — Data Card — `scripts/19_generate_data_card.py`

Once all evaluations are complete, the data card **rolls everything up into one
document**, written as both machine-readable JSON and human-readable Markdown.

**It reads (all optional — missing pieces degrade gracefully):**

- `data/document_metadata.json` — source corpus
- `data/08_merged_chunked_documents/*_chunks.json` — chunked knowledge base
- `data/09_evaluation/*/gold_questions.json` — gold evaluation sets
- `data/10_adversarial/*_questions.json` — safety evaluation sets
- `reports/embedding_model_comparison/*.json` — evaluation 1
- `reports/retrieval_mode_comparison/*.json` — evaluation 2
- `reports/generator_model_comparison/*.json` — evaluation 4
- `reports/safety_gate_evaluation/*.json` — evaluation 5
- `vector_db/*/embedding_model_info.json` — embedding metadata

**Output → `reports/data_card/`**: `data_card.json` and `data_card.md`.

This is the single artifact that describes the corpus, the knowledge base, the
evaluation sets, the model-selection experiments and the quality/safety results
together — the project's summary deliverable.

---

## End-to-end data flow

```
data/09_evaluation/  (gold set, 400 Qs)      data/10_adversarial/ (safety set)
        │                                             │
        ├─► 09 embedding comparison ─► reports/embedding_model_comparison/
        ├─► 10 retrieval comparison ─► reports/retrieval_mode_comparison/
        │
        ├─► 11 human gold-Q UI (SQLite) ─► 12 report
        │        └─► reports/human_golden_question_evaluation/   (curate gold set ↺)
        │
        ├─► 13 generator comparison ─► reports/generator_model_comparison/<model>/*.jsonl
        │        ├─► 14 diagnostics    └─► generator_model_comparison.{json,md}
        │        └─► 15 rescore (from disk)
        │
        └─► 18 safety gates ─► reports/safety_gate_evaluation/
                                          │
   all report JSONs ──────────────────────┴─► 19 data card ─► reports/data_card/
```
