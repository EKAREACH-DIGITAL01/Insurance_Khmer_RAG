# Evaluation Methodology

How the Khmer Insurance RAG chatbot is evaluated end-to-end — every evaluation
in the project, what each one measures, the metrics and frameworks used, where
the resulting data is written, and what happens to that data afterwards.

The project runs **six distinct evaluations** plus a final **data card** roll-up.
They fall into two families:

- **Automated evaluations** (scripts 09, 10, 13→14→15, 16, 17) — deterministic,
  embedding-based, or LLM-judged, no human in the loop. Used to *select
  components* (embedding model, retrieval mode, generator model) and to
  *stress-test the safety gates*. The generated answers are scored by an LLM
  judge (13 generates, 14 prepares the judge input, 15 aggregates), not by hand.
- **Human evaluation** (scripts 11→12) — a Streamlit rating UI backed by SQLite,
  where people score the **gold question set** itself (clarity, relevance,
  difficulty, answer quality, uniqueness). Script 12 turns the raw ratings into
  aggregate metrics and curation flags.

The guiding design is a **two-stage funnel**: cheap automated proxies rank and
narrow the field first, and the single most expensive judgement — humans — is
spent on validating the yardstick itself. The gold question set is validated by
humans (stage 11/12) *before* it is trusted as the measuring stick for
everything downstream.

---

## The evaluation pipeline at a glance

| # | Evaluation | Script(s) | Type | What it decides / produces | Reports folder |
|---|------------|-----------|------|----------------------------|----------------|
| 1 | Embedding Model Comparison | `09` | Automated (retrieval) | Best embedding model → **BGE-M3** | `reports/embedding_model_comparison/` |
| 2 | Retrieval Mode Comparison | `10` | Automated (retrieval) | Dense vs BM25 vs Hybrid → **Hybrid (RRF)** | `reports/retrieval_mode_comparison/` |
| 3 | Human Golden-Question Evaluation | `11` → `12` | Human | Validates **quality of the gold set** | `reports/human_golden_question_evaluation/` |
| 4 | Answer Generation | `13` | Automated (generation only) | Produces every model's answers + latency | `reports/generator_model_comparison/` |
| 5 | LLM-as-Judge Scoring | `14` → LLM judge → `15` | LLM (RAGAS + hallucination) | Ranks the models, picks the generator | `reports/generator_model_comparison/` |
| 6 | Safety Gate Evaluation | `16` | Automated (adversarial) | Confirms the safety gates hold | `reports/safety_gate_evaluation/` |
| — | **Data Card** | `17` | Roll-up | One card summarising everything | `reports/data_card/` |

The shared yardstick for evaluations 1, 2, 4 and 5 is the **gold question set** in
`data/09_evaluation/<category>/gold_questions.json` — **400 questions** across four
categories (`guidance`, `law`, `prakas`, `sub_decree`), split into two tiers: a
**core set of 100** (question ids `001–025` per category), each question and
answer written and checked individually, and an **extended set of 300** (ids
above `025`) of more uneven quality. Generator comparison uses the core set only;
retrieval evaluation uses all 400. Each gold entry carries:

```
question_id · question · expected_chunks · context · expected_answer
```

This **400** is the set *before* human curation. Evaluation 3 (below) is a
curation gate run by legal professionals: it **removes** weak questions and lets
experts **rewrite target answers**, so the finalised gold set used by evaluations
1, 2, 4 and 5 is **smaller than 400 and legally vetted**.

Evaluation 6 uses a separate **adversarial set** in
`data/10_adversarial/*.json` (`question_id · question · expected_behavior`).

Website expansion note: the active repository may also include website-category
seed questions under `data/09_evaluation/` with ids starting at `qid_026`.
Retrieval scripts load all present gold-question files, while generator scripts
continue to use only the hand-checked core ids `001–025`. See
`docs/gold_questions_methodology.md` for the current category coverage.

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
`scripts/16`. Its index in `vector_db/BAAI-BGE-M3/` is the retrieval backbone for
every later evaluation.

---

## 2. Retrieval Mode Comparison — `scripts/10_compare_retrieval_modes.py`

**Purpose.** With the embedding model fixed, decide **how** to retrieve: pure
dense (FAISS), pure lexical (BM25), or **hybrid** fusion of the two.

**Framework / method.**
- **Dense** — BGE-M3 cosine over the FAISS index.
- **BM25** — `rank_bm25.BM25Okapi` with a **Khmer-aware tokenizer** that
  character-**n-grams** (n = 3) Khmer-script words (which have no spaces) while
  lower-casing Latin words. Two additions carry the lexical signal that plain
  n-gramming loses: every maximal **digit run** (Khmer ០-៩ or ASCII) is pulled
  out of a Khmer-script token as its own term, so an article number survives
  regardless of spacing; and an explicit **provision reference** ("មាត្រា ១០",
  "ប្រការ ៤") is emitted as one compound token, repeated `ARTICLE_REF_BOOST = 20`
  times so the near-unique match outweighs the n-gram noise from shared
  document-title grams.
- **Hybrid** — **Reciprocal Rank Fusion (RRF)**, `RRF_K = 60`, fusing the dense
  and BM25 rankings.

**Metrics.** The same retrieval family as evaluation 1 — **Hit Rate**, **Recall**,
**MRR**, **File Hit Rate**, at K = 1…10, per category, plus a macro-average at
K = 5 for the headline comparison.

**Output → `reports/retrieval_mode_comparison/`**: `retrieval_mode_comparison.json`
and `.md`.

**What happens next.** The winning mode (**Hybrid / RRF**) is the retrieval path
used by the live `app.py` and reproduced byte-for-byte inside `scripts/10`,
`scripts/13` and `scripts/16`, so both this comparison and the downstream
generator scores reflect the retrieval the client actually gets. The tokenizer
and its constants are duplicated across all four files and must stay identical;
they have drifted twice, and each time the measurement — never production — was
the side that silently diverged.

---

## 3. Human Golden-Question Evaluation — `scripts/11` → `scripts/12`

**Purpose.** Validate the **quality of the gold question set itself** *before* it
is trusted as the measuring stick for everything else. This judges the questions,
their context, and their target answers — **not** any RAG output.

**Collection UI — `scripts/11_human_golden_question_evaluation.py`.** A Streamlit
app (port 8503) backed by SQLite
(`reports/human_golden_question_evaluation/human_golden_question_evaluation.db`).
Each evaluator logs in by name and is assigned a fixed slice of **50** of the
400 questions, drawn least-rated-first so slices spread across the set rather
than overlapping at random. Evaluators can also **edit/correct the target
answer** (shared, versioned, full history kept).

**Collected:** **11 evaluators × 50 = 550 responses** over all **400** gold
questions, with **150** questions rated by two or more evaluators — the base
for the inter-evaluator agreement figures below.

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

## 4. Answer Generation — `scripts/13_generate_model_answers.py`

**Purpose.** Run every installed generator LLM over the **100 core gold
questions** (25 × 4 categories), retrieving context exactly as `app.py`
does, and store each generated answer. This script **does not score or rank
anything** — it only produces the answers that stage 5 judges.

**Parity.** Retrieval, prompting, gating and token limits mirror `app.py`
exactly, so a generated answer here is the answer the live chatbot would have
given. See `docs/generation_robustness_and_parity.md`.

**Data produced.** Every generated answer is written immediately (crash-safe,
resumable) to
`reports/generator_model_comparison/<model>/generated_answer_results.jsonl`, holding the
question, the retrieved `context_chunk_ids`, the gold `expected_answer`, the
`generated_answer` and timing. No scores.

---

## 5. LLM-as-Judge Scoring — `scripts/14` → LLM judge → `scripts/15`

**Purpose.** Score every generated answer on the five RAGAS metrics **plus a
separate hallucination judgement**, using an LLM judge rather than deterministic
proxies. This ranks the four models and selects the shipped generator.

**Why not proxies.** The previous scorer used chunk-ID overlap, numeral overlap
and BGE-M3 cosine similarity. Those are cheap and reproducible but shallow: the
faithfulness proxy could only detect fabricated *numbers*, so a fluent, entirely
invented Khmer paragraph containing no figures scored a perfect 5. Judging the
substance of a legal answer needs a reader, not a regex.

**Preparation — `scripts/14_prepare_llm_judge_input.py`.** Scores
`context_relevance` deterministically as **recall of `expected_chunks` in
`context_chunk_ids`** — an exact set comparison, so it is measured rather than
judged. (Recall, not precision: extra retrieved chunks are cheap, a missing
expected chunk means the answer could not have been grounded.) It then resets
the four judged metrics to `0` and `hallucination` to `""`, both meaning *not
yet judged*, and strips the retrieved chunk text — leaving ~1.6k chars per
record, so one model fits comfortably in a single judging session.

**Judging.** The rubric and output contract are in
`docs/llm_judge_prompt.md`. Scope: 4 models × 100 questions = **400
judgements**, one session per model. The judge must differ from the model being
judged, to avoid self-preference bias, and models are never mixed in one
session — a judge that has just read one model's answers anchors its scale
there.

**Metrics (1–5):** Context Relevance (.20) — measured, not judged —
plus Faithfulness (.25), Answer Correctness (.25), Answer Relevance (.15) and
Completeness (.15), all assigned by the judge comparing `generated_answer`
against `expected_answer`.

**Hallucination** is judged separately — a Yes/No flag with a quoted evidence
span — and is **never averaged into the combined score**.

**Report — `scripts/15_generate_comparison_report.py`.** Aggregates the judged
JSONLs into a per-model `generated_answer_results_summary.{json,md}` and the
cross-model `generator_model_comparison.{json,md}`, which names the winner. It
recomputes `combined` from the five metrics rather than trusting the judge's own
arithmetic, and excludes any answer still scoring 0 (not yet judged) from every
average — reporting the count so a partial run can never look complete.

**Output → `reports/generator_model_comparison/`.**

**Known limitation.** An LLM judge on Khmer legal text is not reliable by
default. The judge should itself be validated against a hand-scored sample
before its ranking is treated as measured rather than asserted.

---

## 6. Safety Gate Evaluation — `scripts/16_evaluate_safety_gates.py`

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

## The final step — Data Card — `scripts/17_generate_data_card.py`

Once all evaluations are complete, the data card **rolls everything up into one
document**, written as both machine-readable JSON and human-readable Markdown.

**It reads (all optional — missing pieces degrade gracefully):**

- `data/document_metadata.json` — source corpus
- `data/08_merged_chunked_documents/*_chunks.json` — chunked knowledge base
- `data/09_evaluation/*/gold_questions.json` — gold evaluation sets
- `data/10_adversarial/*_questions.json` — safety evaluation sets
- `reports/embedding_model_comparison/*.json` — evaluation 1
- `reports/retrieval_mode_comparison/*.json` — evaluation 2
- `reports/generator_model_comparison/*.json` — evaluation 5
- `reports/safety_gate_evaluation/*.json` — evaluation 6
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
        ├─► 13 generate answers ─► generator_model_comparison/<model>/*.jsonl
        │        └─► 14 inline context + reset scores
        │                 └─► LLM judge scores each JSONL in place
        │                          └─► 15 report ─► generator_model_comparison.{json,md}
        │
        └─► 16 safety gates ─► reports/safety_gate_evaluation/
                                          │
   all report JSONs ──────────────────────┴─► 17 data card ─► reports/data_card/
```
