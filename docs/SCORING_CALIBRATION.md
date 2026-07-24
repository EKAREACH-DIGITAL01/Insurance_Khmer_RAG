# Generator Evaluation: Scoring Calibration Notes

This document explains the 1–5 scoring of each automated proxy metric in
`scripts/13_compare_generator_models.py` — where the thresholds and the metric
*logic* came from, why they're set where they are, and their limitations. Keep
it alongside that script.

This is **Stage 1** of a two-stage generator evaluation:

1. **Stage 1 — fully programmable (this doc).** Score all installed generator
   models on the gold question set with deterministic + embedding proxies, no
   human and no LLM judge. Pick one winner. The expensive part is *generating*
   the answers once; re-scoring them is free (see Rescore workflow below).
2. **Human validation is applied to the gold set, not to the answers.**
   `scripts/11` → `scripts/12` has people validate the *questions and target
   answers* that this proxy scores against. The generated answers themselves
   are not human-scored, so true *answer relevance* and human-grade judgement
   are outside what this pipeline measures.

Companion tools:
- **`scripts/14_diagnose_generator_scores.py`** — runs every
  percentile/distribution check referenced below against the live
  `reports/generator_model_comparison/<model>/generated_answer_results.jsonl`
  files and writes `generator_score_diagnostics.{json,md}`.
- **`scripts/15_rescore_generator_scores.py`** — thin wrapper that delegates to
  `scripts/13 --rescore` (the canonical rescorer).

Re-run the diagnostics any time the corpus, embedding model, or generator
lineup changes — every number here is a snapshot against THIS corpus's real
data (4 models × 100 human-authored gold questions), not a universal constant.

---

## Rescore workflow (iterate on scoring for free)

The answers are generated once (the slow, ~hours-long part). Every scoring
change after that is applied **without regenerating** anything:

```bash
# 1. edit a metric function in scripts/13_compare_generator_models.py
python scripts/13_compare_generator_models.py --rescore   # re-judge from disk
python scripts/14_diagnose_generator_scores.py            # refresh diagnostics
```

`--rescore` reloads each model's JSONL, **rebuilds each answer's retrieval
context from its saved `context_chunk_ids`** (so faithfulness sees exactly the
context the answer was generated against), recomputes every metric with the
current functions, and rewrites the JSONL + summaries + final comparison
report. It never starts llama-server. This is why all the numbers below can be
trusted to reflect the *current* code, not a stale generation run.

> **Why not score from the stored `detail` values?** An earlier rescorer
> re-bucketed the cosine similarities and faithfulness counts already stored in
> `detail`. That can only change *thresholds*, never the *measurement*. The
> moment faithfulness started normalising digit forms and excluding list
> ordinals (below), re-bucketing stored counts became wrong — so the canonical
> rescorer re-reads the answer text and context and recomputes from scratch.

---

## Weights

```python
WEIGHTS = {
    "context_relevance":  0.20,
    "faithfulness":       0.25,
    "answer_correctness": 0.40,
    "completeness":       0.15,
}
```

`answer_correctness` (semantic match to the **verbatim gold answer**) carries
the most weight: on this corpus the gold answers are exact passages, so
similarity-to-gold is the single strongest programmable signal of whether the
model answered correctly. `answer_relevance` is **not weighted** — see below.

---

## Why `answer_relevance` was removed from the score

`answer_relevance` compares the **answer to the question** (cosine). A question
and its answer are different grammatical forms, so this similarity is
structurally capped — across all 400 real answers the **maximum ever observed
was ~0.90, with medians at ~0.77**. Under any reasonable threshold a 5 is
nearly unreachable, and the metric penalised answers that were *byte-identical
to the gold answer* simply because the answer doesn't lexically resemble the
question. Example: `prakas_qid_024` (SeaLLM) was near-identical to gold
(`answer_correctness` cosine 0.999) yet `answer_relevance` dragged its combined
score down to 3.9.

Its job — "does this actually address the question" — is covered far better by
`answer_correctness` here (match to the gold answer). So the similarity is still computed
into `detail.answer_relevance_similarity` for traceability, but it carries **no
weight**. Its 0.15 was folded into `answer_correctness`.

---

## Metric 1: Context Relevance — deterministic, no calibration

```python
recall = len(found_chunks) / len(expected_chunks)
score  = round(1 + recall * 4)
```

Exact chunk-ID overlap between the retrieved `CONTEXT_K` chunks and
`expected_chunks`. Generator-independent, so it should be ~equal across models;
a model that differs here signals a retrieval/ordering issue, not a generation
difference. It clusters at 1 and 5 because most questions have a single
expected chunk (near-binary hit/miss).

| Model        | avg |
|--------------|-----|
| SeaLLM v3 7B | 4.88 |
| Qwen 3.5 9B  | 4.80 |
| Sailor2-L 8B | 4.92 |
| Llama 3.1 8B | 4.92 |

A perfect generation can still score low *overall* when the gold chunk simply
wasn't retrieved — e.g. SeaLLM's worst answer (`prakas_qid_007`) is
verbatim-perfect (`answer_correctness`=5) but `context_relevance`=1, so its
combined is 4.2. That's correct behaviour: the retrieval miss is real and
model-independent.

---

## Metric 2: Faithfulness — number grounding, with normalisation

Checks what fraction of the **fact-numbers** in the answer also appear in the
retrieved context. Numbers (dates, fees, counts, article refs) are the
highest-risk hallucination in a legal domain, and two texts can be semantically
close while disagreeing on a figure — which is why this is a number check, not
a cosine.

```python
no fact-numbers          -> 5     # nothing to fabricate = faithful
fraction == 1.0          -> 5
fraction >= 0.75         -> 4
fraction >= 0.50         -> 3
fraction  > 0            -> 2
else                     -> 1
```

Three corrections were made after reading real answers (not just the script):

1. **No-numbers scores 5, not a neutral 4.** The old neutral-4 rule
   *systematically penalised the cleanest model.* SeaLLM writes concise prose:
   70/100 answers contain no figures and **100 % of the figures it does write
   are grounded** — yet under neutral-4 its faithfulness averaged only ~4.3,
   *below* a wordier model that padded in many (grounded) list/citation
   numbers. Absence of numbers is not a fabrication; it is maximally faithful
   on the only axis this proxy measures. This single change moved SeaLLM from a
   false near-tie back to a clear #1.

2. **Digit-form / number-word equivalence.** Arabic digits are normalised to
   Khmer, and a spelled-out Khmer cardinal in the context grounds its digit
   form in the answer. Without this, `guidance_qid_017` (Llama) — which wrote
   "៣ types" where the source said "បី (three) types" — was flagged as a
   fabricated `៣` and scored faithfulness **1** despite being near-verbatim
   correct. It now scores 5.

3. **List-ordinal exclusion.** A line-leading ordinal marker (`៣.`, `3)`, or
   bulleted `- ៤.`) is the model's own list numbering, not a fact from the
   source, so it is not counted. The leading-bullet case mattered: the smaller
   models routinely emit `- ៣. …` items.

What it still *correctly* catches: genuine fabricated figures (e.g. Sailor
inventing `(មាត្រា ៨)` article citations not in its context), and degenerate
number-dumps from broken Llama generations.

**Current evidence:**

| Model        | fact-numbers | grounded | fraction | zero-number answers | faith avg |
|--------------|--------------|----------|----------|---------------------|-----------|
| SeaLLM v3 7B | 45           | 45       | 1.000    | 70                  | 5.00 |
| Qwen 3.5 9B  | 226          | 226      | 1.000    | 35                  | 5.00 |
| Sailor2-L 8B | 450          | 438      | 0.973    | 1                   | 4.88 |
| Llama 3.1 8B | 280          | 272      | 0.971    | 23                  | 4.91 |

---

## Metric 3: Answer Correctness — cosine vs. the gold answer

Compares the generated answer to `expected_answer` (both full statements, same
grammatical role), BGE-M3 cosine.

```python
ANSWER_CORRECTNESS_THRESHOLDS = [
    (0.90, 5), (0.80, 4), (0.70, 3), (0.60, 2), (0.50, 1), (0.00, 1),
]
```

Floor at 0.50 mirrors `OUT_OF_DOMAIN_THRESHOLD` in `app.py` (0.49) — every gold
question is in-domain by construction, so no genuine answer should fall below
it; `(0.00, 1)` catches blanks (clamped to 0.0).

This metric is the discriminator and it behaves: the low-scoring cases are
*genuinely* worse answers, verified by reading them —

- Llama `guidance_qid_005` says "**15 days**" where gold says "**10 days**" → a
  real factual error → correctly low.
- Llama `guidance_qid_021` is degenerate repetition (`​ ។ ​ ។ ​ ។`) → score 1.
- Sailor/Qwen answers that wrap the gold content in citation framing or
  markdown score 3–4 — the embellishment lowers similarity, which is the right
  place to register a style/quality gap (it is **not** punished again under
  completeness; see below).

**Similarity percentiles and final distribution:**

| Model        | min    | p50    | p90    | max    | corr avg |
|--------------|--------|--------|--------|--------|----------|
| SeaLLM v3 7B | 0.8848 | 0.9943 | 1.0000 | 1.0000 | 4.99 |
| Qwen 3.5 9B  | 0.7824 | 0.9490 | 0.9989 | 1.0000 | 4.78 |
| Sailor2-L 8B | 0.7961 | 0.9177 | 0.9662 | 0.9991 | 4.67 |
| Llama 3.1 8B | 0.3736 | 0.8730 | 0.9594 | 0.9991 | 4.17 |

---

## Metric 4: Completeness — truncation, not concision

Two signals: (1) does the answer end on a real Khmer sentence-final mark
(។ / ៕), and (2) a length-ratio **floor** vs. the gold answer.

```python
if ends_cleanly:
    if length_ratio >= 0.80:  score = 5
    elif ratio >= 0.60:       score = 4
    elif ratio >= 0.40:       score = 3
    else:                     score = 2
else:  # truncated / blank
    if length_ratio >= 0.80:  score = 3
    elif ratio >= 0.40:       score = 2
    else:                     score = 1   # blank: ratio = 0.0
```

Two corrections, both driven by reading the JSONL:

1. **No over-length ceiling.** The old design penalised a complete, correct
   answer purely for being wordier than the terse gold passage — double-counting
   against `answer_correctness`, which is where verbosity/padding belongs (extra
   words that are wrong lower the cosine; extra words that are right shouldn't be
   penalised at all). Verbosity is not incompleteness. Genuine rambling still
   shows up: it usually runs past the token cap and gets cut **mid-word with no
   final mark**, which the `ends_cleanly` branch catches (e.g. Qwen's 32
   truncated answers, p50 length-ratio 1.49, max 9.2).

2. **Decoration-aware `ends_cleanly`.** A last-character check produced massive
   *false* truncations: the smaller models routinely finish a complete sentence
   with `។` and then append decoration — `(បញ្ចប់ ៕)`, `(ចប់)`, a literal `(។)`,
   markdown `**`, or a chunk-id citation `(… sub_decree_01_chunk_37)`.
   `ends_with_sentence_final()` strips trailing markdown/whitespace and balanced
   trailing bracket groups, then tests for the sentence-final mark. This dropped
   Sailor's "truncated" count from 31 → 4 (its genuine truncations remain).
   Subtlety worth keeping: closing brackets are handled **only** by the bracket
   regex — putting `)` in the decoration class strips it before the bracket
   group can be matched, leaving an orphan `(` and mislabelling a complete
   answer as truncated.

**Current evidence:**

| Model        | ends_cleanly | length-ratio p50 | max    | comp avg |
|--------------|--------------|------------------|--------|----------|
| SeaLLM v3 7B | 100/100      | 1.00             | 3.20   | 4.92 |
| Qwen 3.5 9B  | 68/100       | 1.49             | 9.25   | 4.36 |
| Sailor2-L 8B | 96/100       | 1.18             | 5.14   | 4.79 |
| Llama 3.1 8B | 74/100       | 1.08             | 5.98   | 3.99 |

---

## Final weighted combined scores

| Model        | Ctx Rel | Faith | Ans Corr | Complete | Combined | Truncated |
|--------------|---------|-------|----------|----------|----------|-----------|
| **SeaLLM v3 7B** | 4.88 | 5.00 | 4.99 | 4.92 | **4.960** | 0  |
| Sailor2-L 8B | 4.92    | 4.88  | 4.67     | 4.79     | **4.791** | 4  |
| Qwen 3.5 9B  | 4.80    | 5.00  | 4.78     | 4.36     | **4.776** | 32 |
| Llama 3.1 8B | 4.92    | 4.91  | 4.17     | 3.99     | **4.478** | 26 |

**Recommendation: SeaLLM v3 7B** — clear #1 (gap 0.17), highest answer
correctness (4.99), 100 % number grounding, zero truncated answers, no
fabricated citations. The ranking matches what reading the answers shows:
SeaLLM produces clean, verbatim-faithful Khmer; the others embellish (markdown,
invented citations, Arabic numerals), ramble past the token cap, or degenerate.

Combined-score sanity (discrimination, not collapse): SeaLLM mean 4.96 (89/100
≥4.99); Sailor 4.79; Qwen 4.78; Llama 4.48 (min 2.95).

---

## Cross-checks against retrieval evaluation

- `scripts/09_compare_embedding_models.py` confirmed BGE-M3 (Hit@5 ≈ 0.97) is
  the retriever, so Context Relevance isn't being dragged down by a weak one.
- `scripts/10_compare_retrieval_modes.py` confirmed hybrid retrieval
  (Recall@5 ≈ 0.99) underneath `scripts/13`, which is why Context Relevance
  clusters tightly at 1 and 5.

---

## Calibration history

| Stage      | Change |
|------------|--------|
| Initial    | 5 weighted metrics incl. `answer_relevance`; faithfulness neutral-4 for no-numbers; completeness floor **and** 2.0× ceiling; last-character `ends_cleanly`. |
| Revision A | Removed `answer_relevance` from the weighted score (structurally capped; penalised gold-identical answers). Reweighted `answer_correctness` 0.25 → 0.40. |
| Revision B | Faithfulness: Arabic/Khmer digit + Khmer number-word normalisation; list-ordinal (incl. leading-bullet) exclusion. |
| Revision C | Completeness: removed the over-length ceiling (verbosity ≠ incompleteness); decoration-aware `ends_with_sentence_final()`. |
| Revision D | Faithfulness no-numbers 4 → **5** (no number = nothing to fabricate). Restored the cleanest model (SeaLLM) to a clear #1. |
| Tooling    | `scripts/13 --rescore` is the canonical rescorer (recomputes from text + rebuilt context). `scripts/15` delegates to it. `scripts/14` model lineup fixed (Sailor2 replaces the retired Gemma slot). |

---

## The honest limitation

**These thresholds are empirically derived from real data and are not
validated against human judgement of the generated answers.** This project
human-validates the *gold set* (`scripts/11` → `scripts/12`) but does not
human-score the model's answers, so there is no human baseline to confirm that
these proxy scores track what a reader would say. Treat the generator ranking
as a defensible automated comparison, not as a measured answer-quality result.
The fixes in this revision were all made by reading the actual generated vs.
expected answers in the JSONL — the metrics agree with what a reader sees on
the cases inspected by hand, which is the strongest claim available here. If
the ranking ever needs to be defended more strongly, use
`scripts/14_diagnose_generator_scores.py` to re-examine the score spread.
