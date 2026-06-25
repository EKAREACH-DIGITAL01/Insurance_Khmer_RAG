# Generator Evaluation: Scoring Calibration Notes

This document explains where the 1-5 scoring thresholds for each
automated proxy metric in `scripts/10_compare_generator_models.py` came
from, why they're set where they are, and what their limitations are.
Keep this alongside that script and `scripts/12_rescore_generator_scores.py`
so the reasoning doesn't get lost the next time these need adjusting.

Companion tool: **`scripts/11_diagnose_generator_scores.py`** runs every
percentile/distribution check referenced below against the live
`reports/generator_model_comparison/<model>/generated_answer_results.jsonl`
files and writes the results to:
- `reports/generator_model_comparison/generator_score_diagnostics.json`
- `reports/generator_model_comparison/generator_score_diagnostics.md`

Re-run it any time the corpus, embedding model, or generator lineup
changes, since every threshold here is a snapshot calibration against
THIS corpus's real data, not a universal constant.

---

## Blank Answer Policy

**A blank `generated_answer` is NOT a free pass.**

When a model produces no output (zero tokens, empty string), the
affected metrics are scored as follows:

| Metric             | Treatment           | Stored value                          |
|--------------------|---------------------|---------------------------------------|
| Answer Correctness | score = **1**       | `similarity = 0.0` (clamped)          |
| Answer Relevance   | score = **1**       | `similarity = 0.0` (clamped)          |
| Completeness       | score = **1**       | `ends_cleanly=False`, `ratio=0.0`     |
| Faithfulness       | score = **4**       | `total_numbers=0` → neutral           |
| Context Relevance  | unchanged           | deterministic, unaffected by output   |

**Why 1, not None?**

`None` is reserved for metrics that are *genuinely uncomputable* — the
only case in this project is `answer_correctness` when `expected_answer`
is absent from the gold set. That is a data-availability problem, not a
model failure.

A blank answer is a computable outcome: the model definitively failed to
respond. Setting similarity = 0.0 → score = 1 is the correct
representation. Using `None` instead caused `weighted_combined()` to
normalise by the sum of weights that *had* values, which excluded two
0.25-weight metrics (answer_correctness + answer_relevance) from the
denominator for Gemma's 7 blank entries — artificially inflating its
combined score by roughly 0.05 points per blank answer.

**`detail.blank_answer`** is written as `True` for every blank entry so
`scripts/12_rescore_generator_scores.py` can identify them reliably on
future rescore passes without re-inspecting `generated_answer`.

**Blank answer counts (full 4-model run):**

| Model         | Blank / Total |
|---------------|---------------|
| SeaLLM v3 7B  | 0 / 100       |
| Qwen 3.5 9B   | 0 / 100       |
| Gemma 4 E4B   | 7 / 100       |
| Llama 3.1 8B  | 0 / 100       |

Gemma's 7 blank answers are distributed across all four categories
(guidance: 2, laws: 2, prakas: 1, sub_decrees: 2), indicating a
model-level failure mode rather than a category-specific one. Gemma
spent the full generation window (generation_time ≈ 43–54s, tokens_out=0)
on these entries — the server accepted the prompt but produced no tokens.

---

## The core idea: thresholds are calibrated to THIS corpus's real data

There is no fixed, universal mapping from "cosine similarity X" to
"quality score Y." Cosine similarity's meaning depends entirely on:

- which embedding model produced the vectors (BGE-M3 here),
- what KIND of text is being compared (two answers? an answer and its
  question?),
- and what the actual population of real text pairs looks like.

So instead of guessing a generic scale, every threshold below was set by
looking at the **real distribution of raw measurements** this corpus
actually produced across a full 4-model × 100-question run (400 real
generated answers total), and placing band boundaries so the real data
spreads across the 1-5 scale instead of collapsing into one or two
buckets.

---

## Metric 1: Context Relevance — no calibration needed, CONFIRMED stable

```python
score_context_relevance(context_chunk_ids, expected_chunks):
    recall = len(found) / len(expected_chunks)
    return round(1 + recall * 4)
```

Deterministic, not a similarity proxy. No calibration needed because
recall is already a clean, meaningful 0.0–1.0 fraction.

**Confirmed by real evidence** — score distribution across all 4 models:

| Model         | Score 1 | Score 3 | Score 5 |
|---------------|---------|---------|---------|
| SeaLLM v3 7B  | 11      | 4       | 85      |
| Qwen 3.5 9B   | 12      | 4       | 84      |
| Gemma 4 E4B   | 12      | 4       | 84      |
| Llama 3.1 8B  | 12      | 4       | 84      |

Nearly identical across all 4 models — exactly as expected since
retrieval is generator-independent. The distribution clusters at 1 and
5 because this corpus mostly has 1 expected chunk per question, making
retrieval close to binary hit/miss at `CONTEXT_K=5`.

---

## Metric 2: Faithfulness — CONFIRMED working correctly

```python
fraction = grounded_numbers / total_numbers_in_answer
5 if fraction == 1.0, 4 if >=0.75, 3 if >=0.50, 2 if >0, else 1
# total_numbers == 0 → score = 4 (neutral, not automatic 5)
```

Not a cosine-similarity metric — checks whether every Arabic/Khmer
numeral in the generated answer can be found somewhere in the retrieved
context. Chosen over embedding similarity specifically because two texts
can be semantically close while disagreeing on a specific fact.

**Real evidence from the 400-answer run:**

| Model         | n   | Total Numbers | Grounded | Fraction | Zero-Number Answers |
|---------------|-----|--------------|---------|---------|---------------------|
| SeaLLM v3 7B  | 100 | 113           | 111     | 98.2%   | 46                  |
| Qwen 3.5 9B   | 100 | 418           | 301     | 72.0%   | 13                  |
| Gemma 4 E4B   | 100 | 116           | 107     | 92.2%   | 52                  |
| Llama 3.1 8B  | 100 | 145           | 145     | 100.0%  | 50                  |

Qwen writes **2.9–4.3x more numbers per answer** than the other three
models, and only grounds 72% of them — roughly 117 numbers across
Qwen's 100 answers were not found anywhere in the context it was given.
This is NOT a length artifact (the scoring is fraction-based per
answer, already normalized) — it's a real, substantial difference in
hallucination behavior.

Gemma's numbers (n=100, 92.2% grounded) reflect the full 100 answers
including blank entries. Blank answers have `total_numbers=0`, which
correctly scores as 4 (neutral — there's nothing to verify), not 1.
The blank-answer penalty is applied via answer_correctness and
answer_relevance instead.

**Thresholds confirmed as-is. No changes needed.**

---

## Metric 3: Answer Correctness — calibrated against ITS OWN distribution

Compares: **generated answer vs. `expected_answer`** — both full
statements answering the same question, same grammatical role.

**Evidence** (percentiles of `answer_correctness_similarity`, n=100
for all models including blank entries at 0.0):

| Model         | min    | p10    | p25    | p50    | p75    | p90    | max    |
|---------------|--------|--------|--------|--------|--------|--------|--------|
| SeaLLM v3 7B  | 0.6264 | 0.8077 | 0.8861 | 0.9416 | 0.9767 | 0.9947 | 1.0000 |
| Qwen 3.5 9B   | 0.4718 | 0.7922 | 0.8615 | 0.9047 | 0.9349 | 0.9631 | 0.9861 |
| Gemma 4 E4B   | 0.0000 | 0.5188 | 0.7898 | 0.9225 | 0.9668 | 0.9951 | 1.0000 |
| Llama 3.1 8B  | 0.6208 | 0.8029 | 0.8650 | 0.9213 | 0.9682 | 0.9946 | 1.0000 |

Gemma's min=0.0000 and p10=0.5188 reflect the 7 blank entries (clamped
to 0.0) pulling the low end down. The 93 non-blank Gemma answers have
a distribution closely matching SeaLLM and Llama (median ≈ 0.93),
confirming the blank entries are a distinct failure mode, not a general
quality problem.

### Current thresholds

```python
ANSWER_CORRECTNESS_THRESHOLDS = [
    (0.90, 5),
    (0.80, 4),
    (0.70, 3),
    (0.60, 2),
    (0.50, 1),
    (0.00, 1),  # defensive floor
]
```

Floor at 0.50 reflects `OUT_OF_DOMAIN_THRESHOLD` in `app.py` (0.49)
— every gold question is in-domain by construction, so no non-blank
answer should legitimately fall below 0.50. The `(0.00, 1)` catch-all
covers blank answers (similarity = 0.0) and the small number of real
low-similarity values observed near 0.47.

**Final score distribution:**

| Model         | 1  | 2 | 3 | 4  | 5  | avg  |
|---------------|----|---|---|----|----|------|
| SeaLLM v3 7B  | 0  | 1 | 5 | 23 | 71 | 4.64 |
| Qwen 3.5 9B   | 1  | 2 | 8 | 38 | 51 | 4.36 |
| Gemma 4 E4B   | 13 | 5 | 8 | 16 | 58 | 4.01 |
| Llama 3.1 8B  | 0  | 3 | 6 | 31 | 60 | 4.48 |

Gemma's 13 score-1 entries = 7 blanks + 6 genuine low-similarity
answers (similarities in the 0.47–0.60 range). The floor is correctly
catching both cases. **Thresholds confirmed as-is. No changes needed.**

---

## Metric 4: Answer Relevance — a DIFFERENT scale than Answer Correctness

Compares: **generated answer vs. the ORIGINAL QUESTION** — structurally
different content than the Answer Correctness comparison, with a lower
similarity ceiling.

**Evidence** (percentiles of `answer_relevance_similarity`, n=100
for all models including blank entries at 0.0):

| Model         | min    | p10    | p25    | p50    | p75    | p90    | max    |
|---------------|--------|--------|--------|--------|--------|--------|--------|
| SeaLLM v3 7B  | 0.6853 | 0.7328 | 0.7518 | 0.7767 | 0.8155 | 0.8370 | 0.9037 |
| Qwen 3.5 9B   | 0.3489 | 0.7056 | 0.7502 | 0.7711 | 0.7920 | 0.8236 | 0.8753 |
| Gemma 4 E4B   | 0.0000 | 0.4126 | 0.6997 | 0.7484 | 0.7845 | 0.8158 | 0.8741 |
| Llama 3.1 8B  | 0.6771 | 0.7207 | 0.7504 | 0.7721 | 0.8095 | 0.8380 | 0.8741 |

**The max observed across all 400 real answers, every model combined,
was 0.9037** — a single SeaLLM outlier. The real p90 sits at 0.82–0.84
for non-blank answers. Sharing Answer Correctness's thresholds (5
requires ≥0.90) would make a 5 nearly impossible for any model, which
is a measurement ceiling mismatch, not a quality signal.

Gemma's min=0.0000 and p10=0.4126 again reflect the 7 blank entries.
The 93 non-blank Gemma answers have p50=0.7484, consistent with the
other models.

### Current thresholds

```python
ANSWER_RELEVANCE_THRESHOLDS = [
    (0.82, 5),
    (0.75, 4),
    (0.65, 3),
    (0.55, 2),
    (0.00, 1),
]
```

**Final score distribution:**

| Model         | 1  | 2 | 3  | 4  | 5  | avg  |
|---------------|----|----|----|----|----|----|
| SeaLLM v3 7B  | 0  | 0  | 22 | 55 | 23 | 4.01 |
| Qwen 3.5 9B   | 1  | 0  | 23 | 63 | 13 | 3.87 |
| Gemma 4 E4B   | 14 | 5  | 31 | 41 | 9  | 3.26 |
| Llama 3.1 8B  | 0  | 0  | 24 | 58 | 18 | 3.94 |

Gemma's 14 score-1 entries = 7 blanks + 7 non-blank answers with
relevance similarity below 0.55 (note: the 0.55 floor catches more
non-blank Gemma answers here than Answer Correctness does, consistent
with Gemma's answers being structurally less question-focused, possibly
a side-effect of its answer style or the missing-punctuation pattern).
**Thresholds confirmed as-is. No changes needed.**

---

## Metric 5: Completeness — floor AND ceiling, CONFIRMED after real evidence

Two checks: (1) does the answer end with real Khmer sentence-final
punctuation (។ or ៕), and (2) is the answer's length appropriate
relative to `expected_answer`.

### The bug the original design had

The original version only checked a FLOOR (`length_ratio >= 0.80` →
automatic 5 if ends cleanly), with no ceiling. A `length_ratio` of
0.85 and a `length_ratio` of 10.0 scored identically.

### Evidence that drove the ceiling fix

| Model         | p50   | p90   | max    | >2.0x  | >3.0x  |
|---------------|-------|-------|--------|--------|--------|
| SeaLLM v3 7B  | 0.997 | 1.736 | 5.772  | 9/100  | 5/100  |
| Qwen 3.5 9B   | 1.979 | 4.386 | 10.009 | 49/100 | 30/100 |
| Gemma 4 E4B   | 1.000 | 1.460 | 2.416  | 2/100  | 0/100  |
| Llama 3.1 8B  | 1.127 | 2.695 | 5.935  | 21/100 | 5/100  |

For Qwen, exceeding 2x is the **median behavior** (49/100 answers,
p50=1.98x). The ceiling was placed so SeaLLM and Gemma's typical range
(p90 ≈ 1.4–1.7x) stays inside the top band, while Qwen's routine
over-length no longer gets a free 5.

Gemma's `length_ratio min=0.000` reflects the 7 blank entries
(length_ratio=0.0 → truncated branch → score=1). The 93 non-blank
answers have a clean distribution (p50=1.000, p90=1.460, max=2.416)
with only 2 entries exceeding 2x.

### Current thresholds

```python
if ends_cleanly:
    if 0.80 <= length_ratio <= 2.00:   score = 5
    elif (0.60 <= ratio < 0.80)
      or (2.00 < ratio <= 3.00):        score = 4
    else:                               score = 3
else:  # truncated or blank
    if length_ratio >= 0.80:            score = 3
    elif length_ratio >= 0.40:          score = 2
    else:                               score = 1   # blank: ratio=0.0
```

**Final score distribution:**

| Model         | 1  | 2  | 3  | 4  | 5  | avg  |
|---------------|----|----|----|----|----|------|
| SeaLLM v3 7B  | 1  | 0  | 12 | 13 | 74 | 4.59 |
| Qwen 3.5 9B   | 0  | 0  | 40 | 14 | 46 | 4.06 |
| Gemma 4 E4B   | 18 | 13 | 18 | 2  | 49 | 3.51 |
| Llama 3.1 8B  | 0  | 2  | 28 | 17 | 53 | 4.21 |

Gemma's 18 score-1 entries = 7 blanks (ratio=0.0) + 11 truncated
answers with ratio < 0.40 (very short answers that also failed to end
cleanly). The 13 score-2 entries are truncated answers with
0.40 ≤ ratio < 0.80. **Thresholds confirmed as-is. No changes needed.**

---

## Final weighted combined scores

| Model         | Ctx Rel | Faith | Ans Corr | Ans Rel | Complete | Combined |
|---------------|---------|-------|---------|---------|---------|---------|
| SeaLLM v3 7B  | 4.48    | 4.52  | 4.64    | 4.01    | 4.59    | **4.476** |
| Llama 3.1 8B  | 4.44    | 4.50  | 4.48    | 3.94    | 4.21    | **4.356** |
| Qwen 3.5 9B   | 4.44    | 3.79  | 4.36    | 3.87    | 4.06    | **4.115** |
| Gemma 4 E4B   | 4.44    | 4.30  | 4.01    | 3.26    | 3.51    | **3.981** |

Weights: Context Relevance 0.20, Faithfulness 0.25, Answer Correctness
0.25, Answer Relevance 0.15, Completeness 0.15.

**Recommendation: SeaLLM v3 7B** — highest combined score, fewest
truncated answers (8/100), fastest average TTFT (23.7s), no blank
answers.

---

## Cross-checks against retrieval evaluation

- **`scripts/08_compare_embedding_models.py`** confirmed BGE-M3 is the
  right retriever (Hit@5=0.97 macro-avg) vs. e5-large (0.65) and MiniLM
  (near zero) — meaning Context Relevance scores are not being dragged
  down by a weak retriever.
- **`scripts/09_compare_retrieval_modes.py`** confirmed hybrid retrieval
  (Recall@5 = 0.995 macro-average) is what's running underneath
  `scripts/10`, which is why Context Relevance clusters tightly at 1
  and 5 (recall is nearly always 1.0 or 0.0 at this corpus's scale).

---

## Calibration history

| Date       | Change                                                              |
|------------|---------------------------------------------------------------------|
| Initial    | Answer Correctness thresholds (0.85,5)/(0.75,4) — median scored 5 for all models (too lenient) |
| Revision 1 | Recalibrated both threshold sets from real percentile distributions |
| Revision 2 | Added Completeness ceiling at 2.0x after Qwen p50=1.98x revealed missing-ceiling bug |
| Revision 2 | Split Answer Relevance to its own threshold set (max observed = 0.9037, not 1.0) |
| Revision 3 | Blank answer policy: score=1 (not None) for Answer Correctness and Answer Relevance; `blank_answer` field added to `detail`; Gemma updated from n=93 to n=100 |

---

## The honest limitation of all of this

**These thresholds are empirically derived from real data, not validated
against human judgment.** That validation step is what
`scripts/13_human_answer_evaluation.py` and
`scripts/14_human_answer_evaluation_report.py` are for: have a person
score the same 100 answers (for the winning model, SeaLLM v3 7B) by
hand, and compare their scores to this proxy's scores. If they broadly
agree, that's real evidence the proxy can be trusted for faster
iteration later. If they disagree systematically, that's the signal to
come back to this document and `scripts/11_diagnose_generator_scores.py`
and recalibrate.