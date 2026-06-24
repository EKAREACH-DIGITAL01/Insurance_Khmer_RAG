# Generator Evaluation: Scoring Calibration Notes

This document explains where the 1-5 scoring thresholds for each
automated proxy metric in `scripts/10_compare_generator_models.py` came
from, why they're set where they are, and what their limitations are.
Keep this alongside that script and `scripts/rescore_generator_eval.py`
so the reasoning doesn't get lost the next time these need adjusting.

Companion tool: **`scripts/diagnose_generator_eval.py`** runs every
percentile/distribution check referenced below against the live
`reports/generator/generator_eval_*.jsonl` files. Re-run it any time the
corpus, embedding model, or generator lineup changes, since every
threshold here is a snapshot calibration against THIS corpus's real
data, not a universal constant.

---

## The core idea: thresholds are calibrated to THIS corpus's real data, not a universal rule

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

## Metric 1: Context Relevance -- no calibration needed, CONFIRMED stable

```python
score_context_relevance(context_chunk_ids, expected_chunks):
    recall = len(found) / len(expected_chunks)
    return round(1 + recall * 4)
```

Deterministic, not a similarity proxy -- `expected_chunks` lists the
real chunk IDs a correct answer should retrieve from; the score is an
exact linear mapping of retrieval recall. No calibration needed because
recall is already a clean, meaningful 0.0-1.0 fraction.

**Confirmed by real evidence**: score distribution was checked across
all 4 models on the actual 400-answer run:

```
SeaLLM: {1: 11, 3: 4, 5: 85}
Qwen:   {1: 12, 3: 4, 5: 84}
Gemma:  {1: 12, 3: 4, 5: 84}
Llama:  {1: 12, 3: 4, 5: 84}
```

Nearly identical across all 4 models -- exactly as expected, since
retrieval is generator-independent (same BGE-M3 index, BM25, RRF
regardless of which LLM is generating). The distribution clusters at 1
and 5 specifically because this corpus mostly has 1 expected chunk per
question, making retrieval close to binary hit/miss at `CONTEXT_K=5`. If
the corpus later has questions with multiple expected_chunks, or if
retrieval quality drops (see `reports/retrieval/retrieval_eval.md`),
re-check this distribution -- a weaker retriever would produce real
spread across 2/3/4 that this linear formula handles correctly, but
hasn't yet been stress-tested against.

---

## Metric 2: Faithfulness -- number-overlap, CONFIRMED working correctly by real evidence

```python
fraction = grounded_numbers / total_numbers_in_answer
5 if fraction == 1.0, 4 if >=0.75, 3 if >=0.50, 2 if >0, else 1
```

Not a cosine-similarity metric -- checks whether every Arabic/Khmer
numeral in the generated answer can be found somewhere in the retrieved
context. Chosen over embedding similarity specifically because two
texts can be semantically close while disagreeing on a specific fact (a
wrong date, a fabricated fee) -- exactly the failure mode cosine
similarity is worst at catching, and exactly what matters most in a
legal-citation domain.

**This is the metric the calibration pass most CONFIRMED, not fixed.**
Real evidence from the 400-answer run:

```
SeaLLM: 111 numbers total, 97.3% grounded, 48/100 answers had zero numbers
Qwen:   418 numbers total, 72.0% grounded, 13/100 answers had zero numbers
Gemma:   98 numbers total, 93.9% grounded, 63/100 answers had zero numbers
Llama:  145 numbers total, 100.0% grounded, 50/100 answers had zero numbers
```

Qwen writes **2.9-4.3x more numbers per answer** than the other three
models, and only grounds 72% of them -- roughly 117 numbers across
Qwen's 100 answers were not found anywhere in the context it was given.
This is NOT a length artifact (the scoring is fraction-based per answer,
already normalized) -- it's a real, substantial difference in
hallucination behavior. The bands were left exactly as originally
designed because the evidence supports them as-is.

**Known limitation, unchanged**: only catches fabricated *numbers*. A
fabricated qualitative claim with no numbers in it would not be caught.

---

## Metric 3: Answer Correctness -- calibrated against ITS OWN distribution

Compares: **generated answer vs. `expected_answer`** -- both full
statements answering the same question, same grammatical role.

**Evidence** (percentiles of `answer_correctness_similarity` across all
400 real answers):

| Model  | min    | p10    | p25    | p50    | p75    | p90    | max    |
|--------|--------|--------|--------|--------|--------|--------|--------|
| SeaLLM | 0.5697 | 0.8010 | 0.8861 | 0.9398 | 0.9749 | 0.9946 | 1.0000 |
| Qwen   | 0.4718 | 0.7922 | 0.8615 | 0.9047 | 0.9349 | 0.9631 | 0.9861 |
| Gemma  | 0.4718 | 0.7249 | 0.8892 | 0.9401 | 0.9848 | 0.9960 | 1.0000 |
| Llama  | 0.6208 | 0.8029 | 0.8650 | 0.9213 | 0.9682 | 0.9946 | 1.0000 |

Median sits at 0.90-0.94 for every model -- expected, since every gold
question was written directly from a real source chunk (no out-of-scope
noise). The FIRST threshold attempt for this project --
`(0.85,5)/(0.75,4)/(0.60,3)/(0.45,2)/(0.00,1)` -- put the median answer
from every model at a perfect 5, measuring "is this in-domain Khmer
legal text" more than "is this answer actually correct."

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

Floor at 0.50 reflects this project's own `OUT_OF_DOMAIN_THRESHOLD` in
`app.py` (0.49) -- every gold question here is in-domain by
construction, so no answer should legitimately fall below 0.50. The
(0.00, 1) catch-all covers the tiny number of real values observed just
under it (0.4718, 0.47).

---

## Metric 4: Answer Relevance -- a DIFFERENT scale than Answer Correctness, by design

Compares: **generated answer vs. the ORIGINAL QUESTION** -- a question
asks, an answer states; structurally different content than the
Answer Correctness comparison, with a LOWER similarity ceiling.

**Evidence** (percentiles of `answer_relevance_similarity` across all
400 real answers):

| Model  | min    | p10    | p25    | p50    | p75    | p90    | max    |
|--------|--------|--------|--------|--------|--------|--------|--------|
| SeaLLM | 0.6853 | 0.7328 | 0.7518 | 0.7803 | 0.8154 | 0.8405 | 0.9037 |
| Qwen   | 0.3489 | 0.7056 | 0.7502 | 0.7711 | 0.7920 | 0.8236 | 0.8753 |
| Gemma  | 0.3489 | 0.6444 | 0.7244 | 0.7622 | 0.7891 | 0.8211 | 0.8741 |
| Llama  | 0.6771 | 0.7207 | 0.7504 | 0.7721 | 0.8095 | 0.8380 | 0.8741 |

**The max observed across all 400 real answers, every model combined,
was 0.9037 -- a single outlier; the real p90 sits at 0.82-0.84.** The
ORIGINAL design shared Answer Correctness's thresholds for this metric
too -- under those (5 requires >=0.90), a 5 was nearly IMPOSSIBLE here
for any model, regardless of answer quality. This was a measurement
ceiling mismatch, not a quality signal, and was the first real fix made
during this calibration pass.

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

No floor adjustment was added here the way Answer Correctness has one
-- there's no equivalent domain assumption ("this should never
legitimately happen below X") for question-vs-answer similarity.

---

## Metric 5: Completeness -- rule-based, FIXED after real evidence revealed a missing ceiling

Two checks: (1) does the answer end with real Khmer sentence-final
punctuation (។ or ៕), as opposed to being cut off mid-sentence, and (2)
is the answer's length appropriate relative to `expected_answer`.

### The bug the original design had

```python
# ORIGINAL (floor only):
if ends_cleanly:
    if length_ratio >= 0.80: score = 5
    elif length_ratio >= 0.60: score = 4
    else: score = 3
```

A `length_ratio` of 0.85 and a `length_ratio` of **10.0** scored
**identically** -- a perfect 5 either way, as long as the answer ended
on a clean sentence boundary. This went undetected until the actual
`length_ratio` distribution was checked directly:

| Model  | p50    | p75    | p90    | p95    | max     | count >2.0x | count >3.0x |
|--------|--------|--------|--------|--------|---------|-------------|-------------|
| SeaLLM | 0.992  | 1.260  | 1.736  | 2.808  | 5.772   | 9/100       | 5/100       |
| Qwen   | **1.979** | **3.240** | **4.386** | 4.812  | 10.009  | **49/100**  | **30/100**  |
| Gemma  | 0.914  | 1.135  | 1.444  | 1.786  | 2.416   | 2/100       | 0/100       |
| Llama  | 1.127  | 1.885  | 2.695  | 2.940  | 5.935   | 21/100      | 5/100       |

For SeaLLM and Gemma, exceeding 2x length is a genuine outlier (9% and
2% of answers). **For Qwen, exceeding 2x is the median behavior** --
49/100 answers, with the typical (p50) answer already at 1.98x. This
isn't occasional verbosity, it's a systematic pattern -- and combined
with Metric 2's finding that Qwen also grounds a smaller fraction of
the (much larger number of) numbers it states, the evidence points to a
real, related pattern: Qwen routinely writes long, elaborated answers
containing more unverified detail, and the floor-only Completeness
logic was silently rewarding this with a free perfect score every time.

### Current thresholds (floor AND ceiling)

```python
if ends_cleanly:
    if 0.80 <= length_ratio <= 2.00:
        score = 5
    elif (0.60 <= length_ratio < 0.80) or (2.00 < length_ratio <= 3.00):
        score = 4
    else:
        score = 3   # genuinely short (<0.60) OR well past 3x (>3.00)
else:
    # truncated branch -- UNCHANGED, this evidence was specifically
    # about the ceiling, not truncation handling
    if length_ratio >= 0.80: score = 3
    elif length_ratio >= 0.40: score = 2
    else: score = 1
```

The ceiling at 2.0x (full marks) / 3.0x (still acceptable) was placed
where SeaLLM's and Gemma's typical range (p90 1.4-1.7x) stays well
inside the top band, Llama's slightly heavier tail (p90 2.7x) mostly
lands in the second band, and Qwen's routine over-length (p50 already
1.98x, with half its answers past 2x) no longer clears the top band by
default.

**Known limitation**: a model that's *correctly* more thorough than the
terse gold answer (legitimately covering more sub-points, all real and
grounded) could still be penalized by the ceiling alone. This metric
doesn't distinguish "padded/repetitive" from "genuinely more complete
and correct" -- that distinction is exactly what Faithfulness (Metric 2)
and Answer Correctness (Metric 3) are for. Completeness alone should
never be read in isolation from those two.

---

## Cross-checks that validated this calibration against independent evidence

- **`scripts/08_compare_embedding_models.py`** confirmed BGE-M3 is the
  right retriever (Hit@5=0.98, Recall@5=0.96) vs. e5-large (0.66/0.64)
  and MiniLM (near zero) -- meaning Context Relevance scores aren't
  being dragged down by a weak retriever underneath the generator
  comparison.
- **`scripts/09_compare_retrieval_modes.py`** confirmed hybrid retrieval
  (Recall@5 = 0.995 macro-average) is what's actually running underneath
  `scripts/10` -- which is why Context Relevance's score distribution
  above clusters so tightly at 1 and 5 (recall is nearly always 1.0 or
  0.0 at this corpus's scale).

---

## The honest limitation of all of this

**These thresholds are empirically derived from real data, not validated
against human judgment.** They were placed to spread the observed
distribution sensibly across 1-5 and to fix one confirmed measurement
bug (Completeness's missing ceiling) -- not confirmed to track what a
human reader would actually call "good" vs "bad." That validation step
is exactly what `scripts/11_evaluate_rag.py`'s human-scoring template is
for: have a person score the SAME 100 answers (for the winning model) by
hand, and compare their scores to this proxy's scores side by side. If
they broadly agree, that's real evidence the proxy can be trusted for
faster iteration later. If they disagree systematically, that's the
signal to come back to this document -- and to
`scripts/diagnose_generator_eval.py` -- and recalibrate, the same way
every threshold here was derived in the first place.