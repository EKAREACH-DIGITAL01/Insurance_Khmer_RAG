# Gold Questions — Provenance & Methodology

This document covers the evaluation gold sets at
`data/09_evaluation/<category>/gold_questions.json` for all four categories:
**guidance, laws, sub_decrees, prakas**. It records where every question came
from (provenance) and the exact method used to author the AI-assisted half so
the set can be reproduced and defended.

## Provenance

There are **400 high-quality gold questions** in total, split across the four
categories. Every category opens with **25 questions written entirely by hand
(0% AI)** — these are the gold standard that defines the tone and the
question↔answer shape for everything that follows.

| Category     | Total | Human (qid 1–25) | Claude AI-assisted |
| ------------ | :---: | :--------------: | :----------------: |
| guidance     |  100  |        25        |   26–100  (75)     |
| laws         |   75  |        25        |   26–75   (50)     |
| sub_decrees  |   75  |        25        |   26–75   (50)     |
| prakas       |  150  |        25        |   26–150  (125)    |
| **Total**    | **400** |    **100**     |     **300**        |

- **Human-authored (qid 1–25 of every category): 100 questions, 0% AI.** A
  person studied the source documents and wrote each question and its answer by
  hand. These are never modified by tooling.
- **Claude AI-assisted, human-supervised: 300 questions.** Each answer is a
  *verbatim* passage sliced straight from the corresponding source chunk in
  `data/08_merged_chunked_documents/<category>_chunks.json` — no fabricated
  text. The questions were written to mirror the manual style of qid 1–25, and
  every entry was hand-verified before being committed.

The deterministic slicing and validation that produced the AI-assisted half
live in `data/09_evaluation/build_gold_questions.py`; the `SPECS` table in that
script is the per-question record of which chunk and which slice each answer
came from.

## The point (read this first)

A gold entry is `{question, expected_chunks: [ONE chunk], expected_answer}`. At
evaluation time the chatbot **retrieves that one chunk** and writes an answer.
We then compare the chatbot's *generated* answer against our `expected_answer`.

> Goal: generated ≈ expected (identical, or off only a little).

That only happens when `expected_answer` is a **complete, coherent, verbatim
passage of the retrieval chunk** — the exact text the model would naturally
produce from that chunk — **and** the question opens with the same words the
answer opens with. Matching only the question/answer opening is not enough; the
whole answer body must equal the chunk's passage. Do not match blindly.

## How to instruct Claude next time

One line is enough, for example:

> "Read `docs/gold_questions_methodology.md`, then build gold questions 26–N for
> **sub_decrees** using `sub_decrees_chunks.json`, same as we did for laws."

Optional extras: how many to generate, which document to weight toward, or "skip
chunks X/Y". Claude then follows the runbook below and hand-verifies every entry
before saving. You do not need to re-explain the pattern — it is all here.

## Runbook (the exact steps Claude follows)

1. **Read this whole file** and skim the category's first 25 entries in
   `gold_questions.json` to absorb tone, plus the guidance/law first-25 for the
   question↔answer shape.
2. **Dump the chunk file** (`data/08_merged_chunked_documents/<cat>_chunks.json`)
   and read the cleaned content of candidate chunks. Note the chunk-id ↔ article
   mapping and verify the off-by-one.
3. **Pick chunks** — weight toward the most chatbot-relevant document, cover all
   documents, and skip chunks already used by the first 25.
4. **For each chunk**, choose the slice and write the question per the seven
   rules below.
5. **Self-validate in a scratch run** (verbatim, opening-match, coverage, clean
   endings, length, uniqueness) and fix markers until there are zero failures.
6. **Audit coverage against the full chunk** — confirm list answers are ~100%
   complete and narrow answers are the right single clause; no truncation, no
   trailing junk.
7. **Hand-read every pair** for typos and natural Khmer, then write to JSON,
   preserving the first 25 and any of the user's existing edits.
8. **Report** what was generated and any judgement calls (chunks skipped, the
   `ច្បាប់នេះ` → law-name edits, anything that hit the 1024-character cap).

## The seven rules

1. **One chunk = one provision.** `expected_chunks` is a single chunk id, and
   the answer comes only from that chunk.

2. **Answer = the complete coherent passage** that fully answers the question,
   sliced verbatim (whitespace-normalised) from the cleaned chunk:
   - A "what / which / list" question → the **whole provision**: lead line, all
     list items, and any concluding clause. Never stop mid-list.
   - A narrow question (when / how many / percent / who / penalty) → the
     **single precise sentence or clause** that answers it. A short complete
     answer is fine (e.g. the human guidance "ប៉ុន្មានភាគរយ" answers).
   - A chunk that defines several things (e.g. agent / broker / risk-assessor in
     one article) → split into **one Q&A per item**, each a complete definition.

3. **Strip the article header** from the answer start: drop `មាត្រា N - _` or
   `មាត្រា N - <title>`. Begin the slice at the first substantive word.

4. **Exclude trailing junk** the chunk drags in after the provision:
   next-article text, `###` section headings (`ជំពូក`, `ផ្នែក`), page artifacts
   (`១៦/ ៣១`), and `---`. Use an explicit end-marker to stop at the provision's
   final `។`.

5. **The question opens with the answer's exact opening words**, then swaps the
   specific predicate for an interrogative — `អ្វី / អ្វីខ្លះ / ដូចម្តេច /
   បែបណា / ដើម្បីអ្វី / ពេលណា / ប៉ុន្មាន / ចំពោះជនណា`. **No yes/no questions**
   (`...ដែរឬទេ? / ...បានទេ?`). Pattern, from human law qid_008:
   - A: `អំពើល្មើស…ត្រូវផ្ដន្ទាទោស` **`ដាក់ពន្ធនាគារ…`**
   - Q: `តើ` `អំពើល្មើស…ត្រូវផ្ដន្ទាទោស` **`ដូចម្តេច?`**

6. **The one allowed non-verbatim edit.** Goal/purpose/scope provisions open with
   `ច្បាប់នេះ` ("this law"). Replace `ច្បាប់នេះ` with the **actual law name** so
   the answer is self-contained and the question can name the law. This is exactly
   what the human first-25 do (law qid_001/009/010/019). Use the law's real title
   (e.g. `ច្បាប់ស្តីពីកិច្ចការពារអ្នកប្រើប្រាស់`). Everything else stays verbatim.

7. **No reference codes** in the question or the answer: `មាត្រា<digit>` (no
   space), `លេខ ០…`, `/សណន`, `ឧ.ធ.ក`. Real users do not cite codes. Answer
   length is **80–1024** characters; if a full list would exceed 1024, pick a
   different self-contained provision rather than truncating the list.

## Tooling

`data/09_evaluation/build_gold_questions.py` performs deterministic, verbatim
slicing. Add tuples to `SPECS["<category>"]`: `(chunk_id, question, slice_mode)`.

- `clean()` joins PDF-wrapped lines (no space, since Khmer is continuous),
  de-hyphenates, and restores the space after a sentence-final `។`. Always slice
  from `clean()` output.
- Slice modes:
  - `('m', start, end)` — verbatim text between two short markers. `start=None`
    means "from the beginning", `end=None` means "to the end". **Preferred** for
    laws: set `start` to the first substantive words (this excludes the glued
    header) and `end` to the provision's last words (this excludes trailing junk).
    Keep markers short and within one physical line so a wrap-join cannot break
    the match.
  - `('h', n)` / `('hs', marker, n)` — lead line plus the first n list items.
    Avoid for laws because the `មាត្រា N - _` header glues onto the lead line.
  - `('full',)` — the whole cleaned chunk; only when the chunk is one tight
    passage.
- **Chunk-id off-by-one (laws):** `law_XX_chunk_N` is `មាត្រា (N+1)`. Verify the
  chunk content; do not trust the number.

## Verify every entry (do not rely on the script alone)

Read each pair by eye, and check programmatically:

1. **Verbatim:** `nows(answer) in nows(clean(chunk))` (`nows` = strip whitespace).
2. **Opening match:** the first ~8 non-space characters of the answer equal the
   first ~8 of the question after dropping a leading `តើ`.
3. **Completeness:** for list/"what" questions, the answer covers ~100% of the
   provision body (lead + all items). Low coverage is only acceptable for narrow
   questions or multi-definition chunks split per item.
4. **Clean:** the answer contains no `###`, `---`, page artifact, next-article
   `មាត្រា<digit>`, `ធានា-រ៉ាប់រង`, or `ថាល់`; it ends on `។` or `)`.
5. **Length** 80–1024; **questions unique** and none duplicating a first-25
   question.

## Coverage balance

Weight toward the most relevant law for the chatbot (insurance = `law_04`), but
cover every document in the category. The human first-25 already span all
documents, so do not duplicate their chunks.

## Which evaluation uses which gold scope

Not every evaluation script uses all 400 questions. The choice of scope is
deliberate and depends on what the script measures.

**Human-authored only (the 100 questions, qid 1–25 per category):**

- **`scripts/09_compare_embedding_models.py`** — embedding-model comparison.
- **`scripts/10_compare_retrieval_modes.py`** — retrieval-mode comparison.
- **`scripts/11_compare_generator_models.py`** — generator-model comparison
  (the cross-model ranking).

  *Why:* the 300 AI-assisted questions were authored by reusing the target
  chunk's **exact opening words verbatim** and swapping only the final predicate
  for a question word (see [The seven rules](#the-seven-rules)). That makes each
  AI query near-lexically-identical to its own gold chunk. For **retrieval**
  evaluation (scripts 09 and 10) this leaks the answer into the query: it
  inflates hit/recall/MRR — especially for the BM25/lexical mode — and collapses
  the gap between the models/modes being compared, which is the entire point of
  those scripts. For the **generator comparison** (script 11) it keeps every
  model ranked on the same independently-phrased 100 questions so the comparison
  is apples-to-apples. These scripts enforce the cut with a shared
  `is_human_authored()` helper keyed on `question_id <= 25`.

  Note: script 11 still *runs* SeaLLM over the full 400 (writing its own
  `..._full.jsonl`), but that file is **not** fed into the cross-model
  comparison — only `generated_answer_results.jsonl` (the 100) is.

**All 400 questions (full gold set):**

- **`scripts/14_human_answer_evaluation*.py`** — the human/RAGAS answer
  evaluation of the **winning** generator.

  *Why:* script 11's
  `reports/generator_model_comparison/generator_model_comparison.md` selects
  **SeaLLM** as the winning generator. Once the single production model is
  chosen, the goal shifts from *comparing models fairly* to *measuring that one
  model as broadly as possible*. The lexical-overlap concern that biases a
  *between-model* comparison does not invalidate a *single-model* quality
  measurement, and the extra 300 questions give far wider coverage of provisions
  and document types. So the human answer evaluation runs over all 400 gold
  questions, scored against the answers SeaLLM generated for the full set.
