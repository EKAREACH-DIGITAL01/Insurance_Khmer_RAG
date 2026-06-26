# Gold Question Methodology (guidance / laws / sub_decrees / prakas)

How to author the AI-assisted half (question 26 onward) of each
`gold_questions.json` so it reads like the hand-written first 25
(0% AI) of **both** guidance and law. The first 25 of every category are the
gold standard — study them before authoring.

## How to instruct Claude next time (one line is enough)

Say something like:

> "Read `data/09_evaluation/GOLD_QUESTIONS_METHODOLOGY.md`, then build gold
> questions 26–N for **sub_decrees** using `sub_decrees_chunks.json`, same as we
> did for laws."

That's all you need. Optional extras you can add: how many to generate, which
document to weight toward, or "skip chunks X/Y". Claude will then follow the
runbook below and **hand-verify every entry** before saving. You do NOT need to
re-explain the pattern — it's all in this file.

## Runbook (the exact steps Claude follows)

1. **Read this whole file** + skim the category's first 25 in `gold_questions.json`
   to absorb tone, and check guidance/law first-25 for the question↔answer shape.
2. **Dump the chunk file** (`data/08_merged_chunked_documents/<cat>_chunks.json`)
   and read the cleaned content of candidate chunks. Note the chunk-id ↔ article
   mapping (verify the off-by-one).
3. **Pick chunks** (weight toward the most chatbot-relevant document; cover all
   docs; skip chunks already used by the first 25).
4. **For each: choose the slice + write the question** per the 7 rules below.
5. **Self-validate in a scratch script** (verbatim, opening-match, coverage,
   clean endings, length, uniqueness) — fix markers until 0 failures.
6. **Audit coverage vs the full chunk** — confirm list answers are ~100% complete,
   narrow answers are the right single clause; no truncation, no trailing junk.
7. **Hand-read every pair** for typos and natural Khmer, then write to JSON
   (preserve the first 25 and any of the user's existing edits).
8. **Report** what was generated and any judgement calls (chunks skipped, the
   `ច្បាប់នេះ`→law-name edits, anything that hit the 1024 cap).

## The point (read this first)

A gold entry is `{question, expected_chunks:[ONE chunk], expected_answer}`.
At evaluation time the chatbot **retrieves that one chunk** and writes an answer.
We then compare the chatbot's *generated* answer to our `expected_answer`.

> Goal: generated ≈ expected (identical, or off only a little).

That only happens if **`expected_answer` is a complete, coherent, verbatim
passage of the retrieval chunk** — the exact text the model would naturally
produce from that chunk — AND the **question opens with the same words the
answer opens with**. Matching only the question/answer opening is NOT enough;
the whole answer body must equal the chunk's passage. Do not match blindly.

## The 7 rules

1. **One chunk = one provision.** `expected_chunks` is a single chunk id.
   The answer comes only from that chunk.

2. **Answer = the COMPLETE coherent passage** that fully answers the question,
   sliced **verbatim** (whitespace-normalised) from the cleaned chunk:
   - "what / which / list" question → the **whole provision**: lead line + **all**
     list items + any concluding clause. Never stop mid-list.
   - narrow question (when / how many / percent / who / penalty) → the **single
     precise sentence/clause** that answers it (a short complete answer is fine,
     e.g. human guidance "ប៉ុន្មានភាគរយ" answers).
   - a chunk that defines **several things** (e.g. agent / broker / risk-assessor
     in one article) → split into **one Q&A per item**, each a complete definition.

3. **Strip the article header** from the answer start: drop `មាត្រា N - _` or
   `មាត្រា N - <title>`. Start the slice at the first substantive word.

4. **Exclude trailing junk** that the chunk drags in after the provision:
   next-article text, `###` section headings (`ជំពូក`, `ផ្នែក`), page artifacts
   (`១៦/ ៣១`), `---`. Use an explicit end-marker to stop at the provision's final `។`.

5. **Question opens with the answer's exact opening words**, then swaps the
   specific predicate for an interrogative — `អ្វី / អ្វីខ្លះ / ដូចម្តេច / បែបណា /
   ដើម្បីអ្វី / ពេលណា / ប៉ុន្មាន / ចំពោះជនណា`. **No yes/no questions**
   (`...ដែរឬទេ? / ...បានទេ?`). Pattern, from human law qid_008:
   - A: `អំពើល្មើស…ត្រូវផ្ដន្ទាទោស` **`ដាក់ពន្ធនាគារ…`**
   - Q: `តើ` `អំពើល្មើស…ត្រូវផ្ដន្ទាទោស` **`ដូចម្តេច?`**

6. **The one allowed non-verbatim edit.** Goal/purpose/scope provisions open with
   `ច្បាប់នេះ` ("this law"). Replace `ច្បាប់នេះ` with the **actual law name** so the
   answer is self-contained and the question can name the law. This is exactly what
   the human first-25 do (law qid_001/009/010/019). Everything else stays verbatim.
   Use the law's real title (e.g. `ច្បាប់ស្តីពីកិច្ចការពារអ្នកប្រើប្រាស់`).

7. **No reference codes** in question or answer: `មាត្រា<digit>` (no space),
   `លេខ ០…`, `/សណន`, `ឧ.ធ.ក`. Real users don't cite codes. Length **80–1024**
   chars. If a full list would exceed 1024, pick a different self-contained
   provision rather than truncating the list.

## Tooling

`build_gold_questions.py` does deterministic, verbatim slicing. Add tuples to
`SPECS["<category>"]`: `(chunk_id, question, slice_mode)`.

- `clean()` joins PDF-wrapped lines (no space, Khmer is continuous), de-hyphenates,
  restores the space after a sentence-final `។`. Always slice from `clean()` output.
- Slice modes:
  - `('m', start, end)` — verbatim between two short markers. `start=None` = from
    beginning, `end=None` = to end. **Preferred** for laws: set `start` to the first
    substantive words (excludes the glued header) and `end` to the provision's last
    words (excludes trailing junk). Keep markers SHORT and within one physical line
    so a wrap-join can't break the match.
  - `('h', n)` / `('hs', marker, n)` — lead line + first n list items. Avoid for laws
    because the `មាត្រា N - _` header glues onto the lead line.
  - `('full',)` — whole cleaned chunk; only when the chunk is one tight passage.
- **Chunk-id off-by-one (laws):** `law_XX_chunk_N` is `មាត្រា (N+1)`. Verify the
  chunk content, don't trust the number.

## Verify EVERY entry (don't rely on the script alone)

Read each pair by eye, and check programmatically:
1. **Verbatim**: `nows(answer) in nows(clean(chunk))` (`nows` = strip whitespace).
2. **Opening match**: first ~8 non-space chars of answer == first ~8 of question
   after dropping leading `តើ`.
3. **Completeness**: for list/"what" questions, answer covers ~100% of the
   provision body (lead + all items). Low coverage is only OK for narrow questions
   or multi-definition chunks split per item.
4. **Clean**: answer contains no `###`, `---`, page artifact, next-article
   `មាត្រា<digit>`, `ធានា-រ៉ាប់រង`, `ថាល់`; ends on `។` or `)`.
5. **Length** 80–1024; **questions unique** and none duplicates a first-25 question.

## Coverage balance

Weight toward the most relevant law for the chatbot (insurance = `law_04`), but
cover every document in the category. The human first-25 already span all docs;
don't duplicate their chunks.
