# LLM-as-judge prompt (RAGAS + hallucination)

Scores the answers produced by `scripts/13_generate_model_answers.py`.

Run `scripts/14_prepare_llm_judge_input.py` once first. It scores
`context_relevance` deterministically from the chunk ids, resets the four
judged metrics to 0, and strips the retrieved chunk text — leaving ~1.6k chars
per record.

**What the judge compares:** `generated_answer` against `expected_answer`.
Nothing else. `expected_answer` is human-curated and human-validated ground
truth, so it is the standard worth measuring against.

**What the judge does NOT do:** score `context_relevance`. That is recall of
`expected_chunks` in `context_chunk_ids` — an exact set comparison. A script
computes it perfectly; an LLM asked to do it just returns 5 every time.

| | |
|---|---|
| Records | 4 models × 100 = **400** |
| Sessions | **4 — one per model, all 100 records in one go** (~1.6k chars/record) |
| Judged metrics | faithfulness, answer_relevance, answer_correctness, completeness |
| Also filled in | `combined`, `hallucination`, `hallucination_evidence` |

**One model per session, never mixed.** A judge that has just read SeaLLM's
answers anchors its scale there and starts scoring the next model relative to
SeaLLM instead of against the rubric.

**The judge must not be the model being judged** — self-preference bias.

---

## Running it

In `codex`, paste:

```
Read docs/llm_judge_prompt.txt and follow it exactly.
Substitute {{FILE}} with:
reports/generator_model_comparison/SeaLLMs-v3-7B-Chat-Q4_K_M/generated_answer_results.jsonl
```

Then verify, every time:

```bash
python scripts/check_judge_progress.py \
  reports/generator_model_comparison/SeaLLMs-v3-7B-Chat-Q4_K_M/generated_answer_results.jsonl
```

Repeat in a **fresh session** for `Qwen3.5-9B-Q4_K_M`,
`Sailor2-L-8B-Chat-Q4_K_M`, `Llama-3.1-8B-Instruct-Q4_K_M`.

---

## The prompt

Kept verbatim in `docs/llm_judge_prompt.txt`.

```
You are an impartial evaluator for a Khmer-language RAG system answering
questions about Cambodian insurance law.

=============================================================================
SCOPE -- read this first
=============================================================================
This is NOT a coding task. Work ONLY inside the file named below.

Do NOT open, read or modify any other file -- not this prompt file, not any
script, not any report, not README. Do NOT run scripts, tests or git. Do NOT
create files. If you think an instruction here is wrong, score anyway and say
so in your final message; do not edit anything to fix it.

FILE: {{FILE}}
JSONL, one record per line, 100 records. Score EVERY record.

=============================================================================
WHAT YOU FILL IN
=============================================================================
  scores.faithfulness         integer 1-5
  scores.answer_relevance     integer 1-5
  scores.answer_correctness   integer 1-5
  scores.completeness         integer 1-5
  combined                    decimal, 2 places (formula below)
  hallucination               the string "true" or "false"
  hallucination_evidence      string (verbatim quote, or "" when false)

DO NOT TOUCH scores.context_relevance. It is already scored -- measured from
the chunk ids, not judged. Leave whatever value is there.

Change nothing else: not question, expected_answer, generated_answer,
expected_chunks, context_chunk_ids, gate_reason, timing.

=============================================================================
WHAT YOU COMPARE
=============================================================================
The comparison that matters is:

    generated_answer   vs   expected_answer

expected_answer is the human-curated, human-validated ground truth. Judge the
generated answer against it. Do NOT use your own knowledge of Cambodian law --
an answer that is true in the real world but absent from expected_answer is
not evidence of quality here.

You are comparing MEANING, not wording. A correct paraphrase scores full
marks. Do not reward an answer for matching the gold phrasing, and do not
punish one for restating it differently.

=============================================================================
FORMAT RULES -- absolute
=============================================================================
1. Every metric is an INTEGER: 1, 2, 3, 4 or 5. Never a decimal (not 4.5),
   never a string ("4"), never null, never 0. 0 means "not yet judged" and
   must not survive your pass.
2. combined is the ONLY decimal field. Two decimal places.
3. hallucination is the lowercase STRING "true" or "false". Not a JSON
   boolean, not "True", not "yes"/"no".
4. hallucination_evidence is "" when hallucination is "false".
5. Score every record. Never skip one, never leave a 0 in the four fields
   above.
6. When genuinely torn between two levels, choose the LOWER. In a legal
   domain an overstated score is more harmful than an understated one.

=============================================================================
METRIC RUBRICS -- integer 1-5, judged against expected_answer
=============================================================================

answer_correctness -- does it say the same thing as expected_answer?
  This is the primary metric. Weight your attention here.
  5  Same substance as expected_answer. Wording may differ completely.
  4  Correct, with a small omission or an immaterial imprecision.
  3  Partially correct: a substantive element missing, or one detail wrong,
     but the thrust is right.
  2  Largely incorrect; only a fragment overlaps expected_answer.
  1  Contradicts expected_answer, or is unrelated to it.

faithfulness -- does it avoid asserting things expected_answer does not support?
  5  Every specific claim is supported by expected_answer.
  4  Substantively supported; one minor extra detail that changes nothing.
  3  One clear unsupported factual claim.
  2  Several unsupported claims, or one invented legal specific (an article
     number, a figure, a deadline) stated as fact.
  1  Substantially invented, or contradicts expected_answer outright.
  NOTE: correctly saying the material does not cover something is FAITHFUL (5).

answer_relevance -- does it address the question asked?
  5  Directly answers the question.
  4  Answers it, with padding or a digression that does not mislead.
  3  Partly answers it, or answers a related-but-different question.
  2  Talks around the subject without addressing the question.
  1  Does not address the question at all.

completeness -- does it cover expected_answer and finish properly?
  5  Covers the substance of expected_answer and ends as a finished statement.
  4  Covers the substance; one secondary element missing.
  3  A required element of expected_answer is missing.
  2  Only a fragment of what was expected.
  1  Truncated mid-sentence, or too short to be usable.
  NOTE: completeness is COVERAGE and CLOSURE, not length. A complete
  two-sentence answer scores 5.

=============================================================================
COMBINED
=============================================================================
combined = context_relevance    * 0.20   (already in the record -- read it)
         + faithfulness         * 0.25
         + answer_correctness   * 0.25
         + answer_relevance     * 0.15
         + completeness         * 0.15

Weights sum to 1.00. Round to 2 decimal places.

Worked example -- context_relevance 5 (already there), faithfulness 4,
answer_relevance 4, answer_correctness 5, completeness 4:

    5 * 0.20  = 1.00   (context_relevance)
    4 * 0.25  = 1.00   (faithfulness)
    5 * 0.25  = 1.25   (answer_correctness)
    4 * 0.15  = 0.60   (answer_relevance)
    4 * 0.15  = 0.60   (completeness)
                -----
    combined  = 4.45

=============================================================================
HALLUCINATION -- judged separately, never folded into combined
=============================================================================
Apply in order:

  1. Does generated_answer assert a SPECIFIC fact -- an article or provision
     number (មាត្រា/ប្រការ N), a figure, a percentage, a deadline, a monetary
     amount, a named institution, or a concrete procedural step?
        No  -> "false".
  2. Is that specific fact present in expected_answer?
        Yes -> "false".
        No  -> "true", and quote the invented span verbatim into
               hallucination_evidence.

NOT hallucinations:
  - Vague prose that makes no specific claim.
  - An incomplete answer (that is completeness).
  - Correct paraphrase or restructuring of expected_answer.
  - Saying the material does not cover the question.
  - Khmer numerals where expected_answer spells the number in words, or the
    reverse -- ៣ and បី are the same number.
  - A PROVISION LABEL on otherwise correct content. If the answer opens with
    or includes "មាត្រា N" / "ប្រការ N" as the heading of text that does match
    expected_answer, that is ATTRIBUTION, not fabrication -- the answer is
    naming the provision it is quoting. Gold answers often omit the citation.
    Do NOT flag it, and do not lower faithfulness for it.
    Flag a provision number ONLY when the CONTENT attributed to it is absent
    from or contradicts expected_answer.

IS a hallucination:
  - Citing មាត្រា/ប្រការ N and attributing content to it that expected_answer
    does not support (see the attribution exception above -- a bare citation
    on correct content is fine).
  - Any deadline, figure or amount not in expected_answer.
  - Giving a duty or power to an institution expected_answer does not name.
  - A fact that is true of Cambodian law generally but is NOT in
    expected_answer.

CONSISTENCY: if hallucination is "true", faithfulness should normally be 3 or
below -- they measure the same thing. Writing faithfulness 5 with
hallucination "true" means one of the two is wrong; re-read expected_answer.

=============================================================================
EDGE CASES
=============================================================================
- Empty generated_answer: answer_correctness 1, completeness 1,
  answer_relevance 1, faithfulness 5 (nothing claimed, nothing invented),
  hallucination "false".
- A refusal ("this is outside the scope..."): answer_correctness 1,
  completeness 1, faithfulness 5, hallucination "false"; judge
  answer_relevance on its own terms.
- expected_answer missing or empty: answer_correctness 3, and judge the rest
  normally.

=============================================================================
HOW TO EDIT THE FILE -- follow exactly
=============================================================================
The file is Khmer UTF-8. Hand-rewriting it has corrupted the text before
(Khmer turning into "á\x9e\x80..."). To make that impossible:

  - Edit it with a small Python script, not by re-emitting the file yourself.
  - Read with open(path, encoding="utf-8") and write with
    open(path, "w", encoding="utf-8").
  - Use json.dumps(record, ensure_ascii=False) so Khmer stays Khmer.
  - One record per line, same line order, same 100 records.
  - Do not pretty-print or reformat.

Delete any temporary script when you are done.

=============================================================================
OUTPUT
=============================================================================
Your ONLY output is the updated score fields inside {{FILE}}.
Do NOT write reports. Do NOT run scripts/15 or any other pipeline script --
reporting is a separate step the user runs afterwards.

When finished, state: how many records you scored, how many you flagged as
hallucinations, and the distribution of answer_correctness.
```

---

## After each model — verify

```bash
python scripts/check_judge_progress.py \
  reports/generator_model_comparison/<model>/generated_answer_results.jsonl
```

It exits non-zero and tells you what is wrong. What it checks:

- **`records` is 100.** If it dropped, the judge mangled the file.
- **`judged` is 100.** Silent truncation looks like success in the judge's
  own summary but shows up here.
- **Mojibake.** If the judge rewrote the file without `encoding="utf-8"`, the
  Khmer is destroyed and *cannot be repaired from the file* — only
  `git checkout` gets it back. A record count will not catch this; it has
  happened once already.
- Scores outside 1–5, non-integer scores, `hallucination` that is not
  `"true"`/`"false"`, and flags with no evidence.

If it reports a problem:

```bash
git checkout reports/generator_model_comparison/<model>/generated_answer_results.jsonl
python scripts/14_prepare_llm_judge_input.py
```

then redo that model.

---

## When all 400 are scored

```bash
python scripts/15_generate_comparison_report.py
```

That writes the per-model summaries and
`generator_model_comparison.{json,md}`, ranks the four models and names the
winner. **This ranking selects the shipped generator** — the deterministic proxy
scorer that previously made that call has been removed. Any answer still at 0 is
excluded from the averages and reported as unjudged, so a partial run cannot
look complete.

## Validating the judge

An LLM judge on Khmer legal text is not reliable by default. Hand-score ~30
answers yourself and report your agreement with the judge on that sample.
Without it the result is "an LLM said so"; with it, it is a measured method.
