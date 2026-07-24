# LLM-as-judge prompt (RAGAS + hallucination)

Scores the answers produced by `scripts/13_generate_model_answers.py`.

Each model has **one self-contained JSONL** at
`reports/generator_model_comparison/<model>/generated_answer_results.jsonl`. Every record
already carries the question, the retrieved context text, the gold answer and
the generated answer — the judge never needs another file.

Run `scripts/14_prepare_llm_judge_input.py` once first. It resolves the chunk
ids into `retrieved_context`, resets the metrics to 0 and adds
`hallucination: ""`. **0 and `""` mean "not yet judged."**

- **Scope:** 4 models × 100 questions = **400 records.**
- **One model per session. Never mix models in one session.** A judge that has
  just read SeaLLM's answers anchors its scale there and starts scoring Llama
  *relative to SeaLLM* instead of against the rubric. Each answer must be judged
  only against its own retrieved context and gold answer.
- **The judge must not be the model being judged** — self-preference bias.
- **Fresh session per batch.** Records average ~5,700 characters (max ~14,700),
  so one file is ~570,000 characters — far past any single context window. Ten
  records per batch, ten batches per model, **40 batches total**. A fresh
  session each time also stops context accumulating across batches.

---

## Running it

`codex exec` starts a fresh non-interactive session per invocation, which is
exactly what you want here:

```bash
MODEL=SeaLLMs-v3-7B-Chat-Q4_K_M
FILE=reports/generator_model_comparison/$MODEL/generated_answer_results.jsonl

# batch 1 of 10 — repeat with 11-20, 21-30, … 91-100
codex exec --skip-git-repo-check "$(sed -e "s|{{FILE}}|$FILE|" \
                                       -e "s|{{START}}|1|" \
                                       -e "s|{{END}}|10|" \
                                       docs/llm_judge_prompt.txt)"
```

Keep the prompt body in `docs/llm_judge_prompt.txt` (copy it out of the fenced
block below) so the substitution above works.

---

## Record shape

```json
{
  "question_id": "guidance_qid_001",
  "question": "...",
  "retrieved_context": [{"chunk_id": "guidance_01_chunk_1", "text": "..."}],
  "expected_answer": "...",
  "generated_answer": "...",
  "scores": {
    "context_relevance": 0,
    "faithfulness": 0,
    "answer_relevance": 0,
    "answer_correctness": 0,
    "completeness": 0
  },
  "combined": 0,
  "hallucination": "",
  "hallucination_evidence": ""
}
```

The judge fills in `scores`, `combined`, `hallucination` and
`hallucination_evidence`. Everything else is read-only.

---

## The prompt

The body below is also kept verbatim in `docs/llm_judge_prompt.txt`,
which is what the `codex exec` command above substitutes into.

```
You are an impartial evaluator for a Khmer-language RAG system answering
questions about Cambodian insurance law.

SCOPE -- read this first.
This is NOT a coding task. Do not explore the repository. Do not list
directories, do not search the codebase, do not open README or any source file,
do not run any script, test or git command. The ONE file named below contains
everything you need. Touch nothing else.

FILE: {{FILE}}
It is JSONL, one record per line. Score ONLY lines {{START}} to {{END}}
inclusive (1-indexed). Leave every other line byte-identical.

Every record is self-contained: retrieved_context holds the actual chunk text
the system retrieved, so you never need to look up a chunk id anywhere.

WHAT YOU WRITE INTO EACH RECORD IN RANGE
  scores.context_relevance    integer 1-5
  scores.faithfulness         integer 1-5
  scores.answer_relevance     integer 1-5
  scores.answer_correctness   integer 1-5
  scores.completeness         integer 1-5
  combined                    decimal, 2 places (formula below)
  hallucination               the string "true" or "false"
  hallucination_evidence      string (quote, or "" when false)

Change nothing else -- not question, retrieved_context, expected_answer,
generated_answer, gate_reason, timing, nor any record outside the range.

=============================================================================
FORMAT RULES -- these are absolute
=============================================================================
1. Every metric is an INTEGER: one of 1, 2, 3, 4, 5. Never a decimal (not 4.5),
   never a string ("4"), never null, never 0. 0 means "not yet judged" and must
   not survive your pass.
2. combined is the ONLY decimal field. Two decimal places.
3. hallucination is the lowercase STRING "true" or "false". Not a JSON boolean
   (true), not "True", not "yes"/"no".
4. hallucination_evidence is "" when hallucination is "false", and a verbatim
   quote from generated_answer when it is "true".
5. Score every record in range. If an answer is empty, it still gets integers
   (see EDGE CASES) -- never skip a record and never leave a 0.

=============================================================================
HOW TO JUDGE -- read before scoring
=============================================================================
A. Judge generated_answer ONLY against retrieved_context and expected_answer.
   Do NOT use your own knowledge of Cambodian law. An answer that is true in
   the real world but unsupported by the provided material is NOT faithful and
   must be scored as such. This is the single most important rule.
B. The text is Khmer. Do not penalise Khmer script, Khmer numerals (០-៩), or
   legal phrasing that reads oddly in English. Judge the substance.
C. A shorter answer is not automatically worse. A longer one is not better.
D. Score each metric INDEPENDENTLY. A weak retrieval does not force a low
   answer_correctness, and vice versa.
E. When genuinely torn between two levels, choose the LOWER. In a legal domain
   an overstated score is more harmful than an understated one.

=============================================================================
METRIC RUBRICS -- integer 1-5
=============================================================================

context_relevance -- did retrieval supply what was needed?
  Scores the RETRIEVED CONTEXT, not the answer. An excellent answer built on
  poor context still scores low here.
  5  Context contains everything needed to answer fully; the relevant
     provision or passage is present and complete.
  4  Everything essential is present; some minor supporting detail is missing
     or the relevant passage sits among unrelated chunks.
  3  Partially sufficient -- the topic is covered but a key element needed to
     answer the question is absent.
  2  Only tangentially related; the question could not be answered properly
     from this context alone.
  1  Off-topic. Nothing in the context bears on the question.

faithfulness -- is every claim grounded in the context?
  The failure mode is fabrication: article numbers, figures, deadlines,
  obligations, institutions or procedures not present in retrieved_context.
  5  Every factual claim traces to the context. Nothing invented.
  4  Substantively grounded; one minor unsupported detail that does not change
     the meaning.
  3  Mostly grounded, but one clear unsupported factual claim.
  2  Several unsupported claims, or one fabricated legal specific (an article
     number, a figure, a deadline) presented as fact.
  1  Substantially fabricated, or contradicts the context outright.
  NOTE: an answer that correctly says the context does not cover something is
  FAITHFUL (5), not unfaithful.

answer_relevance -- does it address what was asked?
  5  Directly answers the question asked.
  4  Answers it, with some padding or a digression that does not mislead.
  3  Partly answers it, or answers a related-but-different question.
  2  Mostly talks around the subject without addressing the question.
  1  Does not address the question at all.

answer_correctness -- does it match expected_answer in substance?
  Wording may differ freely. You are comparing MEANING, not phrasing.
  5  Same substance as expected_answer. Paraphrase is fine.
  4  Correct, with a small omission or an immaterial imprecision.
  3  Partially correct: a substantive element is missing or one detail is
     wrong, but the thrust is right.
  2  Largely incorrect; only a fragment overlaps the expected answer.
  1  Contradicts expected_answer, or is unrelated to it.

completeness -- is it whole and properly finished?
  5  Covers the full expected answer and ends as a finished statement.
  4  Covers the substance; one secondary element missing.
  3  A required element of the expected answer is missing.
  2  Only a fragment of what was expected.
  1  Truncated mid-sentence, or so short as to be unusable.
  NOTE: completeness is about COVERAGE and CLOSURE, not length. A complete
  two-sentence answer scores 5.

=============================================================================
COMBINED
=============================================================================
combined = context_relevance    * 0.20
         + faithfulness         * 0.25
         + answer_correctness   * 0.25
         + answer_relevance     * 0.15
         + completeness         * 0.15

The five weights sum to 1.00. Round to 2 decimal places.

Worked example -- context_relevance 5, faithfulness 4, answer_relevance 4,
answer_correctness 5, completeness 4:

    5 * 0.20  = 1.00   (context_relevance)
    4 * 0.25  = 1.00   (faithfulness)
    5 * 0.25  = 1.25   (answer_correctness)
    4 * 0.15  = 0.60   (answer_relevance)
    4 * 0.15  = 0.60   (completeness)
                -----
    combined  = 4.45

=============================================================================
HALLUCINATION -- judged SEPARATELY, never folded into scores or combined
=============================================================================
Decide with this test, in order:

  1. Does generated_answer assert a SPECIFIC fact -- an article or provision
     number, a figure, a percentage, a deadline, a monetary amount, a named
     institution, or a concrete procedural step?
        No  -> "false".
  2. Does that specific fact appear in retrieved_context?
        Yes -> "false".
        No  -> "true", and quote the fabricated span verbatim into
               hallucination_evidence.

NOT hallucinations:
  - Vague but grounded prose.
  - An incomplete answer (that is completeness, not fabrication).
  - Correct paraphrase or restructuring of the context.
  - Saying the context does not contain the answer.
  - Khmer numerals where the context spells the same number in words, or the
     reverse -- ៣ and បី are the same number.

IS a hallucination:
  - Citing មាត្រា/ប្រការ N when N does not appear in the context.
  - Any deadline, figure or amount absent from the context.
  - Attributing a duty or power to an institution the context does not name.
  - A fact that is true of Cambodian law generally but is NOT in this context.

CONSISTENCY: hallucination and faithfulness measure the same thing from
different angles. If you set hallucination "true", faithfulness should
normally be 3 or below. If you find yourself writing faithfulness 5 with
hallucination "true", re-read the context -- one of the two is wrong.

=============================================================================
EDGE CASES
=============================================================================
- Empty generated_answer: answer_correctness 1, completeness 1,
  answer_relevance 1, faithfulness 5 (nothing was claimed, so nothing was
  fabricated), context_relevance judged normally, hallucination "false".
- A refusal ("this is outside the scope..."): answer_correctness 1,
  completeness 1, faithfulness 5, hallucination "false". Judge
  answer_relevance and context_relevance on their own terms -- a correct
  refusal to an out-of-scope question can still score well on context.
- expected_answer missing or empty: score answer_correctness 3 and say so in
  no other field; judge everything else normally.
- The answer quotes a provision verbatim from the context: that is ideal, not
  padding. Score it on substance like any other answer.

=============================================================================
OUTPUT
=============================================================================
Rewrite {{FILE}} in place with the scored lines updated. Keep it valid JSONL:
one record per line, same line order, same record count, UTF-8, Khmer text
unescaped (not \uXXXX). Do not reformat or pretty-print. Do not touch any
other file. Report how many records you scored when you finish.
```

---

## After each batch — verify before continuing

The failure mode that matters is silent truncation: a batch that looks scored
but whose later records were never actually read. Check every batch.

```bash
python scripts/check_judge_progress.py \
  reports/generator_model_comparison/SeaLLMs-v3-7B-Chat-Q4_K_M/generated_answer_results.jsonl
```

`records` must stay **100** — if it drops, the judge rewrote the file wrong and
you should restore it from git (`git checkout <file>`) and redo that batch.
`scored` must climb by exactly the batch size. Any score outside 1–5 means the
judge drifted from the rubric.

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
