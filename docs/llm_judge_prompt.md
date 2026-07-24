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

Substitute `{{FILE}}`, `{{START}}`, `{{END}}`.

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

For each record in range, fill in "scores" (five integers 1-5), "combined"
(the weighted average, rounded to 2 decimals), "hallucination" ("true" or
"false") and "hallucination_evidence". Change nothing else -- not the question,
the context, the answers, the timing, nor any record outside the range.

RULES
1. Judge generated_answer ONLY against retrieved_context and expected_answer.
   Do NOT use your own knowledge of Cambodian law. An answer that is true in
   the real world but unsupported by the provided material is NOT faithful.
2. The text is Khmer. Do not penalise Khmer script, Khmer numerals (០-៩), or
   legal phrasing that reads oddly in English.
3. A shorter answer is not automatically worse. Judge substance, not length.
4. If generated_answer is empty or a refusal, score answer_correctness and
   completeness 1.
5. Score each metric independently. Do not let one weak metric drag the rest.

METRICS -- integer 1 to 5 (1 = very poor, 5 = excellent)

context_relevance   Did retrieved_context contain what was needed to answer
                    the question? Scores RETRIEVAL, not the answer.
                    5 = everything needed is present; 3 = key detail missing;
                    1 = off-topic context.

faithfulness        Is every factual claim in generated_answer supported by
                    retrieved_context? Fabricated articles, numbers, dates,
                    obligations or procedures are the failure mode.
                    5 = fully grounded; 3 = one unsupported claim;
                    1 = substantially fabricated.

answer_relevance    Does it address what was actually asked?
                    5 = directly; 3 = partly / answers a related question;
                    1 = does not address it.

answer_correctness  Does it match expected_answer in substance? Wording may
                    differ freely. 5 = same substance; 3 = notable omission or
                    error; 1 = contradicts expected_answer.

completeness        Does it cover the full expected answer and finish properly
                    (not cut off mid-sentence)? 5 = complete; 3 = a required
                    element missing; 1 = truncated or fragmentary.

combined = context_relevance*0.20 + faithfulness*0.25
         + answer_correctness*0.25 + answer_relevance*0.15
         + completeness*0.15

HALLUCINATION -- judged SEPARATELY, never folded into scores or combined
  "true"  if generated_answer asserts ANY specific fact -- an article number,
          figure, deadline, obligation, institution or procedure -- that does
          not appear in retrieved_context.
  "false" otherwise. Vague but grounded prose is NOT a hallucination.
          Being merely incomplete is NOT a hallucination.

  hallucination_evidence
          If "true", quote the exact fabricated span from generated_answer --
          verbatim, no paraphrase, so it can be checked against the context.
          If "false", use "".

OUTPUT
Rewrite {{FILE}} in place with the scored lines updated. Keep it valid JSONL:
one record per line, same line order, same record count, UTF-8, Khmer text
unescaped. Do not reformat or pretty-print. Do not touch any other file.
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
