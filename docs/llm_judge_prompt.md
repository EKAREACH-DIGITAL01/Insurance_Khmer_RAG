# LLM-as-judge prompt (RAGAS) — Khmer insurance law RAG

Judges the generated answers from `scripts/13_generate_model_answers.py`
using an LLM instead of the deterministic proxies. Input files come from
`scripts/16_prepare_llm_judge_input.py`.

- **Scope:** 4 models × 100 human-authored gold questions = **400 judgements**.
- **Batching:** 10 records per run → 10 runs per model, 40 runs total.
- **Judge must differ from the model being judged** (never let SeaLLM judge
  SeaLLM) — self-preference bias.

---

## Run it

```bash
codex exec --skip-git-repo-check - < docs/llm_judge_prompt_filled.txt
```

Or paste the prompt below into an interactive `codex` session, substituting
`{{FILE}}`, `{{START}}`, `{{END}}`.

---

## The prompt

```
You are an impartial evaluator for a Khmer-language RAG system that answers
questions about Cambodian insurance law. You are scoring the RETRIEVAL and the
GENERATED ANSWER, not the law itself.

INPUT
Read the file: {{FILE}}
It is a JSON array. Score ONLY records at index {{START}} through {{END}}
inclusive. Do not score any other record.

Each record has:
  question_id        stable id — echo it back exactly
  question           the user's question, in Khmer
  retrieved_context  the chunks the system retrieved (chunk_id + text)
  expected_answer    the human-curated ground-truth answer
  generated_answer   what the model produced — THIS is what you are scoring

RULES
1. Judge the generated_answer ONLY against retrieved_context and
   expected_answer. Do NOT use your own knowledge of Cambodian law, and do not
   reward an answer for being correct in the real world if it is not supported
   by the material provided.
2. The text is Khmer. Do not penalise an answer for being in Khmer, for Khmer
   numerals (០-៩), or for legal phrasing that reads unusual in English.
3. A shorter answer is not automatically worse. Judge substance, not length.
4. If the generated_answer is a refusal or is empty, score Answer Correctness
   and Completeness 1, and say so in the note.
5. Score independently per metric. Do not let one weak metric drag the others.

METRICS — integer 1 to 5, where 1 = very poor and 5 = excellent

context_relevance   Did retrieved_context actually contain what was needed to
                    answer the question? This scores RETRIEVAL, not the answer.
                    5 = everything needed is present; 3 = partially present,
                    key detail missing; 1 = context is off-topic.

faithfulness        Is every factual claim in generated_answer supported by
                    retrieved_context? Fabricated articles, numbers, dates,
                    obligations or procedures are the failure mode.
                    5 = fully grounded; 3 = mostly grounded, one unsupported
                    claim; 1 = substantially fabricated.

answer_relevance    Does generated_answer actually address what was asked?
                    5 = directly answers it; 3 = partly, or answers a related
                    question; 1 = does not address the question.

answer_correctness  How well does generated_answer match expected_answer in
                    substance? Wording may differ freely.
                    5 = same substance; 3 = partially correct, notable omission
                    or error; 1 = contradicts expected_answer.

completeness        Does it cover the full expected answer and finish properly
                    (not cut off mid-sentence)?
                    5 = complete; 3 = a required element missing; 1 = truncated
                    or fragmentary.

HALLUCINATION — judged SEPARATELY, never averaged into the scores above
  hallucination         "yes" or "no". "yes" if generated_answer asserts ANY
                        specific fact — an article number, figure, deadline,
                        obligation, institution or procedure — that does not
                        appear in retrieved_context. Prose that is vague but
                        grounded is NOT a hallucination. Being merely incomplete
                        is NOT a hallucination.
  hallucination_evidence  If "yes", quote the exact fabricated span from
                        generated_answer. If "no", use "".

OUTPUT
Return ONLY a JSON array, no prose before or after, no markdown fence.
One object per scored record, in input order:

[
  {
    "question_id": "guidance_qid_001",
    "context_relevance": 4,
    "faithfulness": 5,
    "answer_relevance": 4,
    "answer_correctness": 3,
    "completeness": 4,
    "hallucination": "no",
    "hallucination_evidence": "",
    "note": "one sentence justifying the lowest score given"
  }
]

Write the array to: reports/llm_judge_evaluation/raw/{{MODEL}}_{{START}}_{{END}}.json
Do not modify any other file. Do not run any script.
```

---

## After all 40 runs

Merge `reports/llm_judge_evaluation/raw/*.json`, then:

- Weight the five metrics: context_relevance .20, faithfulness .25,
  answer_correctness .25, answer_relevance .15, completeness .15.
- Report the hallucination count separately, never folded into the combined
  score.
- Rank the four models by weighted combined score. **This ranking is what
  selects the shipped generator** — the deterministic proxy scorer that
  previously made that call has been removed.

## Validating the judge

An LLM judge on Khmer legal text is not reliable by default. Hand-score ~30
answers yourself and report your agreement with the judge on that sample.
Without it, the result is "an LLM said so"; with it, it is a measured method.
