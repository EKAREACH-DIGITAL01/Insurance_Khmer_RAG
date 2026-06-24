# Adversarial Test Set

Tests scope-boundary enforcement under "creative user queries," per the
Internship Proposal §11 requirement. Each file mixes `expected_behavior:
"refuse"` and `"answer"` cases deliberately -- testing only refusals would
miss false positives (legitimate questions wrongly blocked because they
share vocabulary with a violation category).

## Schema

```json
{
  "question": "...",
  "expected_behavior": "refuse" | "answer"
}
```

`"refuse"` means the chatbot should NOT generate an answer from context --
whether that's because of the deterministic `is_scope_violation()` gate,
the deterministic `contains_khmer_script()` / length gates, or the
score-based `OUT_OF_DOMAIN_THRESHOLD` gate in `app.py`. This file doesn't
distinguish which gate fired, only whether the final behavior was correct
-- the diagnostics page (or generator eval script's raw output) is where
to look if a refusal needs debugging.

## Provenance

Every question here was either manually tested against the live app
during development (see project chat history) or directly copied from
`app.py`'s own `OUT_OF_DOMAIN_THRESHOLD` calibration log comment.

## Known caveat: `out_of_domain_questions.json`

Five of the `"refuse"` cases in that file (the ones about the assistant
itself, off-topic chit-chat, and emotional distress) do **not** trip any
deterministic gate (`contains_khmer_script`, length, or
`is_scope_violation`) -- they are valid Khmer text with no scope-violation
phrasing. They are expected to refuse via the **score-based**
`OUT_OF_DOMAIN_THRESHOLD` gate instead, because their FAISS similarity
score is too low. That can only be verified against a live FAISS index +
embedding model, not by static pattern inspection -- which is exactly
what was done before shipping this file (see project chat history for the
verification run). If `OUT_OF_DOMAIN_THRESHOLD` or the corpus changes
significantly, re-verify these five against the live app rather than
assuming the label still holds.
