# Generation Robustness & Eval/Production Parity

This document records a set of changes made to keep the offline generator
evaluation (`scripts/11_compare_generator_models.py`) faithful to the live
chatbot (`app.py`), and to make generation robust against the failure modes
observed during real runs — repetition loops, silent truncation, a macOS
segfault, and an intermittent `llama.cpp` parser error.

The guiding principle throughout: **script 11 must generate the same way the
live app does**, so that a model's comparison score predicts how it will behave
for real users. Where the two files *must* differ, the difference is explained
below and is deliberate.

---

## 1. Why parity matters

The next stage after the model comparison is a **human evaluation UI** that
shows evaluators the `expected_answer` alongside the `generated_answer` produced
by script 11. The client, separately, uses the live `app.py` chatbot. If the two
pipelines diverged, the comparison would no longer predict production behaviour
and evaluators would see artifacts the real app never produces. So the retrieval
path, prompt, and generation settings are kept byte-for-byte aligned, and any
shared constant carries a comment saying it must stay in sync.

---

## 2. Retrieval determinism — embedding device pinned to CPU

`app.py` loads the BGE-M3 embedding model with `device="cpu"`. Script 11
originally left the device unspecified, which on Apple Silicon auto-selects
**MPS**. CPU and MPS produce slightly different float embeddings, enough to flip
the FAISS top-K ordering on near-tie chunks — which changes the selected context
and therefore the generated answer.

**Fix:** script 11 now loads `SentenceTransformer(EMBEDDING_HF_ID, device="cpu")`.
Retrieval is now bit-for-bit comparable between the two pipelines: same
embeddings → same FAISS order → same fused chunks → same context → same prompt.

> Note: identical *input* does not guarantee identical *output*. Generation is
> sampled at `temperature=0.2` and `llama.cpp` is not bit-reproducible across
> processes, so answer wording can still vary run to run. The metrics are
> robust to this (semantic similarity, chunk-ID overlap, number overlap), so the
> comparison stays valid.

---

## 3. macOS OpenMP segfault guard

On macOS, FAISS and PyTorch (pulled in by `sentence-transformers`) each load
their own OpenMP runtime. The duplicate runtimes collide and **segfault** the
moment the first real FAISS search and BGE-M3 encode run together. `app.py`
already guarded against this at its top; script 11 did not, and crashed on the
first non-resumed question.

**Fix:** at the very top of script 11, before importing `faiss` or `torch`:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
faiss.omp_set_num_threads(1)
```

These must stay in sync with `app.py`.

---

## 4. Real-time result ordering (script 11)

Answers are written to the per-model `generated_answer_results.jsonl` the
instant they are generated (crash safety). That means a question regenerated on
a resumed run — e.g. after manually deleting its line — used to be **appended at
the bottom**, out of gold order.

**Fix:** `append_result()` now re-sorts the file into gold order (keyed by
`question_id`) immediately after each write (`reorder_results_file()`). The file
is always in correct order on disk, after every single question, while remaining
crash-safe. Rows whose key isn't in the gold set are preserved (appended after
the ordered ones), and duplicates dedupe last-write-wins.

---

## 5. Dynamic output-token cap

`max_tokens` was a fixed `1024` in both files. Two problems:

- **Truncation.** Khmer tokenizes very inefficiently in these quantized GGUF
  models (byte-fallback splits each cluster into several tokens), so a long
  answer can need well over 1024 output tokens and was being silently cut off.
- **Latency.** On this hardware a larger cap measurably raises **first-token
  time** (the server reserves KV/compute buffers up front), so a flat high cap
  needlessly taxes the short-answer majority. (Empirically observed: 1024 →
  ~60s TTFT, 2048 → ~130s TTFT for the same prompt.)

The cap is now sized **per question**, with each model's *own* tokenizer (via
the live server's `/tokenize`), then clamped to a floor and a ceiling and
finally bounded so `prompt + output` can never exceed the context window.

### Eval (script 11): size to the gold answer

A gold answer is a verbatim chunk passage, so the longest answer the eval could
need is the longest `expected_answer`. `compute_gold_ceiling()` tokenizes
**every** gold answer (across all category files, since `gold_entries`
aggregates `rglob` over `data/09_evaluation/*/gold_questions.json`) and sets the
ceiling to `max_gold_answer_tokens × GEN_TOKEN_HEADROOM`. Per question, the cap
is `expected_tokens × 1.5`, clamped to `[1024, gold_ceiling]`.

Because the ceiling is read from the data every run, **adding a longer gold
answer later automatically raises it** — the script self-adjusts and never
silently truncates a newly-lengthened answer.

### Production (`app.py`): size to the retrieved context

Production has **no expected answer** — the client can ask anything. The runtime
signal that *is* available is the retrieved context: the answer is grounded in
(and faithfully bounded by) the chunks we retrieved. `choose_max_tokens_live()`
sizes the cap to the **largest retrieved chunk** × headroom, clamped to
`[1024, hard cap]` and bounded by the context window.

- asks about a long provision → retrieves a big chunk → big cap → no truncation
- asks something with small chunks → small cap → faster first token
- out-of-domain → gated before generation, no cap needed

This is the production analog of the eval's gold-answer sizing, using the only
signal available at chat time.

### Shared constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_TOKENS_FLOOR` | 1024 | never below this |
| `GEN_TOKEN_HEADROOM` | 1.5 | answers run longer than their reference/source |
| `SERVER_CTX_SIZE` | 16384 | must match `start_chatbot.py` `--ctx-size` |
| `PROMPT_TOKEN_MARGIN` | 256 | gap so `prompt + max_tokens < ctx size` |
| `SERVER_OUTPUT_HARD_CAP` | `SERVER_CTX_SIZE - 9000` (~7384) | leaves room for a ~9k prompt |

> Hard limit: with a ~9k-token retrieval prompt and a 16384 context window,
> only ~7k output tokens are ever available. The very longest provisions are
> normally returned by the deterministic provision fast-path (which bypasses the
> LLM and the cap entirely). If non-provision long answers still truncate, raise
> `--ctx-size` in `start_chatbot.py`/`start_llama_server` **and** `SERVER_CTX_SIZE`
> to match.

---

## 6. Repetition-loop suppression — `frequency_penalty`

Small quantized 7–9B models occasionally fall into a **repetition loop**:
emitting the same sentence over and over until they hit `max_tokens`. This is a
stochastic failure at `temperature=0.2` — the same question can answer cleanly
on another run. It is unacceptable for the client UI and for the human-eval
display.

**Fix:** both files now generate with `frequency_penalty = GEN_FREQUENCY_PENALTY`
(`0.3`). `frequency_penalty` scales with how many times a token has already been
produced, so it breaks a runaway loop within a few repeats while leaving normal
legal-term reuse (which only repeats a handful of times) largely untouched —
unlike `presence_penalty`, which flatly penalizes any reused token and would
distort Khmer legal vocabulary. Raise toward 0.6–0.8 if loops still appear.

Verified on `guidance_qid_012`: the looped 7×-repeated answer became a clean,
distinct numbered list; `faithfulness` rose 3 → 5 and the answer ended cleanly.

---

## 7. "Always end with ។" prompt rule

The `completeness` metric flags an answer whose final character isn't Khmer
sentence-final punctuation (`។`/`៕`). Models sometimes produced the full answer
but dropped the trailing `។` on the last list item (e.g. `guidance_qid_007`).

**Fix:** a rule was added to `build_prompt()` in **both** files (the prompt is
duplicated and must stay byte-for-byte identical):

> *Always finish your answer with a complete sentence that ends in the Khmer
> sentence-final sign ។ (or ៕). Even when the answer is a list, the final item
> must end with ។. Never stop on a word, bracket, or list item without it.*

This does **not** mask incomplete answers: real truncation is an external cut
(no `។` possible), and the `completeness` length-ratio check plus
`answer_correctness`/`faithfulness` independently penalize thin answers.

---

## 8. Transient `--jinja` PEG-parser error — retry + resume

With `--jinja`, `llama-server` parses streamed output through the chat-template
PEG parser and intermittently aborts a request with
`"does not match the expected peg-native format"` when the model samples output
it can't reconcile. It is stochastic — a re-request almost always succeeds.

**Fix (both files): retry.** `GEN_MAX_RETRIES = 3`, `GEN_RETRY_BACKOFF = 2.0s`.

- **Script 11:** `generate_answer()` re-rolls from scratch on any transient
  failure (temperature 0.2 makes each attempt sample a different, usually-valid
  trajectory). Only if all attempts fail is a hard error recorded.
- **`app.py`:** the live path streams tokens to the screen, so it retries **only
  while nothing has been streamed yet** (`first_token_time is None and not
  answer_text`). Once a token is on screen a later failure propagates rather than
  restarting mid-answer. PEG errors almost always fail *before* the first token,
  so this catches the vast majority.

**Fix (script 11): resume re-processes failed rows.** A row that recorded a hard
`error` is no longer treated as "done" by `load_existing_results()`, so the next
run regenerates it automatically instead of skipping it forever. The stale error
row is overwritten by the fresh result (reorder dedupes last-write-wins). You no
longer need to delete a failed row by hand.

---

## 9. Caveat surfaced during testing: gold-label narrowness

While testing, `guidance_qid_012` exposed a *data* issue, not a code one. The
question ("what must a broker do to partner with an insurer?") was broad, but the
gold set marked only the narrow training-requirement chunk (`guidance_02_chunk_5`)
as expected. Retrieval correctly surfaced a fuller, on-topic procedural provision
(`prakas_13_chunk_23`, ប្រការ ២៤), and the model answered from it — a legitimate
answer that the cosine-to-gold `answer_correctness` metric under-scored.

Takeaway: when a generated answer looks "wrong," check whether retrieval surfaced
the expected chunk (`context_relevance`) before blaming generation. If the chunk
was retrieved and the answer is factually grounded, the gold label may be too
narrow for the question — a gold-data fix, not a pipeline fix. See
`docs/gold_questions_methodology.md`.

---

## 10. Summary of where each change lives

| Change | `app.py` | `scripts/11_compare_generator_models.py` |
|---|---|---|
| Embedding device pinned to CPU | already had | **added** |
| macOS OpenMP guard | already had | **added** |
| Real-time gold-order JSONL | n/a | **added** |
| Dynamic `max_tokens` | by retrieved chunk | by gold answer |
| `frequency_penalty = 0.3` | **added** | **added** |
| "End with ។" prompt rule | **added** | **added** |
| Retry on PEG error | **added** (pre-first-token) | **added** (full re-roll) |
| Resume re-processes error rows | n/a | **added** |

Shared constants (`GEN_FREQUENCY_PENALTY`, `MAX_TOKENS_FLOOR`,
`GEN_TOKEN_HEADROOM`, `SERVER_CTX_SIZE`, `PROMPT_TOKEN_MARGIN`,
`SERVER_OUTPUT_HARD_CAP`, `GEN_MAX_RETRIES`, `GEN_RETRY_BACKOFF`) and the
`build_prompt()` text must stay identical between the two files.
