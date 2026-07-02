# Generation Robustness & Eval/Production Parity

This document records a set of changes made to keep the offline generator
evaluation (`scripts/13_compare_generator_models.py`) faithful to the live
chatbot (`app.py`), and to make generation robust against the failure modes
observed during real runs — repetition loops, silent truncation, a macOS
segfault, and an intermittent `llama.cpp` parser error.

The guiding principle throughout: **script 13 must generate the same way the
live app does**, so that a model's comparison score predicts how it will behave
for real users. Where the two files *must* differ, the difference is explained
below and is deliberate.

---

## 1. Why parity matters

The next stage after the model comparison is a **human evaluation UI** that
shows evaluators the `expected_answer` alongside the `generated_answer` produced
by script 13. The client, separately, uses the live `app.py` chatbot. If the two
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

**Fix:** script 13 now loads `SentenceTransformer(EMBEDDING_HF_ID, device="cpu")`.
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
already guarded against this at its top; script 13 did not, and crashed on the
first non-resumed question.

**Fix:** at the very top of script 13, before importing `faiss` or `torch`:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import faiss
faiss.omp_set_num_threads(1)
```

These must stay in sync with `app.py`.

---

## 4. Real-time result ordering (script 13)

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

### Eval (script 13): size to the gold answer

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

**Fix (script 13): resume re-processes failed rows.** A row that recorded a hard
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

## 10. Per-model gold scope — manual-only vs. full (script 13)

The gold set has two tiers per category: the first 25 questions are purely
**human-authored** (question, expected chunks, *and* expected answer), and
question 26 onward are **AI-generated**. Across the four categories that's
**100 manual** and **300 AI** questions (400 total).

Most models are compared on the manual-only 100 — that's the clean,
trusted set. But SeaLLM (the model shipped to the client) is additionally
evaluated over the **full 400** so its behaviour is measured on the wider,
messier question distribution real users resemble. To keep the two runs from
colliding, the full run writes to a **separate set of files**.

**Mechanism:**

- A model opts into the full set with `"gold_scope": "full"` in `LLM_MODELS`
  (currently only SeaLLM). Everyone else defaults to the manual set.
- `main()` builds **both** lists once: `gold_entries_manual` (the 100) and
  `gold_entries_full` (the 400). A `scope_for(info)` helper returns the
  `(suffix, gold_entries)` pair for each model.
- A module-level `RESULTS_SUFFIX` (set per-model right before that model runs,
  since models run strictly one at a time) flows into the filename helpers:
  - manual scope → `generated_answer_results.jsonl` / `..._summary.json` / `.md`
  - full scope   → `generated_answer_results_full.jsonl` / `..._full_summary.*`
- **Resume is automatic and free.** The 100 manual SeaLLM answers were copied
  into `..._full.jsonl` first; `load_existing_results()` keys off
  `question_id`, sees those 100 as done, and generates only the ~300 AI
  questions. `reorder_results_file()` then slots everything into full gold
  order.
- The token ceiling self-sizes: `compute_gold_ceiling()` runs over whichever
  list the model uses, so the full run's ceiling reflects the longest of all
  400 answers.

> AI questions may lack an `expected_answer`. That's handled: such entries are
> excluded from the ceiling calc, and their `answer_correctness` scores as
> `None` (genuinely uncomputable, §1's normalisation), so they neither crash
> nor distort the summary.

> Comparison caveat: the final comparison table then mixes a 400-question
> SeaLLM summary with 100-question summaries for the others. The metrics are
> per-question means so they stay comparable, but SeaLLM's combined score now
> reflects the harder AI questions too — bear that in mind when reading the
> ranking.

### Both result files get their own summary

A model that has *both* result files on disk (SeaLLM: the 100-question
`generated_answer_results.jsonl` **and** the 400-question
`generated_answer_results_full.jsonl`) gets a summary for **each**, written to
its matching name:

| Result file | Summary json | Summary md |
|---|---|---|
| `generated_answer_results.jsonl` | `generated_answer_results_summary.json` | `..._summary.md` |
| `generated_answer_results_full.jsonl` | `generated_answer_results_full_summary.json` | `..._full_summary.md` |

`generate_all_summaries(label)` loops the suffixes `("", "_full")`, loads
whichever result file exists (via `load_all_rows()`, which keeps error rows so
`summarise()` can count them), and writes that scope's summary. It runs after a
model finishes **and** when a model is skipped as already-done, so both summary
files stay current even on a no-op resume run. The cross-model comparison still
uses each model's **primary-scope** summary (full for SeaLLM, manual for the
rest), so the final table has exactly one row per model.

---

## 11. Server-launch guards: stale server, context cap, Ctrl+C (both files)

This area saw several iterations; the **final** design is below.

### Stale / wrong server on port 8080

A leftover `llama-server` (a crashed run, a manually-started one, or script 13's
server for a previous model) can still hold **port 8080**. The new process then
fails to bind, dies, and a naive readiness poll — which only checks that
*something* answers on 8080 — would attach to the wrong server.

Guards (both `start_chatbot.py` and `scripts/13_compare_generator_models.py`):

1. **Free port 8080 before launching.** A helper finds whatever is listening on
   8080 and kills it (SIGTERM then SIGKILL on POSIX via `lsof`/`os.kill`;
   `netstat`/`taskkill` on Windows).
2. **Detect our own process dying.** The readiness loop checks `process.poll()`;
   if our `llama-server` exits during startup it aborts with a clear message
   instead of polling into some other server.
3. **Verify the server is OURS by the loaded model name.** After ready, it reads
   `/props` and confirms `model_path`'s filename matches the GGUF we launched.
   This — **not** the context size — is the reliable stale-server signal.
4. `--ctx-size` is driven by `str(SERVER_CTX_SIZE)` so the flag can't drift from
   the constant.

### Context size is the *model's* cap, not a stale-server signal

An earlier version aborted whenever the live `n_ctx` was below the requested
16384, on the theory that a smaller window meant a stale server. That was a
**false-positive bug**: llama.cpp clamps `--ctx-size` down to a model's own
trained context. The plain **`Sailor2-8B-Chat` GGUF is built at
`context_length = 4096`**, so it legitimately came up at `n_ctx=4096` and was
wrongly rejected as "stale".

Final behaviour: `verify_server()` **warns** (does not abort) when the correct
model reports `n_ctx < SERVER_CTX_SIZE`, noting it's the GGUF's own limit. A
small-context model can then still be evaluated on whatever fits.

> **Sailor fix.** Sailor2-8B at 4096 ctx cannot hold the 5k–13k-token RAG
> prompts (most 400'd). The model was swapped to **`Sailor2-L-8B-Chat`** — the
> "L" long-context variant, natively `131072` ctx, no rope scaling — in
> `install_models.py`, `scripts/13`, and `start_chatbot.py`
> (`local_filename = Sailor2-L-8B-Chat-Q4_K_M.gguf`, label `Sailor2-L 8B`). The
> `_is_sailor()` "sailor" substring match still applies the §13 style rule.

### Clean shutdown under Ctrl+C

`stop_llama_server()` (script 13) is interrupt-proof: SIGTERM → wait loop that
**swallows a second Ctrl+C** so the kill always finishes → SIGKILL if needed →
`_free_port_8080()` as a backstop. The per-model `process` is assigned **inside**
the `try`, so the `finally` stops the server even if Ctrl+C lands during
startup. A top-level handler prints a clean message and frees the port instead
of dumping a traceback. Net effect: a single (or impatient double) Ctrl+C leaves
no orphaned server on 8080.

**Recovery in general.** Failed questions are written as error rows and
`load_existing_results()` skips them (§8), so re-running regenerates only the
failures — no manual cleanup.

---

## 12. Sailor-only output style rule

Sailor alone tended to dress answers up — an invented bold heading, markdown
emphasis, injected source citations/dates pulled from chunk metadata (e.g.
`(មាត្រា ១)`, `(2023-10-17)`), and a literal closing tag like `(ចប់) ៕`. The
content was correct, but it read as embellishment next to the plain gold answer,
which matters for the human-eval UI.

`build_prompt(..., model_id=...)` appends a `SAILOR_STYLE_RULES` block **only**
when `_is_sailor(model_id)` matches (case-insensitive "sailor"), so the base
prompt every other model sees is byte-for-byte unchanged and their existing
results stay valid. The rule forbids headings/markdown, source/date citations,
and closing tags, and tells the model to match the source's wording. It is
duplicated identically in `app.py` and `scripts/13`. (Trade-off on record:
Sailor now gets one instruction block the others don't, so the prompt is no
longer perfectly identical across models — a deliberate choice to clean up
Sailor's house style.)

---

## 13. Known limitation: Qwen3.5 reasoning + the PEG parser

`Qwen3.5-9B` is a **reasoning** model. To return the clean, short answers we
want, llama.cpp must *extract* its `<think>…</think>` out of the content — and
that extraction parser (chat format `peg-native` for the `qwen35` arch)
**stochastically crashes** with `"does not match the expected peg-native
format"`. It clears on retry for most prompts (~1% affected), but a few prompts
(e.g. `prakas_qid_021`) trigger it almost every time, so retries don't help.

Investigated alternatives and why each was rejected:

- `--skip-chat-parsing` — does **not** prevent it.
- `--chat-template chatml`, `--no-jinja`, `--reasoning-format none`,
  `--reasoning-budget 0` — all **eliminate** the crash, but Qwen then emits its
  full verbose `<think>` reasoning into the content (≈1000 tokens), which is
  slow and can crowd out the answer. `--reasoning off` only suppresses thinking
  under the native template — the one with the bug.

Decision: **leave Qwen on its original config** (`--jinja --reasoning off`,
plain streaming) to preserve the already-generated answers. The rare hard
failure is accepted as a known limitation rather than regenerating everything in
a slower thinking mode or swapping the model. (Non-reasoning peers like SeaLLM,
which is Qwen2-based, do not have this issue.)

---

## 14. Summary of where each change lives

| Change | `app.py` / `start_chatbot.py` | `scripts/13_compare_generator_models.py` |
|---|---|---|
| Embedding device pinned to CPU | already had | **added** |
| macOS OpenMP guard | already had | **added** |
| Real-time gold-order JSONL | n/a | **added** |
| Dynamic `max_tokens` | by retrieved chunk | by gold answer |
| `frequency_penalty = 0.3` | **added** | **added** |
| "End with ។" prompt rule | **added** | **added** |
| Retry on transient gen error | **added** (pre-first-token) | **added** (full re-roll) |
| Resume re-processes error rows | n/a | **added** |
| Per-model gold scope (manual/full) | n/a | **added** |
| Summary per result file (manual + full) | n/a | **added** |
| Free port 8080 before launch | **added** (`start_chatbot.py`) | **added** |
| Verify server by loaded **model name**; warn (not abort) on model ctx cap | **added** (`start_chatbot.py`) | **added** |
| Ctrl+C-proof server shutdown + port backstop | n/a | **added** |
| Sailor-only style rule | **added** (`app.py`) | **added** |
| Sailor → Sailor2-L (128K ctx) | **added** (`start_chatbot.py` + `install_models.py`) | **added** |

Shared constants (`GEN_FREQUENCY_PENALTY`, `MAX_TOKENS_FLOOR`,
`GEN_TOKEN_HEADROOM`, `SERVER_CTX_SIZE`, `PROMPT_TOKEN_MARGIN`,
`SERVER_OUTPUT_HARD_CAP`, `GEN_MAX_RETRIES`, `GEN_RETRY_BACKOFF`) and the
`build_prompt()` text (including the Sailor-only block) must stay identical
between the two files. The `--ctx-size` launch flag is driven by
`SERVER_CTX_SIZE` in both launchers, so the server window and the token-budget
maths can never disagree.
