# Image Generation Prompts

Prompts for regenerating the diagrams in `assets/`. Each one describes **what
the diagram must explain** — the logic of the system — and leaves style,
palette, iconography and layout to the image model.

Kept in version control so the diagrams can be regenerated consistently when
the pipeline changes.

---

## How to use these

The images are produced in two passes:

1. **Image pass.** The image model generates the visual from the prompt below.
2. **Text pass.** Exact filenames, paths and figures are composited on top
   afterward with PIL, read directly from the repository.

The second pass is not optional. Image models corrupt long `snake_case`
tokens — `11_human_golden_question_evaluation.py` reliably comes back
misspelled, and re-prompting does not fix it. Every prompt therefore asks for
empty label space rather than baked-in text.

Values to read from the repo during the text pass, never typed from memory:

| Label content | Source |
|---|---|
| Script filenames | `scripts/` |
| Gate thresholds | `app.py` |
| Corpus and chunk counts | `reports/data_card/data_card.json` |
| Model names | `install_models.py` |
| Result figures | the JSON files under `reports/` |

The fourth generator model is **Sailor2-L 8B Chat**, not Gemma.

If an image returns with garbled text baked in despite the instruction,
regenerate it. A report figure carrying invented filenames is worse than no
figure.

---

## The image set

| File | Job |
|---|---|
| `architecture_overview.png` | The three stages: build → evaluate → serve |
| `chat_request_path.png` | One question's journey, with the three gates |
| `hybrid_retrieval_rrf.png` | Dense + lexical ranked lists fusing via RRF |
| `evaluation_framework.png` | Gold set → 4 models → LLM judge + human validation |
| `corpus_provenance.png` *(optional)* | 49 PDFs → OCR → clean → chunk → 1435 chunks |

The first four are the README set. `corpus_provenance.png` duplicates what the
data card already tabulates — include it in a written report, skip it in the
README.

---

## 1. `architecture_overview.png`

> Design a diagram explaining the overall architecture of a fully-local
> Khmer-language RAG chatbot for Cambodian insurance law. You choose the visual
> style, palette, icons and layout — I'm describing only the logic that must be
> conveyed.
>
> The system has three stages that run at different times, and that distinction
> is the most important thing the diagram must communicate:
>
> **Stage A — build, runs once.** Raw Cambodian legal PDFs (49 documents in four
> categories: laws, sub-decrees, prakas, and guidance) are converted to page
> images, OCR'd by an external API, cleaned at page level, stitched into whole
> documents, cleaned again, split into 1435 clause-aware chunks, and embedded
> into a searchable vector index. This is a linear assembly line of eight steps.
>
> **Stage B — evaluate, runs once.** Six independent evaluations decide which
> components to ship and prove the system works: comparing three embedding
> models, comparing retrieval strategies, having humans validate the test
> question set, generating answers from four candidate LLMs, having an LLM judge
> score those answers, and replaying adversarial questions against the safety
> gates. Every evaluation writes a report. All reports roll up into a final data
> card.
>
> **Stage C — serve, runs continuously.** The live chatbot. It reuses the index
> built in Stage A and the LLM chosen in Stage B. Everything runs locally on the
> user's machine with no external API calls.
>
> Three relationships must be visible: Stage A feeds Stage B (the index and test
> set are what get evaluated); Stage B's decisions determine Stage C's
> configuration; and there is a feedback loop where the human evaluation in
> Stage B corrects the test question set that Stage B itself depends on.
>
> Leave clear empty label space next to every element — exact filenames will be
> added afterward.

---

## 2. `chat_request_path.png`

> Design a diagram showing what happens when a user asks the chatbot one
> question. You choose the style; I'm describing the logic.
>
> The central idea: this system refuses far more often than a normal chatbot,
> deliberately, because giving wrong legal or financial advice is worse than
> giving none. Refusal is a designed feature, not a failure. Three checkpoints
> along the path can stop a question, and each is cheaper than the one after it
> — that ordering is intentional and should be legible.
>
> The path, left to right:
>
> 1. A user question arrives in Khmer.
> 2. **Checkpoint 1 — costs nothing.** Rejects non-Khmer text, questions that are
>    too short, and questions asking for product recommendations, pricing, or
>    personal advice. Pure pattern matching, no AI involved.
> 3. The question is converted to a vector by a local embedding model.
> 4. Two searches run in parallel over the document collection — one by meaning,
>    one by exact keyword — and their results are combined into a single ranked
>    list.
> 5. **Checkpoint 2 — costs one search.** If the best match is too weak, the
>    question is off-topic and gets refused here, before the expensive step.
> 6. The top five passages are assembled into a prompt.
> 7. A local LLM generates an answer from those passages only.
> 8. **Checkpoint 3 — costs a full generation.** The answer is checked before it
>    reaches the user.
> 9. The answer is delivered with citations pointing back to the specific source
>    documents.
>
> Show the three refusal branches leaving the main path, drawn as deliberately as
> the success path. Show that the cost of stopping increases at each checkpoint.

---

## 3. `hybrid_retrieval_rrf.png`

> Design a diagram explaining a hybrid document retrieval method. You choose the
> style; I'm describing the logic.
>
> The problem: searching Khmer legal text by *meaning* alone misses exact
> technical terminology, and searching by *keyword* alone misses paraphrased
> questions. Neither is sufficient. The solution runs both and merges them.
>
> One user question feeds two independent searches at once:
>
> - A **semantic search** that compares meaning mathematically and returns its
>   own ranked list of passages.
> - A **keyword search** that matches literal words and returns a different
>   ranked list of the same document pool.
>
> The two ranked lists are then merged by a fusion method that scores each
> passage by its *position* in each list rather than by its raw score — so the
> two incomparable scoring systems can be combined fairly. A passage ranked
> highly by both methods rises above one ranked highly by only a single method.
> That reinforcement effect is the key insight the diagram must convey.
>
> The merged list produces a final ranking, and only the top five passages
> proceed to the language model.
>
> Show clearly that the two searches are independent and simultaneous, that they
> can disagree, and that agreement between them is what gets rewarded.

---

## 4. `evaluation_framework.png`

> Design a diagram explaining how a RAG chatbot was evaluated before shipping.
> You choose the style; I'm describing the logic.
>
> Everything depends on one shared resource at the center: a set of 400
> gold-standard questions about Cambodian insurance law, each paired with the
> passages that should be retrieved and the answer that should be given. This is
> the measuring stick, so it was itself validated by people — 11 human evaluators
> produced 550 ratings across all 400 questions, scoring each on clarity,
> relevance, difficulty, answer quality, and uniqueness, and correcting the
> target answers. That correction flows back into the question set before
> anything else uses it. The diagram should show that the yardstick was checked,
> not assumed.
>
> Three evaluation tracks branch from that validated question set:
>
> **Retrieval track.** Three candidate embedding models compete on the same
> questions; the winner becomes the production retriever. Then two retrieval
> strategies compete; the hybrid approach wins.
>
> **Generation track.** Four candidate local LLMs each answer the same 100
> human-written questions under identical conditions. A separate LLM judge then
> scores every answer on five quality dimensions, plus a separate flag for
> fabricated content that is deliberately kept out of the combined score. One
> model wins and ships.
>
> **Safety track.** A set of adversarial questions — requests for
> recommendations, pricing, personal advice, and off-topic questions — is
> replayed against the system's refusal gates. Results are classified four ways:
> correctly refused, correctly answered, wrongly answered, wrongly refused.
>
> All three tracks write reports, and every report feeds a final summary data
> card.
>
> The diagram should convey that generation was judged by a machine while the
> question set was judged by humans — different kinds of evidence with different
> weight.

---

## 5. `corpus_provenance.png` *(optional)*

> Design a diagram showing how a document collection was built. You choose the
> style; I'm describing the logic.
>
> 49 official Cambodian legal PDFs in four categories — laws, sub-decrees,
> prakas (ministerial regulations), and guidance documents — become 1435
> searchable passages.
>
> The transformation narrows and cleans in stages: scanned pages that are
> imperfect and skewed are read by OCR; page-level artifacts like headers and
> page numbers are stripped away; pages are stitched back into whole documents;
> documents are cleaned again and abbreviations expanded; and finally each
> document is split along its own legal structure — by article and clause for
> laws, by heading hierarchy for guidance — rather than by arbitrary length.
>
> The result is four groups of very uneven size: prakas is by far the largest at
> roughly 780 passages, laws and sub-decrees are similar to each other at around
> 250 each, and guidance is smallest at roughly 145. That imbalance is real and
> should be visible, not smoothed out.
>
> Convey that the chunking respects the documents' own legal structure — this is
> not generic text splitting.

---

## Wrapper instruction

Wrap the prompts above with this when driving the generation from a CLI agent:

> Generate the images for `assets/` using the OpenAI image API, one per prompt in
> `docs/image_generation_prompts.md`. Then, in a second pass, write
> `scripts/render_diagram_labels.py` that composites the exact text labels onto
> each generated image using PIL.
>
> All label text must be read from the repository, never typed from memory:
> script filenames from `scripts/`, thresholds from `app.py`, corpus and chunk
> counts from `reports/data_card/data_card.json`, model names from
> `install_models.py`, and result figures from the JSON files under `reports/`.
> Do not let the image model render filenames, paths, or numbers — it will
> corrupt them.
>
> Use a single monospace face for all composited labels. Keep the script
> deterministic and re-runnable so the diagrams can be regenerated when the
> pipeline changes.
>
> Show me each image after the text pass before committing anything.
