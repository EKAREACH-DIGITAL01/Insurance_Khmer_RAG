# Image Generation Prompts

> **Scratch file — delete once the images are generated.** This is a working
> brief for driving an image agent, not project documentation. Nothing in the
> repo links to it, so it can be removed without leaving a dangling reference.
> If you keep it, keep it current: a prompt file describing a system that has
> since changed is worse than no prompt file, because the next regeneration
> will quietly produce diagrams of the old design.

Prompts for the diagrams in `assets/`, for both the README and the internship
report. Each one describes **only what must be in the figure** — the elements,
the relationships between them, and the idea a reader should come away with.

Kept in version control so the diagrams can be regenerated consistently when
the pipeline changes.

---

## Read this before generating anything

### These prompts describe content, not appearance

Nothing here specifies a style, palette, colour, icon set, shape, or layout,
and nothing added to it should. Those decisions belong to the image model —
it is better at them than a written brief is, and a prompt that dictates them
produces the same stiff diagram every time.

A prompt in this file says *what the figure contains and what it must make
clear*. If you find yourself writing "blue boxes", "left to right", or "draw
this as a grid", delete it.

The one exception is functional, not decorative: every prompt asks for **empty
label space**, because exact text is composited afterward (see below).

### Two passes, always

1. **Image pass.** The image model generates the visual from the prompt.
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
| Gate thresholds, `TOP_K`, `CONTEXT_K`, `RRF_K` | `app.py` |
| Corpus, chunk and gold-question counts | `reports/data_card/data_card.json` |
| Model names | `install_models.py` |
| Retrieval scores | `reports/embedding_model_comparison/*.json`, `reports/retrieval_mode_comparison/*.json` |
| Generator scores, hallucination rates | `reports/generator_model_comparison/generator_model_comparison.json` |
| Safety-gate outcomes | `reports/safety_gate_evaluation/safety_gate_evaluation.json` |
| Human-evaluation means and agreement | `reports/human_golden_question_evaluation/*.json` |

The fourth generator model is **Sailor2-L 8B Chat**, not Gemma.

If an image returns with garbled text baked in despite the instruction,
regenerate it. A report figure carrying invented filenames is worse than no
figure.

### What must NOT be image-generated

This matters more for the report than for the README, so it is stated plainly:

- **Never image-generate a results chart.** An image model asked for "a bar
  chart of the four models' scores" will invent bar heights that look
  plausible and are wrong. In a research report that is fabricated evidence,
  and a reader who checks the numbers against your own JSON will find the
  mismatch. Every chart in [Results figures](#results-figures-plot-these-do-not-generate-them)
  is **plotted from the report JSON with matplotlib** instead.
- **Never image-generate a screenshot of the app.** A rendered "UI mockup"
  showing a chat interface that does not match the real one misrepresents the
  artifact. Use real captures — see [Screenshots](#screenshots-capture-these).
- The diagram prompts below are safe to generate because they carry *structure
  and relationships*, not measured values. Any number in them is composited in
  the text pass from the repo.

---

## The figure set

**README set** — the four that explain the system to someone landing on the repo:

| File | Job |
|---|---|
| `architecture_overview.png` | The three stages: build → evaluate → serve |
| `chat_request_path.png` | One question's journey, with the three gates |
| `hybrid_retrieval_rrf.png` | Dense + lexical ranked lists fusing via RRF |
| `evaluation_framework.png` | Gold set → 4 models → LLM judge + human validation |

**Report set** — deeper figures worth a page each in the internship report:

| File | Job |
|---|---|
| `corpus_provenance.png` | 49 PDFs → OCR → clean → chunk → 1435 chunks |
| `chunking_strategy.png` | Clause-aware splitting vs. fixed-length splitting |
| `khmer_retrieval_problem.png` | Why Khmer script breaks ordinary lexical search |
| `safety_gate_ladder.png` | The three gates as a cost ladder, with outcomes |
| `context_assembly.png` | Top-10 retrieved → top-5 selected → prompt |
| `local_first_deployment.png` | What runs on-device vs. the one-time external call |
| `app_shell_anatomy.png` | The chat UI's regions and what each is for |
| `conversation_persistence.png` | Write-through from session state to SQLite |
| `answer_provenance_chain.png` | Answer → cited document → source PDF → diagnostics |

**Extra report figures** — added for a long-form (~100 page) report, where the
sets above leave stretches of text with no figure at all:

| File | Job |
|---|---|
| `ocr_and_cleaning_pipeline.png` | The six data stages between a scanned page and a clean document |
| `gold_question_lifecycle.png` | How one test question is authored, validated, corrected, and used |
| `llm_judge_rubric.png` | Five scored dimensions, plus the hallucination flag held outside the score |
| `request_sequence.png` | The same request as a sequence across processes, not boxes |
| `data_model.png` | What is stored per conversation and per message |
| `scope_boundary.png` | In-domain and answerable vs. in-domain but not permitted |
| `deployment_topology.png` | The two local processes, their ports, and who may reach them |

---

## Where each figure goes

A ~100-page report wants roughly one figure every two to three pages. The full
set is **39 images** — 20 generated diagrams, 8 screenshots, 11 plotted charts —
and this is where each earns its place.

| Report chapter | Figures |
|---|---|
| Introduction / problem | `scope_boundary`, `khmer_retrieval_problem` |
| Background & related work | `hybrid_retrieval_rrf` |
| Corpus construction | `corpus_provenance`, `ocr_and_cleaning_pipeline`, `chunking_strategy`, `fig_corpus_distribution`, `fig_chunk_size_distribution` |
| System design | `architecture_overview`, `chat_request_path`, `request_sequence`, `context_assembly`, `local_first_deployment`, `deployment_topology` |
| Implementation | `app_shell_anatomy`, `conversation_persistence`, `data_model`, `answer_provenance_chain`, screenshots |
| Safety design | `safety_gate_ladder` |
| Evaluation method | `evaluation_framework`, `gold_question_lifecycle`, `llm_judge_rubric` |
| Results — retrieval | `fig_embedding_comparison`, `fig_embedding_space`, `fig_retrieval_mode` |
| Results — generation | `fig_generator_scores`, `fig_hallucination_rate`, `fig_latency_breakdown` |
| Results — safety | `fig_safety_confusion`, `fig_gate_attribution`, `screenshot_refusal`, `screenshot_low_confidence` |
| Results — test-set validity | `fig_human_eval` |

**The README takes six of these, not all 39**: the four README-set diagrams,
plus `screenshot_welcome` and `screenshot_answer`. A README is a landing page —
results belong in its existing tables, and a reader who wants the full argument
follows it to the report.

---

## README set

### 1. `architecture_overview.png`

> Design a diagram explaining the overall architecture of a fully-local
> Khmer-language RAG chatbot for Cambodian insurance law.
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
> user's machine with no external API calls. Alongside the answer it keeps a
> local conversation store, links each cited document to its source PDF, and can
> show a per-answer diagnostics readout.
>
> Three relationships must be visible: Stage A feeds Stage B (the index and test
> set are what get evaluated); Stage B's decisions determine Stage C's
> configuration; and there is a feedback loop where the human evaluation in
> Stage B corrects the test question set that Stage B itself depends on.
>
> Leave clear empty label space next to every element — exact filenames will be
> added afterward.

---

### 2. `chat_request_path.png`

> Design a diagram showing what happens when a user asks the chatbot one
> question.
>
> The central idea: this system refuses far more often than a normal chatbot,
> deliberately, because giving wrong legal or financial advice is worse than
> giving none. Refusal is a designed feature, not a failure. Two checkpoints
> along the path can stop a question, and the second is far more expensive than
> the first — that ordering is intentional and should be legible.
>
> There are exactly TWO refusal points. Do not invent a third that inspects or
> blocks the finished answer: no such check exists. Once generation starts, the
> answer reaches the user. What the system does after generating is *qualify*
> the answer — a low-confidence banner when retrieval was weak, and a permanent
> disclaimer — never withhold it.
>
> The path:
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
> 7. A local LLM generates an answer from those passages only, streamed token by
>    token, and the user can interrupt it mid-stream.
> 8. The answer is delivered with three things attached: the list of source
>    documents it drew on, each linking to that document's original PDF; a row of
>    controls to copy or react to it; and a link to a full diagnostics readout
>    for that specific answer.
>
> Show the three refusal branches leaving the main path, drawn as deliberately as
> the success path. Show that the cost of stopping increases at each checkpoint.

---

### 3. `hybrid_retrieval_rrf.png`

> Design a diagram explaining a hybrid document retrieval method.
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

### 4. `evaluation_framework.png`

> Design a diagram explaining how a RAG chatbot was evaluated before shipping.
>
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

## Report set

### 5. `corpus_provenance.png`

> Design a diagram showing how a document collection was built.
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
> Note the asymmetry between document count and passage count: guidance has 16
> documents but the fewest passages, while laws has only 4 documents and far
> more passages. Document count is not a proxy for corpus weight, and the
> diagram should make that legible.
>
> Convey that the chunking respects the documents' own legal structure — this is
> not generic text splitting.

---

### 6. `chunking_strategy.png`

> Design a diagram contrasting two ways of splitting a legal document into
> searchable passages.
>
> Cover the same source document processed two ways.
>
> **The naive way**, drawn as the rejected option: cut the text every N
> characters. The cuts land wherever they land — mid-sentence, mid-article,
> splitting a provision's number away from the provision's text. A passage
> retrieved this way can be a fragment that no longer states which article it
> came from, which makes it useless as a citation and misleading as evidence.
>
> **The chosen way**: cut along the document's own structure. Cambodian legal
> documents are already divided into numbered provisions — "មាត្រា" (article)
> for laws and sub-decrees, "ប្រការ" for prakas — and guidance documents are
> divided by heading hierarchy. The splitter follows those boundaries, so one
> passage is one provision, carrying its own number and heading with it.
>
> Show the consequence, which is the point of the figure: a passage retrieved
> from the structure-aware split can be cited precisely ("Article 10 of the Law
> on Insurance"), because the boundary and the citation are the same thing. A
> passage from the naive split cannot.
>
> Also convey that passage sizes are therefore uneven by nature — provisions
> differ in length, ranging from very short to very long — and that this
> unevenness is accepted as a consequence of respecting structure, not a defect
> to be normalized away.

---

### 7. `khmer_retrieval_problem.png`

> Design a diagram explaining why searching Khmer text by keyword is harder than
> searching English, and how this system solves it. This figure is about a
> language problem, not a software problem.
>
> Three facts have to land, in order:
>
> **1. Khmer is written without spaces between words.** An English sentence
> arrives pre-split into words, so a keyword index can be built by cutting on
> spaces. A Khmer sentence is a continuous run of characters with no such
> markers, so the same approach yields one enormous meaningless token. Contrast
> a Khmer string with an English one: the English separates cleanly into words,
> the Khmer does not.
>
> **2. The workaround is to index overlapping character fragments.** Instead of
> words, the text is cut into every sliding run of three consecutive characters.
> A question and a document now match when they share many fragments. This
> works, but it is noisy: fragments are short, common ones appear everywhere,
> and the signal from a rare meaningful term gets buried under fragments shared
> by every document in the collection.
>
> **3. Provision numbers need rescuing from that noise.** When someone asks
> about a specific article by number, the fragment-based match ranks it poorly —
> the query is short, and its fragments are mostly generic. The system detects
> the "article + number" pattern explicitly, treats the pair as a single unit,
> and weights it heavily so the one passage that actually contains that
> provision rises to the top. Show this as a targeted rescue of a signal that
> the general method loses: a passage sitting far down a ranked list, lifted to
> the top by the boost.
>
> Convey that steps 2 and 3 exist because of step 1 — this is a chain of
> consequences from a property of the writing system, not three unrelated
> techniques.

---

### 8. `safety_gate_ladder.png`

> Design a diagram showing a chatbot's refusal system as a ladder of
> increasingly expensive checks.
>
> The organising idea: every question that will ultimately be refused should be
> refused as early and as cheaply as possible, so the checks are ordered by what
> they cost to run. There are two such checks, not three.
>
> **Rung 1 — free.** No AI, no search, pure pattern matching on the raw text.
> It catches four things: text that contains no Khmer script at all; questions
> too short to be meaningful; and questions whose *intent* is out of bounds
> regardless of topic — asking which company or product to buy, asking what
> something will cost, or asking for advice about the user's own situation.
> Those last three are refused even though they are genuinely about insurance,
> because the system is not licensed to answer them. That distinction —
> out-of-scope is not the same as off-topic — should be visible.
>
> **Rung 2 — costs one retrieval.** The question is embedded and searched. If
> even the best-matching passage in the entire corpus is too weak a match, the
> question is off-topic and is refused here, before any text is generated.
>
> **After the rungs — generation.** If a question clears both rungs it is
> answered, and the answer is shown. There is no third rung: nothing inspects or
> blocks a finished answer. Where retrieval was weak but not weak enough to
> refuse, the answer is still shown, carrying a visible low-confidence warning.
> Show that qualifying an answer and withholding one are different things.
>
> Then show how outcomes are scored, because this is how the system was
> measured: every adversarial test question has an expected behaviour, so each
> result falls into one of four boxes — correctly refused, correctly answered,
> wrongly answered, wrongly refused. These are the four cells of a confusion
> matrix: one axis is what the question *should* have got (refuse or answer),
> the other is what it actually got (refused or answered), and each cell is the
> intersection of the two. Do not lay it out as an "expected" column beside an
> "actual" column — the outcome names are the cells, not the headings.
>
> Convey that the two error cells are not equally bad: a wrong answer to a
> pricing or advice question is a real harm, while a needlessly refused question
> is only an annoyance.
>
> Leave empty space for the counts in each box and for the threshold values.

---

### 9. `context_assembly.png`

> Design a diagram showing how retrieved passages are turned into a prompt for a
> language model, and why the funnel narrows.
>
> The stages narrow deliberately:
>
> 1. **Retrieve wide.** The fused search returns ten candidate passages ranked
>    best-first. Retrieving more than will be used is intentional — it gives the
>    selection step something to choose between.
> 2. **Cap per document.** No more than two passages from any single document
>    may proceed. Show why this rule exists: a single law split into per-article
>    passages produces many near-identical neighbours, and without a cap they
>    would crowd out every other document, leaving the model with one source's
>    view of the question instead of several. Also show the opposite failure the
>    cap must avoid — set the cap to one and two questions about adjacent
>    articles of the same law collapse to the same answer, because only one of
>    the two articles ever reaches the model.
> 3. **Select five.** The surviving top five become the context.
> 4. **Assemble.** Those five are concatenated with their document names and
>    provision headings, wrapped in an instruction telling the model to answer
>    only from the supplied text, and sent to a local model with a fixed context
>    budget that the assembled prompt must fit inside.
>
> Show the discarded passages leaving at each stage, and show that the passages
> reaching the model carry their source identity with them — that identity is
> what the citation under the answer is later built from.

---

### 10. `local_first_deployment.png`

> Design a diagram showing which parts of an AI system run on the user's own
> machine and which do not. There is a boundary, and everything belongs on one
> side of it or the other.
>
> **Inside, running locally at question time:** the embedding model that turns
> the question into a vector; the vector index holding every passage; the
> keyword index; the fusion step; the quantized large language model generating
> the answer, served by a local inference server bound to the loopback address
> only; the conversation database; and the original source PDFs, served from
> disk.
>
> **Outside, and used exactly once:** a single external OCR API call, during the
> one-time corpus build, to read scanned Khmer pages that no local tool could
> read reliably. It is crossed once during setup and never again.
>
> The point of the figure is the asymmetry: no user question, no retrieved
> passage, and no generated answer ever leaves the machine. For a system
> handling legal questions about people's own insurance situations, that is a
> privacy property, not a performance detail, and it is the main thing the
> figure exists to communicate.
>
> Also show that the inference server listens only on the loopback interface,
> so it is not reachable from the local network.

---

### 11. `app_shell_anatomy.png`

> Design an annotated layout diagram of a chat application's interface,
> showing each region and what it is for. Do not render readable text inside the
> regions; leave callout space instead.
>
> The window is divided into two columns.
>
> **Left column, a fixed-width rail.** Top to bottom: a brand lockup (emblem,
> product name, subtitle) sharing its row with a control that collapses the
> rail; a "new conversation" action; a section heading; then a list of past
> conversations, one row each, with the current one marked as active. Hovering a
> row reveals an overflow control at its right edge that opens a two-item menu
> (rename, delete). The rail is pinned — it does not scroll away
> with the conversation.
>
> **Right column, the conversation.** The user's questions sit in bubbles on one
> side; the assistant's answers run as plain text at full width on the other.
> Under each answer, in order: a short list of the source documents that answer
> drew on, as links; then a row of four icon controls — copy, approve,
> disapprove, and details.
>
> **Pinned to the bottom of the right column, not the window:** the question
> input, centred on the conversation column rather than on the whole window, and
> beneath it a permanent single-line legal disclaimer.
>
> Two structural points the annotation should make: the conversation column
> starts at the same height as the rail's brand, with no header bar above it;
> and the input and disclaimer stay fixed while the conversation scrolls behind
> them.
>
> This figure documents a real interface, so it must not show controls the
> interface does not have. In particular there is **no** pin or favourite
> control on a conversation row, **no** file attachment or upload in the input,
> and **no** avatars beside messages. "New conversation" is a plain row in the
> list, not a large filled button. The four controls under an answer are
> icon-only with tooltips, not labelled buttons. The source list is document
> *names* as text links, not icons.

---

### 12. `conversation_persistence.png`

> Design a diagram explaining how a chat application keeps conversations after
> the browser is closed.
>
> The framing question: a web app's in-memory state dies when the page reloads,
> so what makes a conversation still be there tomorrow?
>
> Show two layers and the traffic between them.
>
> **The working layer**, in memory, belonging to one browser session: the list
> of messages currently on screen and which conversation is open.
>
> **The durable layer**, a small local database file on disk: two tables, one
> row per conversation and one row per message, with each message stored as a
> complete record — the question, the answer, the sources, and the full
> diagnostic payload for that answer.
>
> The important behaviours to convey:
>
> - **Write-through, not save-on-exit.** Every finished message is written to
>   disk the moment it lands, through a single choke point that all outcomes
>   pass through — normal answer, interrupted answer, error, refusal. Show that
>   there is exactly one door to the database, so no path can quietly skip it.
>   The consequence: a crash loses at most the answer still being typed.
> - **A conversation is created when the question is accepted**, not when its
>   answer arrives — so the sidebar shows it as the active conversation for the
>   whole time the answer is being generated.
> - **Reading back is complete, not partial.** Reopening a conversation restores
>   not just the visible text but the diagnostic payload too, so the per-answer
>   detail view still works on a conversation from last week.
> - Show that the database file is local and per-machine — it is not
>   synchronised, shared, or committed to version control.

---

### 13. `answer_provenance_chain.png`

> Design a diagram showing how a reader can trace a generated answer back to the
> law it came from. The figure
> argues that a claim in this system is checkable at three widening levels of
> detail.
>
> Start with a generated answer on screen. From it, three paths lead outward:
>
> **Level 1 — which documents.** Directly under the answer is the list of source
> documents it drew on, named in full. This is the everyday level: a reader sees
> at a glance that the answer rests on, say, three specific laws.
>
> **Level 2 — the source itself.** Each of those names is a link that opens that
> document's original scanned PDF. The reader is not asked to trust a
> paraphrase; they can open the actual instrument. Show this as the jump from
> the interface to a real document.
>
> **Level 3 — how it was retrieved.** A separate control opens a full diagnostic
> readout for that one answer: every candidate passage the search returned, the
> score each retrieval method gave it, which method surfaced it, the full text
> of each passage, token counts, and stage-by-stage timings. This view has its
> own address, so it can be reloaded, bookmarked, or handed to someone else for
> review. That address is a local one carrying the message's identifier — leave
> space for it rather than inventing a public URL; this system is not hosted.
>
> The point to convey: the three levels serve different readers — a user, a
> lawyer checking the citation, and an examiner auditing the retrieval — and the
> same answer supports all three without the user having to ask for anything in
> advance.

---

---

## Extra report figures

These exist because a 100-page report has long stretches — corpus building,
evaluation method, implementation — that the sets above leave without a single
figure.

### 14. `ocr_and_cleaning_pipeline.png`

> Design a diagram showing how a scanned legal document becomes clean machine
> text. This figure is about
> what goes *wrong* at each stage and what fixes it.
>
> Six stages, in order, each with a defect it removes:
>
> 1. **The starting material is bad.** Scanned Cambodian legal PDFs: photographed
>    or faxed pages, skewed, speckled, sometimes stamped or handwritten-annotated.
> 2. **Page rendering.** Each PDF page becomes an image, one file per page.
> 3. **Optical character recognition.** A vision-capable model reads the Khmer
>    script off each page image. Show that this is the one stage that leaves the
>    machine, and that it happens once during the build rather than at question
>    time.
> 4. **Page-level cleaning.** Removes what belongs to the *page* rather than the
>    *text*: running headers, footers, page numbers, and stray marks. These would
>    otherwise appear mid-sentence once pages are joined.
> 5. **Stitching.** Cleaned pages are joined back into whole documents, repairing
>    sentences and provisions that were split across a page break — a provision
>    broken by pagination must come back together before it can be chunked as
>    one unit.
> 6. **Document-level cleaning.** Normalisation that only makes sense once the
>    whole document is visible: expanding abbreviations, unifying spacing, and
>    reconciling numbering.
>
> The idea to convey: the cleaning happens at two different scopes for a reason
> — some defects are only visible per page, others only once the document is
> whole — and doing them in the wrong order would either miss defects or destroy
> real content. Leave empty label space for the script filenames and the folder
> each stage writes to.

---

### 15. `gold_question_lifecycle.png`

> Design a diagram following one test question through its life, from written to
> used as evidence.
>
> The argument: an evaluation is only as trustworthy as the question set it
> measures against, so the question set itself is treated as something to be
> validated rather than assumed.
>
> The life of one question:
>
> 1. **Authored**, paired with two things: which passages *should* be retrieved
>    to answer it, and what the correct answer *is*. Show that the set has two
>    origins — a smaller human-written portion and a larger model-drafted
>    portion — and that the distinction is tracked, because the human-written
>    subset is what the generation comparison is run on.
> 2. **Rated by people.** Multiple evaluators score it on clarity, relevance,
>    difficulty, answer quality, and uniqueness. Some questions are rated by more
>    than one person; that overlapping subset is the only part where agreement
>    between evaluators can be measured, and it is smaller than the whole set.
> 3. **Corrected.** Where evaluators disagreed with the recorded target answer,
>    the target is fixed. Show this correction flowing *back into* the question
>    set.
> 4. **Flagged or kept.** Questions with problems are marked for curation rather
>    than silently dropped.
> 5. **Used**, finally, as the shared measuring stick for the retrieval and
>    generation comparisons.
>
> The point of the figure is the loop at step 3: the yardstick is adjusted before
> it measures anything. Leave empty space for the counts.

---

### 16. `llm_judge_rubric.png`

> Design a diagram explaining how generated answers are scored by a language
> model acting as a judge.
>
> One answer goes in. The judge sees three things: the original question, the
> passages the system retrieved, and the answer produced from them. It never
> sees which system produced the answer — the comparison is blind.
>
> It returns **five quality scores**, each measuring something different:
> whether the retrieved passages were relevant to the question; whether the
> answer is faithful to those passages; whether the answer is factually correct;
> whether it actually addresses what was asked; and whether it is complete.
> These five are combined into one weighted score.
>
> Separately, it returns a **hallucination flag** — a binary judgment of whether
> the answer asserts something the passages do not support.
>
> The single most important thing this diagram must convey: **the hallucination
> flag is deliberately kept out of the combined score.** It stays on its own
> track and never merges into the total. The reason is that averaging it away
> would let a model with a high fabrication rate hide behind otherwise fluent,
> well-structured answers — and in a legal-information setting a fabricated
> provision is a different category of failure from a merely incomplete answer,
> not a smaller amount of the same failure.
>
> Leave empty space for the five dimension names, their weights, and the score
> range.

---

### 17. `request_sequence.png`

> Design a figure showing a single question being answered by a locally-hosted
> RAG chatbot as an interaction over time between the participants below.
>
> Five participants: the **user**, the **web interface**, the **retrieval
> layer**, the **local inference server**, and the **local conversation store**.
>
> The sequence:
>
> 1. User submits a Khmer question to the interface.
> 2. The interface immediately records that a new conversation has begun in the
>    store — before any answer exists — and shows the question on screen.
> 3. The interface asks the retrieval layer for relevant passages. Show the two
>    parallel searches and their fusion happening *within* that participant.
> 4. The retrieval layer returns ranked passages, and the interface checks the
>    best score against a threshold before continuing.
> 5. The interface sends an assembled prompt to the inference server and receives
>    the answer **as a stream of tokens**, not a single reply, redrawing as each
>    one arrives.
> 6. Show that the user can interrupt this stream, and that an interrupted answer
>    is still recorded rather than discarded.
> 7. When the stream ends, the interface writes the finished message to the store
>    and renders the sources and controls beneath it.
>
> Two things worth making visible: the store is written to twice, at the
> beginning and the end; and every participant except the user is a local
> process, so nothing in the figure leaves the machine.

---

### 18. `data_model.png`

> Design an entity-relationship diagram for the small local database behind a
> chat application's history.
>
> Two entities:
>
> - A **conversation**: an identifier, a human-readable title derived from its
>   opening question, when it was created, and when it was last used. The
>   last-used time is what orders the sidebar, and renaming deliberately does
>   *not* update it — annotate that, since it is a behavioural decision encoded
>   in the schema.
> - A **message**: an identifier, the conversation it belongs to, its position in
>   that conversation, and a single stored payload.
>
> One conversation holds many messages; deleting a conversation deletes its
> messages with it.
>
> Then expand the payload, because it is the interesting part: it is not just
> the question and answer text. It carries the sources cited, the full ranked
> retrieval result with each passage's scores, the timings of every pipeline
> stage, and the flags describing how the answer ended — normal, low confidence,
> interrupted, refused, or errored. Show that storing the whole record, rather
> than just what is displayed, is what allows the per-answer diagnostics view to
> still work on a conversation reopened weeks later.
>
> Leave empty space for the table and column names.

---

### 19. `scope_boundary.png`

> Design a diagram that classifies questions a legal-information assistant might
> receive, according to whether it is willing to answer them. This figure is
> about a policy decision, not a technical mechanism.
>
> The classification has three regions rather than two, and the third is the
> entire point:
>
> 1. **Off-topic.** Nothing to do with insurance law. Declined because the system
>    has no relevant material.
> 2. **On-topic and answerable.** Questions about what the regulations say — what
>    a provision requires, what a term means, what a procedure is, what a deadline
>    is. These are answered, with citations.
> 3. **On-topic but not permitted.** Questions squarely about insurance that the
>    system still declines: which company or product someone should choose, what
>    a policy will cost them, what they personally should do about their own
>    situation, and any calculation of an individual's money. These sit inside
>    the subject area but outside what the system will answer.
>
> The reason for region 3 must be legible: these are the questions where being
> confidently wrong causes real financial harm to a specific person, and where
> answering would amount to unlicensed advice. The system explains what it cannot
> do and redirects to a licensed agent or the regulator.
>
> Convey that region 3 is a deliberate refusal, not a limitation — the system
> could produce fluent text for these questions and chooses not to.

---

### 20. `deployment_topology.png`

> Design a deployment diagram showing the processes that make up a locally-run AI
> application and how they talk to each other.
>
> On one machine, two long-running processes:
>
> - A **web application server** hosting the user interface, which the user
>   reaches in a browser. It also serves the corpus PDFs as static files.
> - An **inference server** holding the quantized language model in memory,
>   bound to the loopback interface only, reachable through an
>   OpenAI-compatible HTTP interface.
>
> The web application talks to the inference server over local HTTP. The browser
> talks only to the web application.
>
> Alongside them, on-disk resources the web application reads: the vector index,
> the passage metadata, the embedding model weights, the source PDFs, and the
> conversation database.
>
> Two properties to make visible: the inference server binds to loopback rather
> than all interfaces, so nothing else on the network can reach an
> unauthenticated inference endpoint; and the launcher checks whether the
> inference port is already in use before starting, so a stale server cannot be
> left serving a different model than the one the app believes is loaded.
>
> Leave empty label space for process names and port numbers.

## Screenshots (capture these)

These belong in both the README and the report, and must be **real captures of
the running app**, not generated images. A generated mockup of a UI that does
not exist is a misrepresentation of the artifact, and in a report it will not
survive a reader comparing it to the code.

Run the app, then capture at a consistent window size (roughly 1440×900,
retina if available):

| File | What to capture |
|---|---|
| `screenshot_welcome.png` | The welcome state: rail on the left, greeting, and the centred input |
| `screenshot_answer.png` | A completed Khmer answer with its source list and action row |
| `screenshot_sidebar_menu.png` | The conversation overflow menu open, showing rename and delete |
| `screenshot_rename_inline.png` | A conversation row mid-rename, with the inline field focused |
| `screenshot_low_confidence.png` | An answer carrying the low-confidence warning |
| `screenshot_refusal.png` | A refused question — use one from `data/10_adversarial/` |
| `screenshot_diagnostics.png` | The diagnostics page for one answer, showing the ranked chunk cards |
| `screenshot_source_pdf.png` | A cited document's PDF opened from an answer's source list |

For the report, the refusal and low-confidence captures matter most — they are
the evidence that the safety design is real and not just described.

---

## Results figures (plot these, do not generate them)

Every figure below is a chart of measured values, so it must be **drawn from
the report JSON**, not produced by an image model. Ask the agent to write a
plotting script rather than to generate a picture.

| File | Chart | Source |
|---|---|---|
| `fig_embedding_comparison.png` | Grouped bars: Hit@5, Recall@5, MRR@5, File-Hit@5 for the three embedding models | `reports/embedding_model_comparison/embedding_model_comparison.json` |
| `fig_retrieval_mode.png` | Hit@k curves, dense vs. hybrid, k = 1…10, per category | `reports/retrieval_mode_comparison/retrieval_mode_comparison.json` |
| `fig_generator_scores.png` | Grouped bars: the five judged dimensions for the four models | `reports/generator_model_comparison/generator_model_comparison.json` |
| `fig_hallucination_rate.png` | Bars: hallucination rate per model, kept **separate** from the quality score | same file |
| `fig_safety_confusion.png` | 2×2 confusion matrix, plus a per-category breakdown | `reports/safety_gate_evaluation/safety_gate_evaluation.json` |
| `fig_gate_attribution.png` | Which gate stopped each refused question | same file |
| `fig_human_eval.png` | Mean score per dimension, with inter-evaluator agreement shown alongside | `reports/human_golden_question_evaluation/human_golden_question_evaluation_report.json` |
| `fig_corpus_distribution.png` | Documents vs. chunks per category, on twin axes | `reports/data_card/data_card.json` |
| `fig_chunk_size_distribution.png` | Chunk-length histogram per category, showing the long tail | `reports/data_card/data_card.json` |
| `fig_latency_breakdown.png` | Stacked stage timings for one answer: embed, search, fuse, build, first token, decode | a captured diagnostics payload from `chat_history.db` |
| `fig_embedding_space.png` | 2-D projection of the **actual** chunk embeddings, one panel per candidate model | recompute from `vector_db/<model>/` |

Instruction for the agent:

> Write `scripts/render_report_figures.py`. It reads the JSON files under
> `reports/` and plots each figure listed in the results table of
> `docs/image_generation_prompts.md` with matplotlib, writing PNGs to `assets/`.
>
> Every number must come from the JSON. Do not hardcode a single value, and do
> not fill in a missing metric with a plausible one — if a field is absent,
> fail loudly and say which file and key were missing.
>
> Keep it deterministic and re-runnable, so the figures can be regenerated after
> any evaluation is re-run. Keep one consistent style across all figures, label
> every axis, and state units or score ranges.
> Hallucination rate must never be merged into the combined quality score — it
> is reported separately on purpose.

Two things to be careful about when these are read in a report:

- **MiniLM's near-zero retrieval scores are a real finding, not a plotting
  bug.** It cannot discriminate between Khmer legal passages and scores almost
  everything alike, so its ranking is close to random. Annotate that on the
  chart, or a reader will assume the bar is broken.
- **Inter-evaluator agreement is computed over the subset of questions that
  more than one person rated**, not over all 400. State the subset size on the
  figure, or the agreement number will be read as stronger evidence than it is.
- **`fig_embedding_space.png` is the figure that explains the headline retrieval
  result**, so it is worth the effort: project the real stored chunk vectors
  down to two dimensions with the same method and seed for each model, and show
  and present the three panels together. The winning model should show structure
  while the weakest one collapses into a single undifferentiated cloud — which
  is what "cannot discriminate between Khmer legal passages" looks like. Say
  which projection and seed were used in the caption, and never reuse one
  model's projection for another.
- **`fig_latency_breakdown.png` must state its hardware.** A timing chart
  without the machine it was measured on is not reproducible, and a reader will
  reasonably assume the fastest plausible hardware.

---

## Wrapper instruction

Wrap the diagram prompts with this when driving generation from a CLI agent:

> Generate the diagrams for `assets/` using the OpenAI image API, one per
> diagram prompt in `docs/image_generation_prompts.md` — the README set and the
> report set only. Do **not** generate the screenshots or the results figures;
> those are captured and plotted respectively, and the file explains why.
>
> Then, in a second pass, write `scripts/render_diagram_labels.py` that
> composites the exact text labels onto each generated image using PIL.
>
> All label text must be read from the repository, never typed from memory:
> script filenames from `scripts/`, thresholds and constants from `app.py`,
> corpus and chunk counts from `reports/data_card/data_card.json`, model names
> from `install_models.py`, and result figures from the JSON files under
> `reports/`. Do not let the image model render filenames, paths, or numbers —
> it will corrupt them.
>
> Keep the composited label styling consistent across every figure, and keep the
> script deterministic and re-runnable so the diagrams can be regenerated when
> the pipeline changes.
>
> Show me each image after the text pass before committing anything.
