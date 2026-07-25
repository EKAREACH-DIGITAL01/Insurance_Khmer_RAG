"""fig_embedding_score_spread -- why BGE-M3 ranks and MiniLM cannot.

Replaces the originally specced `fig_embedding_space` (a 2-D projection of chunk
embeddings). That figure would NOT have shown the effect it was meant to explain:
measured over chunk-to-chunk pairs, MiniLM's spread (p95-p5 = 0.226) is
essentially BGE-M3's (0.235), so a chunk-only projection shows no collapse.

The compression is in the QUERY->chunk direction. For each gold question this
plots the similarity of the top-10 retrieved chunks against rank.

Read this figure with care: compression alone does NOT explain ranking failure.
e5-large compresses as hard as MiniLM (mean top-10 spread 0.020 vs 0.021) and
still reaches Hit@5 = 0.663 against MiniLM's 0.011. What separates them is
whether the ordering inside that narrow band carries signal, not how wide the
band is. Hit@5 is printed on each panel so the figure cannot be read as
"flatter => worse".

Encodes the gold questions with each model (e5 gets its required "query: "
prefix) and searches that model's own FAISS index. Raw numbers are written
alongside the PNG so the figure is auditable.
"""
import os
import json
from pathlib import Path
os.environ["HF_HOME"] = str(Path("models/retriever/.cache").resolve())

import faiss, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

OUT = Path("assets/research-paper")
HUE, INK, INK_MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
TOP_K = 10
# Hit Rate @5 from reports/embedding_model_comparison/ -- printed on each panel so
# score compression is never mistaken for the cause of ranking failure.
HIT5 = dict(zip(
    json.loads(Path("reports/embedding_model_comparison/embedding_model_comparison.json")
               .read_text())["comparison"]["headers"][1:],
    json.loads(Path("reports/embedding_model_comparison/embedding_model_comparison.json")
               .read_text())["comparison"]["rows"][0][1:]))

# query_prefix mirrors scripts/08's model config -- e5 was trained with it and
# omitting it would handicap e5 unfairly.
MODELS = [
    ("BAAI/bge-m3",                            "BAAI-BGE-M3",                           ""),
    ("intfloat/multilingual-e5-large",         "multilingual-e5-large",                 "query: "),
    ("paraphrase-multilingual-MiniLM-L12-v2",  "MiniLM-paraphrase-multilingual-L12-v2", ""),
]

questions = []
for gf in sorted(Path("data/09_evaluation").rglob("gold_questions.json")):
    questions += [i["question"] for i in json.loads(gf.read_text())]
print(f"{len(questions)} gold questions")

results = {}
for label, vdir, prefix in MODELS:
    ix = faiss.read_index(f"vector_db/{vdir}/khmer_insurance.index")
    model = SentenceTransformer(label)
    Q = model.encode([prefix + q for q in questions], normalize_embeddings=True,
                     batch_size=16, show_progress_bar=False)
    S, _ = ix.search(np.asarray(Q, dtype="float32"), TOP_K)
    results[label] = S
    print(f"  {label:42} rank1 mean={S[:,0].mean():.4f}  "
          f"top-{TOP_K} spread mean={(S[:,0]-S[:,-1]).mean():.4f}")
    del model

fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharey=True)
ranks = np.arange(1, TOP_K + 1)
for ax, (label, vdir, _) in zip(axes, MODELS):
    S = results[label]
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for row in S:                            # one thin line per question
        ax.plot(ranks, row, color=HUE, lw=0.5, alpha=0.10, zorder=2)
    ax.plot(ranks, S.mean(axis=0), color="#eb6834", lw=2.2, zorder=4,
            marker="o", ms=4.5, mew=0)
    spread = (S[:, 0] - S[:, -1]).mean()
    ax.set_title(label.split("/")[-1], fontsize=9.5, color=INK, pad=7)
    ax.annotate(f"mean top-1  {S[:,0].mean():.3f}\n"
                f"mean spread  {spread:.3f}\n"
                f"Hit@5  {HIT5[label]}",
                xy=(0.96, 0.055), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.5, color=INK_MUTED, linespacing=1.5)
    ax.set_xlabel(f"Retrieved rank (1-{TOP_K})", fontsize=9, color=INK_MUTED)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

axes[0].set_ylabel("Cosine similarity to query", fontsize=9, color=INK_MUTED)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=HUE, lw=1.4, alpha=0.5),
            Line2D([], [], color="#eb6834", lw=2.2, marker="o", ms=4.5, mew=0)],
           [f"One of {len(questions)} gold questions", "Mean across all questions"],
           loc="lower center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.06), labelcolor=INK)
fig.suptitle("Query-to-chunk similarity compresses as the embedding model shrinks",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 01 - fig_embedding_score_spread.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
(OUT / "chart 01 - fig_embedding_score_spread.json").write_text(json.dumps(
    {"top_k": TOP_K, "n_questions": len(questions),
     "models": {l: {"mean_rank1": float(s[:,0].mean()),
                    "mean_spread_top1_to_topk": float((s[:,0]-s[:,-1]).mean()),
                    "mean_by_rank": [float(x) for x in s.mean(axis=0)]}
                for l, s in results.items()}}, indent=1))
print("wrote", p)
