"""Query-to-chunk score spread for three embedding models.

The stored ten-point mean curves are read from the research-paper JSON.  The
400 per-question curves are independently recomputed by encoding every gold
question and searching each model's own FAISS index.  E5 keeps its required
``query: `` prefix.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["HF_HOME"] = str(Path("models/retriever/.cache").resolve())
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np

# faiss MUST be imported before sentence_transformers (i.e. before torch). Both
# link an OpenMP runtime, and on macOS arm64 loading torch's first makes faiss
# search segfault mid-run -- it crashed here right after the first model's 400
# queries encoded. Capping faiss to one search thread is not sufficient on its
# own; the import order is what fixes it. Order-sensitive: do not sort.
import faiss

# Single search thread on top of the import order: deterministic for these flat
# indexes and leaves every score unchanged.
faiss.omp_set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))
import _style as S

S.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT = Path("assets/research-paper")
SOURCE = OUT / "chart 01 - fig_embedding_score_spread.json"
HIT_SOURCE = Path(
    "reports/embedding_model_comparison/embedding_model_comparison.json"
)
TOP_K = 10

# query_prefix mirrors the retriever model configuration.  Omitting the prefix
# would handicap e5 and change the measurement.
MODELS = [
    ("BAAI/bge-m3", "BAAI-BGE-M3", "", "BGE-M3", S.TEAL),
    (
        "intfloat/multilingual-e5-large",
        "multilingual-e5-large",
        "query: ",
        "Multilingual E5-large",
        S.VERMILION,
    ),
    (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-paraphrase-multilingual-L12-v2",
        "",
        "Multilingual MiniLM-L12",
        S.BLUE,
    ),
]


stored = json.loads(SOURCE.read_text())
assert stored["top_k"] == TOP_K

comparison = json.loads(HIT_SOURCE.read_text())["comparison"]
hit_row = next(row for row in comparison["rows"] if row[0] == "Hit Rate @5")
hit5 = dict(zip(comparison["headers"][1:], hit_row[1:]))

questions: list[str] = []
for gold_file in sorted(Path("data/09_evaluation").rglob("gold_questions.json")):
    questions.extend(item["question"] for item in json.loads(gold_file.read_text()))
assert len(questions) == stored["n_questions"] == 400

def encode_in_child(model_id: str, prefix: str) -> np.ndarray:
    """Runs the encode in a separate interpreter -- see _encode_queries.py for
    why torch must never be imported into this process."""
    with tempfile.TemporaryDirectory() as tmp:
        q_path, e_path = Path(tmp) / "q.json", Path(tmp) / "e.npy"
        q_path.write_text(json.dumps(questions, ensure_ascii=False))
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "_encode_queries.py"),
             model_id, prefix, str(q_path), str(e_path)],
            check=True, stdout=subprocess.DEVNULL,
        )
        return np.load(e_path)


scores_by_model: dict[str, np.ndarray] = {}
for model_id, vector_dir, prefix, _, _ in MODELS:
    index = faiss.read_index(f"vector_db/{vector_dir}/khmer_insurance.index")
    embeddings = encode_in_child(model_id, prefix)
    scores, _ = index.search(embeddings, TOP_K)
    assert scores.shape == (len(questions), TOP_K)

    # The JSON is the auditable mean series; recomputation supplies the 400
    # background trajectories and verifies that the saved summary is current.
    stored_mean = np.asarray(stored["models"][model_id]["mean_by_rank"], dtype=float)
    np.testing.assert_allclose(
        scores.mean(axis=0), stored_mean, rtol=0, atol=5e-6,
        err_msg=f"{model_id}: recomputed means differ from {SOURCE}",
    )
    scores_by_model[model_id] = scores
    del index
    gc.collect()


all_values = np.concatenate([scores.ravel() for scores in scores_by_model.values()])
stored_values = np.concatenate(
    [
        np.asarray(stored["models"][model_id]["mean_by_rank"], dtype=float)
        for model_id, *_ in MODELS
    ]
)
data_min = float(min(all_values.min(), stored_values.min()))
data_max = float(max(all_values.max(), stored_values.max()))
y_pad = max(0.012, (data_max - data_min) * 0.035)
y_limits = (data_min - y_pad, data_max + y_pad)

fig = S.figure(
    "Similarity compression does not determine ranking success",
    "Query-to-chunk cosine similarity at retrieved ranks 1–10; all panels share one scale.",
)
grid = fig.add_gridspec(
    2,
    3,
    left=0.07,
    right=0.975,
    bottom=0.115,
    top=0.775,
    height_ratios=[4.25, 0.92],
    hspace=0.40,
    wspace=0.12,
)

ranks = np.arange(1, TOP_K + 1)
main_axes = []
for col, (letter, spec) in enumerate(zip("ABC", MODELS)):
    model_id, _, _, display_name, colour = spec
    ax = fig.add_subplot(grid[0, col], sharey=main_axes[0] if main_axes else None)
    main_axes.append(ax)
    S.axes(ax, grid="y")
    S.panel_heading(ax, letter, display_name)

    scores = scores_by_model[model_id]
    for row in scores:
        ax.plot(
            ranks,
            row,
            color=colour,
            lw=0.48,
            alpha=0.045,
            zorder=1,
        )

    mean_values = np.asarray(stored["models"][model_id]["mean_by_rank"], dtype=float)
    ax.plot(
        ranks,
        mean_values,
        color=colour,
        lw=2.35,
        marker="o",
        markersize=4.2,
        markeredgecolor=S.PAPER,
        markeredgewidth=0.7,
        zorder=4,
    )
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(*y_limits)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_xlabel("Retrieved rank", fontsize=S.SZ_AXIS, labelpad=6)
    # Every panel keeps its own y tick labels. They share one scale, but the
    # panels sit far enough apart that reading a value in C off A's axis is
    # genuinely hard.
    ax.tick_params(labelleft=True)

    # Metrics live in their own band, never over the trajectories.
    card = fig.add_subplot(grid[1, col])
    card.set_xlim(0, 1)
    card.set_ylim(0, 1)
    card.axis("off")
    card.add_patch(
        plt.Rectangle(
            (0, 0.05),
            1,
            0.86,
            facecolor=S.WASH,
            edgecolor=S.RULE,
            linewidth=0.7,
        )
    )
    card.plot([0, 1], [0.91, 0.91], color=colour, lw=2.1, clip_on=False)

    metrics = [
        ("MEAN TOP-1", stored["models"][model_id]["mean_rank1"], ".3f"),
        (
            "MEAN SPREAD",
            stored["models"][model_id]["mean_spread_top1_to_topk"],
            ".3f",
        ),
        ("HIT@5", hit5[model_id], None),
    ]
    for x, (metric_label, value, fmt) in zip((0.17, 0.50, 0.83), metrics):
        rendered = format(value, fmt) if fmt else str(value)
        card.text(
            x,
            0.61,
            metric_label,
            ha="center",
            va="center",
            fontsize=S.SZ_METRIC_LABEL,
            fontfamily="sans-serif",
            fontweight="bold",
            color=S.INK_FAINT,
        )
        card.text(
            x,
            0.30,
            rendered,
            ha="center",
            va="center",
            fontsize=S.SZ_METRIC_VALUE,
            fontweight="bold",
            color=colour if metric_label == "HIT@5" else S.INK,
        )
    card.plot([0.335, 0.335], [0.20, 0.70], color=S.RULE, lw=0.6)
    card.plot([0.665, 0.665], [0.20, 0.70], color=S.RULE, lw=0.6)

main_axes[0].set_ylabel("Cosine similarity to query", fontsize=S.SZ_AXIS, labelpad=8)

fig.legend(
    handles=[
        Line2D([], [], color=S.BLUE, lw=0.8, alpha=0.28),
        Line2D(
            [],
            [],
            color=S.BLUE,
            lw=2.35,
            marker="o",
            markersize=4.2,
            markeredgecolor=S.PAPER,
        ),
    ],
    labels=[
        f"One of {len(questions)} gold questions",
        "Stored mean across all questions",
    ],
    loc="lower left",
    bbox_to_anchor=(0.067, 0.018),
    ncol=2,
    columnspacing=2.2,
    handlelength=2.5,
    labelcolor=S.INK_SOFT,
)
fig.text(
    0.974,
    0.037,
    "Spread is mean rank-1 score minus mean rank-10 score.",
    ha="right",
    va="center",
    fontsize=S.SZ_NOTE,
    color=S.INK_FAINT,
)

path = S.save(fig, OUT / "chart 01 - fig_embedding_score_spread.png")

# Print every plotted trajectory and every stored mean value without display
# rounding, so the complete figure can be audited from stdout.
plotted = {
    "ranks": ranks.tolist(),
    "y_limits": list(y_limits),
    "models": {
        model_id: {
            "question_scores": scores_by_model[model_id].tolist(),
            "mean_by_rank": stored["models"][model_id]["mean_by_rank"],
            "mean_rank1": stored["models"][model_id]["mean_rank1"],
            "mean_spread_top1_to_topk": stored["models"][model_id][
                "mean_spread_top1_to_topk"
            ],
            "hit_at_5": hit5[model_id],
        }
        for model_id, *_ in MODELS
    },
}
print(json.dumps(plotted, ensure_ascii=False))
print("wrote", path)
