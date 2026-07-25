"""Hit Rate@k for dense and hybrid retrieval, by document category."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _style as S

S.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT = Path("assets/research-paper")
SOURCE = Path("reports/retrieval_mode_comparison/retrieval_mode_comparison.json")

data = json.loads(SOURCE.read_text())
ks = data["eval_ks"]
rows = {
    mode: {row[0]: row[1:] for row in data[mode]["hit"]}
    for mode in ("dense", "hybrid")
}
categories = [row[0] for row in data["dense"]["hit"]]
display = {
    "guidance": "Guidance",
    "laws": "Laws",
    "prakas": "Prakas",
    "sub_decrees": "Sub-decrees",
}
question_counts = {
    row[0]: row[1] for row in data["dense"]["summary"]
}

assert set(categories) == set(rows["hybrid"])
assert all(len(rows[mode][category]) == len(ks) for mode in rows for category in categories)

global_min = min(
    value
    for mode in ("dense", "hybrid")
    for category in categories
    for value in rows[mode][category]
)
global_max = max(
    value
    for mode in ("dense", "hybrid")
    for category in categories
    for value in rows[mode][category]
)
y_floor = global_min - 0.035
y_ceiling = global_max + 0.012
assert y_floor < 0.72 <= global_min

fig = S.figure(
    "Hybrid retrieval improves hit rate in every document category",
    "Dense FAISS versus dense-plus-BM25 retrieval fused with Reciprocal Rank Fusion.",
)
grid = fig.add_gridspec(
    2,
    2,
    left=0.085,
    right=0.975,
    bottom=0.135,
    top=0.775,
    hspace=0.47,
    wspace=0.20,
)

axes = []
for index, (letter, category) in enumerate(zip("ABCD", categories)):
    ax = fig.add_subplot(
        grid[index // 2, index % 2],
        sharex=axes[0] if axes else None,
        sharey=axes[0] if axes else None,
    )
    axes.append(ax)
    S.axes(ax, grid="both")
    S.panel_heading(
        ax,
        letter,
        display[category],
        f"n = {question_counts[category]} gold questions",
        y=1.075,
    )

    dense = rows["dense"][category]
    hybrid = rows["hybrid"][category]
    ax.fill_between(
        ks,
        dense,
        hybrid,
        color=S.TEAL,
        alpha=0.075,
        linewidth=0,
        zorder=1,
    )

    # Hybrid is drawn first.  Dense then uses a dashed line and larger open
    # circles, so coincident values still show a filled diamond inside a ring.
    ax.plot(
        ks,
        hybrid,
        color=S.TEAL,
        lw=2.0,
        marker="D",
        markersize=3.8,
        markeredgecolor=S.PAPER,
        markeredgewidth=0.55,
        zorder=3,
    )
    ax.plot(
        ks,
        dense,
        color=S.VERMILION,
        lw=1.55,
        linestyle=(0, (4.0, 2.6)),
        marker="o",
        markersize=6.0,
        markerfacecolor="none",
        markeredgecolor=S.VERMILION,
        markeredgewidth=1.15,
        zorder=4,
    )

    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(y_floor, y_ceiling)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_yticks([0.72, 0.80, 0.90, 1.00])
    ax.tick_params(labelbottom=index >= 2, labelleft=index % 2 == 0)

    gain_at_5 = hybrid[4] - dense[4]
    ax.text(
        0.975,
        0.075,
        f"at k = 5   dense {dense[4]:.4f}   hybrid {hybrid[4]:.4f}   gain {gain_at_5:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=S.SZ_NOTE,
        fontfamily="sans-serif",
        color=S.INK_SOFT,
        bbox={
            "boxstyle": "round,pad=0.32,rounding_size=0.08",
            "facecolor": S.PAPER,
            "edgecolor": S.RULE,
            "linewidth": 0.6,
        },
        zorder=6,
    )

fig.text(
    0.026,
    0.47,
    "Hit Rate@k",
    rotation=90,
    ha="center",
    va="center",
    fontsize=S.SZ_AXIS,
    color=S.INK_SOFT,
)
fig.text(
    0.53,
    0.073,
    "Retrieved cutoff, k",
    ha="center",
    va="center",
    fontsize=S.SZ_AXIS,
    color=S.INK_SOFT,
)
fig.legend(
    handles=[
        Line2D(
            [],
            [],
            color=S.VERMILION,
            lw=1.55,
            linestyle=(0, (4.0, 2.6)),
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.15,
        ),
        Line2D(
            [],
            [],
            color=S.TEAL,
            lw=2.0,
            marker="D",
            markersize=4,
            markeredgecolor=S.PAPER,
        ),
    ],
    labels=["Dense (FAISS)", "Hybrid (RRF)"],
    loc="center right",
    bbox_to_anchor=(0.972, 0.892),
    ncol=2,
    columnspacing=2.0,
    handlelength=2.5,
    labelcolor=S.INK_SOFT,
)
fig.text(
    0.972,
    0.028,
    f"Shared y-scale; floor {y_floor:.3f}, below the observed minimum {global_min:.2f}. "
    "Open circles preserve dense points where the two modes coincide.",
    ha="right",
    va="bottom",
    fontsize=S.SZ_NOTE,
    color=S.INK_FAINT,
)

path = S.save(fig, OUT / "chart 02 - fig_retrieval_mode.png")
print(
    json.dumps(
        {
            "eval_ks": ks,
            "y_limits": [y_floor, y_ceiling],
            "dense": {category: rows["dense"][category] for category in categories},
            "hybrid": {category: rows["hybrid"][category] for category in categories},
        },
        ensure_ascii=False,
    )
)
print("wrote", path)
