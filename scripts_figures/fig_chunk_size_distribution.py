"""Chunk-length distributions with a visible, retained long tail.

Each category uses the same fixed 100-character bins and reports percentage
within category, making shapes comparable despite n ranging from 146 to 783.
The x-axis is explicitly compressed after 1,200 characters.  No observation is
winsorised, moved, or discarded.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _style as S

S.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SOURCE = Path("data/08_merged_chunked_documents")
CARD_SOURCE = Path("reports/data_card/data_card.json")
OUT = Path("assets/research-paper")

card = json.loads(CARD_SOURCE.read_text())["knowledge_base"]
display = {
    "guidance": "Guidance",
    "laws": "Laws",
    "prakas": "Prakas",
    "sub_decrees": "Sub-decrees",
}
order = ["guidance", "laws", "prakas", "sub_decrees"]

lengths: dict[str, list[int]] = {}
for source_file in sorted(SOURCE.glob("*_chunks.json")):
    category = source_file.name.replace("_chunks.json", "")
    lengths[category] = [
        len(chunk["content"]) for chunk in json.loads(source_file.read_text())
    ]

# Fail before drawing if the source files disagree with the recorded data card.
for category, values in lengths.items():
    reference = card["by_category"][category]
    assert len(values) == reference["chunks"]
    assert min(values) == reference["min_chunk_size"]
    assert max(values) == reference["max_chunk_size"]
    assert abs(sum(values) / len(values) - reference["avg_chunk_size"]) < 0.05
total = sum(len(values) for values in lengths.values())
assert total == card["total_chunks"]

BIN_WIDTH = 100
BREAK_AT = 1200
TAIL_LIMIT = 4900
bin_edges = np.arange(0, TAIL_LIMIT + BIN_WIDTH, BIN_WIDTH)

counts: dict[str, np.ndarray] = {}
percent: dict[str, np.ndarray] = {}
for category in order:
    category_counts, _ = np.histogram(lengths[category], bins=bin_edges)
    assert int(category_counts.sum()) == len(lengths[category])
    counts[category] = category_counts
    percent[category] = category_counts.astype(float) * (100.0 / len(lengths[category]))

max_percent = max(float(values.max()) for values in percent.values())
y_ceiling = max_percent * 1.22

fig = S.figure(
    "Chunking strategy shapes the corpus length profile",
    "Heading-based guidance is broader; legal-unit chunks concentrate at shorter lengths.",
)
outer = fig.add_gridspec(
    2,
    2,
    left=0.085,
    right=0.975,
    bottom=0.16,
    top=0.76,
    hspace=0.55,
    wspace=0.23,
)

all_main_axes = []
for index, (letter, category) in enumerate(zip("ABCD", order)):
    inner = outer[index // 2, index % 2].subgridspec(
        1, 2, width_ratios=[3.6, 1.0], wspace=0.045
    )
    ax = fig.add_subplot(
        inner[0],
        sharey=all_main_axes[0] if all_main_axes else None,
    )
    tail = fig.add_subplot(inner[1], sharey=ax)
    all_main_axes.append(ax)

    S.axes(ax, grid="y")
    S.axes(tail, grid="y")
    tail.spines["left"].set_visible(False)
    tail.tick_params(axis="y", left=False, labelleft=False)
    S.panel_heading(
        ax,
        letter,
        display[category],
        (
            f"n = {len(lengths[category])} · "
            + ("heading-based" if category == "guidance" else "legal-unit")
        ),
        y=1.06,
    )

    colour = S.VERMILION if category == "guidance" else S.BLUE
    values = percent[category]
    left_edges = bin_edges[:-1]
    widths = np.diff(bin_edges)
    main_mask = left_edges < BREAK_AT
    tail_mask = left_edges >= BREAK_AT

    ax.bar(
        left_edges[main_mask],
        values[main_mask],
        width=widths[main_mask] * 0.88,
        align="edge",
        color=colour,
        alpha=0.90,
        edgecolor=S.PAPER,
        linewidth=0.45,
        zorder=3,
    )
    tail.bar(
        left_edges[tail_mask],
        values[tail_mask],
        width=widths[tail_mask] * 0.88,
        align="edge",
        color=colour,
        alpha=0.90,
        edgecolor=S.PAPER,
        linewidth=0.35,
        zorder=3,
    )

    median = statistics.median(lengths[category])
    ax.axvline(
        median,
        ymax=0.91,
        color=S.INK,
        lw=1.15,
        linestyle=(0, (4, 2.6)),
        zorder=5,
    )
    ax.text(
        median,
        y_ceiling * 0.96,
        f"  median {median:g}",
        ha="left",
        va="top",
        fontsize=S.SZ_NOTE,
        color=S.INK_SOFT,
    )

    ax.set_xlim(0, BREAK_AT)
    tail.set_xlim(BREAK_AT, TAIL_LIMIT)
    ax.set_ylim(0, y_ceiling)
    ax.set_xticks([0, 400, 800])
    tail.set_xticks([3000, 4500])
    ax.tick_params(labelleft=index % 2 == 0)

    # Diagonal marks make the change in horizontal scale impossible to miss.
    mark = 0.018
    kwargs = dict(color=S.INK_FAINT, clip_on=False, lw=0.9)
    ax.plot((1 - mark, 1 + mark), (-mark, +mark), transform=ax.transAxes, **kwargs)
    tail.plot((-mark, +mark), (-mark, +mark), transform=tail.transAxes, **kwargs)

    over_3000 = sum(value > 3000 for value in lengths[category])
    tail.text(
        0.96,
        0.92,
        f"max {max(lengths[category]):,}\n> 3,000: {over_3000}",
        transform=tail.transAxes,
        ha="right",
        va="top",
        fontsize=S.SZ_NOTE,
        fontfamily="sans-serif",
        linespacing=1.4,
        color=S.INK_SOFT,
        bbox={
            "boxstyle": "round,pad=0.3,rounding_size=0.08",
            "facecolor": S.PAPER,
            "edgecolor": S.RULE,
            "linewidth": 0.6,
        },
        zorder=6,
    )

fig.text(
    0.026,
    0.45,
    "Share of category (%)",
    rotation=90,
    ha="center",
    va="center",
    fontsize=S.SZ_AXIS,
    color=S.INK_SOFT,
)
fig.text(
    0.53,
    0.105,
    "Chunk length (characters)",
    ha="center",
    va="center",
    fontsize=S.SZ_AXIS,
    color=S.INK_SOFT,
)
fig.legend(
    handles=[
        Patch(facecolor=S.VERMILION, edgecolor="none", label="Heading-based"),
        Patch(facecolor=S.BLUE, edgecolor="none", label="Legal-unit"),
        Line2D(
            [],
            [],
            color=S.INK,
            lw=1.15,
            linestyle=(0, (4, 2.6)),
            label="Median",
        ),
    ],
    loc="center right",
    bbox_to_anchor=(0.972, 0.892),
    ncol=3,
    columnspacing=1.8,
    handlelength=1.9,
    labelcolor=S.INK_SOFT,
)
fig.text(
    0.5,
    0.043,
    f"{total:,} chunks; corpus mean {card['avg_chunk_size']} characters. "
    f"Fixed {BIN_WIDTH}-character bins and a shared percentage scale make categories comparable.\n"
    f"The horizontal scale is compressed after {BREAK_AT:,} characters; "
    f"all observations through the {max(max(values) for values in lengths.values()):,}-character maximum are retained.",
    ha="center",
    va="center",
    fontsize=S.SZ_NOTE,
    color=S.INK_FAINT,
    linespacing=1.5,
)

path = S.save(fig, OUT / "chart 03 - fig_chunk_size_distribution.png")
print(
    json.dumps(
        {
            "raw_chunk_lengths": lengths,
            "bin_edges": bin_edges.tolist(),
            "bin_counts": {category: counts[category].tolist() for category in order},
            "bin_percent_of_category": {
                category: percent[category].tolist() for category in order
            },
            "y_limits": [0, y_ceiling],
            "axis_break_after_characters": BREAK_AT,
        },
        ensure_ascii=False,
    )
)
print("wrote", path)
