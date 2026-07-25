"""fig_chunk_size_distribution -- chunk-length histogram per category, long tail.

Lengths are measured from the actual chunk files in
data/08_merged_chunked_documents/, then cross-checked against the avg/min/max
already recorded in reports/data_card/data_card.json. If the two disagree the
script fails rather than plotting a number that contradicts the data card.
"""
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("data/08_merged_chunked_documents")
CARD = json.loads(Path("reports/data_card/data_card.json").read_text())["knowledge_base"]
OUT = Path("assets/research-paper")

HUE, INK, INK_MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
MEDIAN = "#eb6834"
label = {"guidance": "Guidance", "laws": "Laws",
         "prakas": "Prakas", "sub_decrees": "Sub-decrees"}

lengths = {}
for f in sorted(SRC.glob("*_chunks.json")):
    cat = f.name.replace("_chunks.json", "")
    lengths[cat] = [len(c["content"]) for c in json.loads(f.read_text())]

# --- verify against the data card before drawing anything ---
for cat, xs in lengths.items():
    ref = CARD["by_category"][cat]
    assert len(xs) == ref["chunks"], f"{cat}: {len(xs)} chunks vs card {ref['chunks']}"
    assert min(xs) == ref["min_chunk_size"], f"{cat}: min {min(xs)} vs {ref['min_chunk_size']}"
    assert max(xs) == ref["max_chunk_size"], f"{cat}: max {max(xs)} vs {ref['max_chunk_size']}"
    assert abs(sum(xs)/len(xs) - ref["avg_chunk_size"]) < 0.05, f"{cat}: avg mismatch"
total = sum(len(v) for v in lengths.values())
assert total == CARD["total_chunks"], f"{total} vs {CARD['total_chunks']}"
print(f"verified against data_card.json: {total} chunks, per-category min/max/avg all match")

order = ["guidance", "laws", "prakas", "sub_decrees"]
XMAX = 3000  # tail beyond this is annotated, not clipped silently
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), sharey=True)
for ax, cat in zip(axes, order):
    xs = lengths[cat]
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    # Percentage of this category's chunks, not raw counts: n ranges 146..783,
    # so counts on a shared axis would flatten the three smaller categories into
    # unreadable slivers. Percentages make the SHAPES comparable, which is the
    # point of the figure.
    w = [100.0 / len(xs)] * len(xs)
    ax.hist([min(x, XMAX) for x in xs], bins=range(0, XMAX + 100, 100),
            weights=w, color=HUE, zorder=3, rwidth=0.86)
    med = statistics.median(xs)
    ax.axvline(med, color=MEDIAN, lw=1.6, ls=(0, (4, 2)), zorder=4)
    over = sum(1 for x in xs if x > XMAX)
    ax.set_title(f"{label[cat]}  (n = {len(xs)})", fontsize=10, color=INK, pad=7)
    ax.set_xlabel("Chunk length (characters)", fontsize=8.5, color=INK_MUTED)
    ax.annotate(f"median {med:.0f}\nmax {max(xs)}"
                + (f"\n{over} chunk{'s' if over != 1 else ''} > {XMAX}" if over else ""),
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=INK_MUTED, linespacing=1.45)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    ax.set_xticks([0, 1000, 2000, 3000])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

axes[0].set_ylabel("Share of category's chunks (%)", fontsize=9, color=INK_MUTED)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=MEDIAN, lw=1.6, ls=(0, (4, 2)))],
           ["Median chunk length"],
           loc="lower center", frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), labelcolor=INK)
fig.suptitle("Chunk-length distribution by document category "
             f"({CARD['total_chunks']} chunks, mean {CARD['avg_chunk_size']} characters)",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 03 - fig_chunk_size_distribution.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", p)
