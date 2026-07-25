"""fig_chunk_size_distribution -- chunk-length histogram per category, long tail.

Lengths are measured from the actual chunk files in
data/08_merged_chunked_documents/, then cross-checked against the avg/min/max
already recorded in reports/data_card/data_card.json. If the two disagree the
script fails rather than plotting a number that contradicts the data card.
"""
import json, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import _style as S
S.use()
import matplotlib.pyplot as plt

SRC = Path("data/08_merged_chunked_documents")
CARD = json.loads(Path("reports/data_card/data_card.json").read_text())["knowledge_base"]
OUT = Path("assets/research-paper")

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
PANEL_HUE = [S.TEAL, S.CORAL, S.SLATE, S.OCHRE]
XMAX = 3000  # tail beyond this is annotated, not clipped silently

fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.2), sharey=True)
for ax, letter, hue, cat in zip(axes, "ABCD", PANEL_HUE, order):
    xs = lengths[cat]
    S.panel(ax, letter=letter, title=f"{label[cat]}  (n = {len(xs)})")
    # Percentage of this category's chunks, not raw counts: n ranges 146..783,
    # so counts on a shared axis would flatten the three smaller categories into
    # unreadable slivers. Percentages make the SHAPES comparable, which is the
    # point of the figure.
    w = [100.0 / len(xs)] * len(xs)
    ax.hist([min(x, XMAX) for x in xs], bins=range(0, XMAX + 100, 100),
            weights=w, color=hue, zorder=3, rwidth=0.82)
    med = statistics.median(xs)
    ax.axvline(med, color=S.REFLINE, lw=1.1, ls=(0, (3.5, 2.5)), zorder=4)
    over = sum(1 for x in xs if x > XMAX)
    ax.annotate(f"median {med:.0f}\nmax {max(xs)}"
                + (f"\n{over} > {XMAX}" if over else ""),
                xy=(0.96, 0.93), xycoords="axes fraction", ha="right", va="top",
                fontsize=S.SZ_ANNOT, color=S.INK_SOFT, linespacing=1.6)
    ax.set_xlabel(S.smallcaps("Chunk length (chars)"), fontsize=S.SZ_AXIS_L)
    ax.set_xticks([0, 1000, 2000, 3000])

axes[0].set_ylabel(S.smallcaps("Share of category (%)"), fontsize=S.SZ_AXIS_L)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=S.REFLINE, lw=1.1, ls=(0, (3.5, 2.5)))],
           ["Median chunk length"], loc="lower center",
           bbox_to_anchor=(0.5, -0.07), labelcolor=S.INK_SOFT, handlelength=2.2)
S.suptitle(fig, f"Chunk-length distribution by document category", y=1.04)
S.footnote(fig, f"{CARD['total_chunks']} chunks, mean {CARD['avg_chunk_size']} characters. "
              f"Guidance uses heading-based chunking; the other three use legal-unit chunking.",
           y=-0.15)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 03 - fig_chunk_size_distribution.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", p)
