"""fig_retrieval_mode -- Hit@k curves, dense vs hybrid, k = 1..10, per category.

Plotted from reports/retrieval_mode_comparison/retrieval_mode_comparison.json.
Nothing here is invented: every point is read from that file.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("assets/research-paper")
SRC = Path("reports/retrieval_mode_comparison/retrieval_mode_comparison.json")

# dataviz categorical slots 1-2 (validated: adjacent CVD dE 24.7, normal 33.6)
DENSE, HYBRID = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

d = json.loads(SRC.read_text())
ks = d["eval_ks"]
rows = {m: {r[0]: r[1:] for r in d[m]["hit"]} for m in ("dense", "hybrid")}
cats = [r[0] for r in d["dense"]["hit"]]
label = {"guidance": "Guidance", "laws": "Laws",
         "prakas": "Prakas", "sub_decrees": "Sub-decrees"}

fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1), sharey=True)
for ax, cat in zip(axes, cats):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for mode, colour in (("dense", DENSE), ("hybrid", HYBRID)):
        ax.plot(ks, rows[mode][cat], color=colour, lw=2,
                marker="o", ms=4.5, mew=0, zorder=3,
                label="Dense (FAISS)" if mode == "dense" else "Hybrid (RRF)")
    ax.set_title(label[cat], fontsize=10, color=INK, pad=7)
    ax.set_xlabel("k", fontsize=9, color=INK_MUTED)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    # Floor sits below the global minimum (dense sub-decrees @1 = 0.72) so no
    # point is ever clipped -- a truncated axis that hides a data point is worse
    # than a tall axis.
    ax.set_ylim(0.68, 1.01)
    ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

axes[0].set_ylabel("Hit Rate @k", fontsize=9, color=INK_MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.03), labelcolor=INK)
fig.suptitle("Hybrid retrieval (FAISS + BM25, RRF) vs. dense-only, by document category",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 02 - fig_retrieval_mode.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", p)
