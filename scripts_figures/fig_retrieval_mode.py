"""fig_retrieval_mode -- Hit@k curves, dense vs hybrid, k = 1..10, per category.

Plotted from reports/retrieval_mode_comparison/retrieval_mode_comparison.json.
Nothing here is invented: every point is read from that file.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _style as S
S.use()
import matplotlib.pyplot as plt

OUT = Path("assets/research-paper")
SRC = Path("reports/retrieval_mode_comparison/retrieval_mode_comparison.json")

DENSE, HYBRID = S.CORAL, S.TEAL

d = json.loads(SRC.read_text())
ks = d["eval_ks"]
rows = {m: {r[0]: r[1:] for r in d[m]["hit"]} for m in ("dense", "hybrid")}
cats = [r[0] for r in d["dense"]["hit"]]
label = {"guidance": "Guidance", "laws": "Laws",
         "prakas": "Prakas", "sub_decrees": "Sub-decrees"}

lo = min(min(v) for m in rows for v in rows[m].values())

fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.3), sharey=True)
for ax, letter, cat in zip(axes, "ABCD", cats):
    S.panel(ax, letter=letter, title=label[cat])
    for mode, colour, name in (("dense", DENSE, "Dense (FAISS)"),
                               ("hybrid", HYBRID, "Hybrid (RRF)")):
        ax.plot(ks, rows[mode][cat], color=colour, lw=1.6, marker="o", ms=3.6,
                mew=0, zorder=3, label=name, clip_on=False)
    ax.set_xlabel("$k$", fontsize=S.SZ_AXIS_L + 1.5)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    # Floor sits below the global minimum (dense sub-decrees @1 = 0.72) so no
    # point is ever clipped -- a truncated axis that hides a data point is worse
    # than a tall axis.
    ax.set_ylim(lo - 0.04, 1.008)

axes[0].set_ylabel(S.smallcaps("Hit rate") + "  @$k$", fontsize=S.SZ_AXIS_L)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05),
           labelcolor=S.INK_SOFT, handlelength=1.8, columnspacing=2.4)
S.suptitle(fig, "Hybrid retrieval versus dense-only, by document category")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 02 - fig_retrieval_mode.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
print("wrote", p)
print(f"  global minimum plotted = {lo:.4f} (y floor {lo - 0.04:.4f})")
for m in ("dense", "hybrid"):
    print(f"  {m:7} @5 " + "  ".join(f"{c}={rows[m][c][4]:.4f}" for c in cats))
