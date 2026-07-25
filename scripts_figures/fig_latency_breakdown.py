"""fig_latency_breakdown -- where a single answer's wall-clock time actually goes.

Read from the real diagnostics payloads persisted in chat_history.db by the live
app; nothing is synthesised. Hardware is printed on the figure, because a timing
chart without its machine is unreadable.

Two panels, because the stages differ by three orders of magnitude and a single
stacked bar would render FAISS search (0.1 ms) as nothing:
  left  -- total wall clock per answer: retrieval, prefill, decode, other
  right -- the retrieval slice on its own scale

One answer is a prompt-cache HIT: llama.cpp had the identical prompt in its KV
cache, so prefill cost 0.23 s instead of 34-58 s. It is labelled as such rather
than dropped -- it is the honest illustration of what prefill costs, but it is
NOT a typical cold answer and must never be averaged in silently.
"""
import json, sqlite3, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("assets/research-paper")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# dataviz slots 1-4 (validated; slots 3-4 WARN on contrast -> segments carry
# visible value labels, which satisfies the relief rule)
C_RETR, C_PREFILL, C_DECODE, C_OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_EMBED, C_BM25, C_FAISS, C_FUSE, C_CTX = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"

rows = []
with sqlite3.connect("chat_history.db") as c:
    for (p,) in c.execute("select payload from messages order by created_at"):
        d = json.loads(p)["diag"]
        prefill = d["first_token_time"]
        retr = (d["embed_time"] + d["search_time"] + d["bm25_time"]
                + d["fusion_time"] + d["context_time"])
        decode = d["generation_time"] - prefill
        rows.append(dict(
            retr=retr, prefill=prefill, decode=decode,
            other=d["total_time"] - (retr + prefill + decode),
            total=d["total_time"], tokens=d["generated_tokens"],
            ptokens=d["prompt_tokens"],
            embed=d["embed_time"], faiss=d["search_time"], bm25=d["bm25_time"],
            fuse=d["fusion_time"], ctx=d["context_time"],
            cache_hit=prefill < 1.0,
        ))

model = Path(json.loads(next(iter(
    sqlite3.connect("chat_history.db").execute("select payload from messages")))[0]
)["diag"]["model"]).name
chip = subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],
                      capture_output=True, text=True).stdout.strip()
ram = int(subprocess.run(["sysctl","-n","hw.memsize"],
                         capture_output=True, text=True).stdout) // 1073741824
cores = subprocess.run(["sysctl","-n","hw.ncpu"],
                       capture_output=True, text=True).stdout.strip()

labels = [f"Answer {i+1}" + ("\n(prompt-cache hit)" if r["cache_hit"] else "")
          for i, r in enumerate(rows)]
y = range(len(rows))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 3.5),
                               gridspec_kw={"width_ratios": [1.55, 1]})

# ---- left: total wall clock -------------------------------------------------
stack = [("Retrieval", "retr", C_RETR), ("Prefill (to first token)", "prefill", C_PREFILL),
         ("Decode", "decode", C_DECODE), ("Other", "other", C_OTHER)]
left = [0.0] * len(rows)
for name, key, colour in stack:
    vals = [r[key] for r in rows]
    axl.barh(list(y), vals, left=left, color=colour, height=0.62,
             label=name, zorder=3, edgecolor="white", linewidth=1.4)
    for i, v in enumerate(vals):
        if v / max(r["total"] for r in rows) > 0.055:      # only where it fits
            axl.annotate(f"{v:.1f}s", (left[i] + v / 2, i), ha="center", va="center",
                         fontsize=8, color="white", zorder=5, fontweight="bold")
    left = [a + b for a, b in zip(left, vals)]
for i, r in enumerate(rows):
    axl.annotate(f"  {r['total']:.1f}s total · {r['ptokens']:,} prompt tok · "
                 f"{r['tokens']} generated",
                 (r["total"], i), va="center", fontsize=8, color=INK_MUTED)
axl.set_xlim(0, max(r["total"] for r in rows) * 1.46)
axl.set_xlabel("Wall clock (seconds)", fontsize=9, color=INK_MUTED)
axl.set_title("Total time per answer", fontsize=10, color=INK, pad=10)
axl.legend(loc="lower left", bbox_to_anchor=(0, 1.17), frameon=False,
           fontsize=8.2, labelcolor=INK, ncol=4, columnspacing=1.4,
           handlelength=1.2, handletextpad=0.5)

# ---- right: retrieval only --------------------------------------------------
rstack = [("Embed query", "embed", C_EMBED), ("BM25", "bm25", C_BM25),
          ("FAISS search", "faiss", C_FAISS), ("RRF fuse", "fuse", C_FUSE),
          ("Build context", "ctx", C_CTX)]
left = [0.0] * len(rows)
for name, key, colour in rstack:
    vals = [r[key] for r in rows]
    axr.barh(list(y), vals, left=left, color=colour, height=0.62,
             label=name, zorder=3, edgecolor="white", linewidth=1.4)
    left = [a + b for a, b in zip(left, vals)]
for i, r in enumerate(rows):
    axr.annotate(f"  {r['retr']:.2f}s", (r["retr"], i), va="center",
                 fontsize=8, color=INK_MUTED)
axr.set_xlim(0, max(r["retr"] for r in rows) * 1.42)
axr.set_xlabel("Wall clock (seconds)", fontsize=9, color=INK_MUTED)
axr.set_title("Retrieval slice, on its own scale", fontsize=10, color=INK, pad=10)
axr.legend(loc="lower left", bbox_to_anchor=(0, 1.17), frameon=False,
           fontsize=8.2, labelcolor=INK, ncol=3, columnspacing=1.2,
           handlelength=1.2, handletextpad=0.5)
_tiny = max(r["faiss"] + r["fuse"] + r["ctx"] for r in rows)
_note = (f"Embedding the query is nearly all of the retrieval slice: FAISS search, "
         f"RRF fusion and context building together never exceed {_tiny*1000:.0f} ms.")

for ax in (axl, axr):
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

fig.suptitle("Prompt processing, not retrieval, dominates answer latency",
             fontsize=11.5, color=INK, y=1.13)
fig.text(0.5, -0.10, _note + "\n" +
         f"{chip}, {cores} cores, {ram} GB RAM · macOS · embedding on CPU · "
         f"generator {model} via llama.cpp · n = {len(rows)} answers",
         ha="center", fontsize=8, color=INK_MUTED)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 04 - fig_latency_breakdown.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", p)
for i, r in enumerate(rows):
    print(f"  answer {i+1}: total={r['total']:.2f}s retrieval={r['retr']:.2f}s "
          f"prefill={r['prefill']:.2f}s decode={r['decode']:.2f}s"
          + ("  [prompt-cache hit]" if r["cache_hit"] else ""))
