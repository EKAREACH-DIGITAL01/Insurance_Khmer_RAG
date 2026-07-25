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
import json, sqlite3, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _style as S
S.use()
import matplotlib.pyplot as plt

OUT = Path("assets/research-paper")
DB = "chat_history.db"

rows = []
with sqlite3.connect(DB) as c:
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
            ptokens=d["prompt_tokens"], embed=d["embed_time"], bm25=d["bm25_time"],
            # FAISS search, RRF fusion and context building are sub-millisecond to
            # a few ms each; kept as one segment because three slivers that size
            # are indistinguishable from the bar edge.
            rest=d["search_time"] + d["fusion_time"] + d["context_time"],
            cache_hit=prefill < 1.0,
        ))

with sqlite3.connect(DB) as c:
    model = Path(json.loads(
        c.execute("select payload from messages").fetchone()[0])["diag"]["model"]).name


def sysctl(k):
    return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()


chip, cores = sysctl("machdep.cpu.brand_string"), sysctl("hw.ncpu")
ram = int(sysctl("hw.memsize")) // 1073741824

labels = [f"Answer {i+1}" + ("\n(prompt-cache hit)" if r["cache_hit"] else "")
          for i, r in enumerate(rows)]
y = list(range(len(rows)))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.6, 3.4),
                               gridspec_kw={"width_ratios": [1.6, 1]})

# ---- left: total wall clock -------------------------------------------------
LSTACK = [("Retrieval", "retr", S.SLATE), ("Prefill (to first token)", "prefill", S.CORAL),
          ("Decode", "decode", S.TEAL), ("Other", "other", S.OCHRE)]
S.panel(axl, letter="A", title="Total time per answer", grid="x")
left = [0.0] * len(rows)
tmax = max(r["total"] for r in rows)
for name, key, colour in LSTACK:
    vals = [r[key] for r in rows]
    axl.barh(y, vals, left=left, color=colour, height=0.58, label=name,
             zorder=3, edgecolor="white", linewidth=1.1)
    for i, v in enumerate(vals):
        if v / tmax > 0.06:
            axl.annotate(f"{v:.1f}", (left[i] + v / 2, i), ha="center", va="center",
                         fontsize=S.SZ_ANNOT, color="white", zorder=5)
    left = [a + b for a, b in zip(left, vals)]
for i, r in enumerate(rows):
    axl.annotate(f"   {r['total']:.1f}s total  ·  {r['ptokens']:,} prompt tok",
                 (r["total"], i), va="center", fontsize=S.SZ_ANNOT, color=S.INK_SOFT)
axl.set_xlim(0, tmax * 1.44)
axl.set_xlabel(S.smallcaps("Wall clock (seconds)"), fontsize=S.SZ_AXIS_L)
axl.legend(loc="lower left", bbox_to_anchor=(0, 1.14), ncol=4, columnspacing=1.5,
           handlelength=1.1, handletextpad=0.5, labelcolor=S.INK_SOFT,
           fontsize=S.SZ_ANNOT)

# ---- right: retrieval only --------------------------------------------------
RSTACK = [("Embed query", "embed", S.SLATE), ("BM25", "bm25", S.CORAL),
          ("FAISS + RRF + context", "rest", S.TEAL)]
S.panel(axr, letter="B", title="Retrieval slice, own scale", grid="x")
left = [0.0] * len(rows)
for name, key, colour in RSTACK:
    vals = [r[key] for r in rows]
    axr.barh(y, vals, left=left, color=colour, height=0.58, label=name,
             zorder=3, edgecolor="white", linewidth=1.1)
    left = [a + b for a, b in zip(left, vals)]
for i, r in enumerate(rows):
    axr.annotate(f"   {r['retr']:.2f}s", (r["retr"], i), va="center",
                 fontsize=S.SZ_ANNOT, color=S.INK_SOFT)
axr.set_xlim(0, max(r["retr"] for r in rows) * 1.40)
axr.set_xlabel(S.smallcaps("Wall clock (seconds)"), fontsize=S.SZ_AXIS_L)
axr.legend(loc="lower left", bbox_to_anchor=(0, 1.14), ncol=3, columnspacing=1.5,
           handlelength=1.1, handletextpad=0.5, labelcolor=S.INK_SOFT,
           fontsize=S.SZ_ANNOT)

for ax in (axl, axr):
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=S.SZ_TICK, color=S.INK)
    ax.invert_yaxis()

S.suptitle(fig, "Prompt processing, not retrieval, dominates answer latency", y=1.17)
tiny = max(r["rest"] for r in rows) * 1000
S.footnote(fig,
           f"Embedding the query is nearly all of the retrieval slice: FAISS search, RRF "
           f"fusion and context building together never exceed {tiny:.0f} ms.\n"
           f"{chip}, {cores} cores, {ram} GB RAM  ·  macOS  ·  embedding on CPU  ·  "
           f"generator {model} via llama.cpp  ·  n = {len(rows)} answers",
           y=-0.17)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "chart 04 - fig_latency_breakdown.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
print("wrote", p)
for i, r in enumerate(rows):
    print(f"  answer {i+1}: total={r['total']:.2f}s retrieval={r['retr']:.2f}s "
          f"prefill={r['prefill']:.2f}s decode={r['decode']:.2f}s"
          + ("  [prompt-cache hit]" if r["cache_hit"] else ""))
