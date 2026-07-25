"""Wall-clock answer latency and retrieval sub-stages from chat_history.db.

Every value is read from the diagnostics payloads the live app persisted; nothing
is synthesised or averaged. Answer 4 was a prompt-cache hit -- llama.cpp still
held the identical prompt in its KV cache, so its prefill cost 0.23 s instead of
34-58 s. It is kept and labelled, never blended into a cold-answer figure.

Two panels, because the stages span five orders of magnitude: RRF fusion runs in
~0.02 ms against a ~50 s prefill. Panel A is seconds; panel B is milliseconds on
a log axis, as grouped bars rather than scattered markers.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _style as S

S.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

OUT = Path("assets/research-paper")
DB = Path("chat_history.db")

rows = []
diagnostics = []
with sqlite3.connect(DB) as connection:
    payloads = connection.execute(
        "select payload from messages order by created_at"
    ).fetchall()

for (payload,) in payloads:
    diag = json.loads(payload)["diag"]
    diagnostics.append(diag)
    prefill = diag["first_token_time"]
    retrieval = (diag["embed_time"] + diag["search_time"] + diag["bm25_time"]
                 + diag["fusion_time"] + diag["context_time"])
    decode = diag["generation_time"] - prefill
    rows.append({
        "retrieval": retrieval, "prefill": prefill, "decode": decode,
        "other": diag["total_time"] - (retrieval + prefill + decode),
        "total": diag["total_time"], "prompt_tokens": diag["prompt_tokens"],
        "generated_tokens": diag["generated_tokens"],
        "embed_query": diag["embed_time"], "bm25": diag["bm25_time"],
        "faiss_search": diag["search_time"], "rrf_fusion": diag["fusion_time"],
        "build_context": diag["context_time"], "cache_hit": prefill < 1.0,
    })

assert len(rows) == 4
assert [i for i, row in enumerate(rows) if row["cache_hit"]] == [3]
model = Path(diagnostics[0]["model"]).name
assert all(Path(d["model"]).name == model for d in diagnostics)
assert all(d["embedding_device"] == "cpu" for d in diagnostics)


def system_profile() -> tuple[str, str, str]:
    """Mac hardware, with system_profiler as a sandbox-safe fallback."""
    def sysctl(key: str) -> str:
        return subprocess.run(["sysctl", "-n", key],
                              capture_output=True, text=True).stdout.strip()

    chip, cores, memory = (sysctl("machdep.cpu.brand_string"),
                           sysctl("hw.ncpu"), sysctl("hw.memsize"))
    if chip and cores and memory:
        return chip, cores, str(int(memory) // 1_073_741_824)
    profile = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
        capture_output=True, text=True).stdout
    chip_m = re.search(r"^\s*Chip:\s*(.+)$", profile, re.MULTILINE)
    cores_m = re.search(r"^\s*Total Number of Cores:\s*(\d+)", profile, re.MULTILINE)
    mem_m = re.search(r"^\s*Memory:\s*(\d+)\s*GB", profile, re.MULTILINE)
    assert chip_m and cores_m and mem_m, "unable to read Mac hardware"
    return chip_m.group(1), cores_m.group(1), mem_m.group(1)


chip, cores, ram = system_profile()

fig = S.figure(
    "Prompt prefill dominates cold-answer latency",
    "Answer 4 reused an identical prompt in llama.cpp's KV cache and is shown separately.",
)
# `top` is held well below the subtitle baseline (y = 0.892) so panel headings
# cannot collide with the figure header -- the previous layout overlapped them.
grid = fig.add_gridspec(1, 2, left=0.088, right=0.972, bottom=0.235, top=0.665,
                        width_ratios=[1.38, 1.0], wspace=0.34)
ax_total = fig.add_subplot(grid[0])
ax_detail = fig.add_subplot(grid[1])

# ---- Panel A: whole-answer wall clock ---------------------------------------
S.axes(ax_total, grid="x")
S.panel_heading(ax_total, "A", "Wall clock by answer",
                "Seconds, stages stacked", y=1.07)

y = list(range(len(rows)))
ax_total.axhspan(2.5, 3.5, color=S.CACHE_WASH, zorder=0)

STAGES = [("Retrieval", "retrieval", S.BLUE), ("Prefill", "prefill", S.VERMILION),
          ("Decode", "decode", S.TEAL), ("Other", "other", S.LIGHT_GREY)]
max_total = max(row["total"] for row in rows)
left = [0.0] * len(rows)
for label, key, colour in STAGES:
    values = [row[key] for row in rows]
    ax_total.barh(y, values, left=left, height=0.5, color=colour,
                  edgecolor=S.PAPER, linewidth=0.6, label=label, zorder=3)
    for i, value in enumerate(values):
        # Label only a segment wide enough to hold the text. Anything narrower
        # spilled over its neighbour in the previous version.
        if value / max_total >= 0.11:
            ax_total.text(left[i] + value / 2, i, f"{value:.1f}s",
                          ha="center", va="center", fontsize=S.SZ_NOTE,
                          fontfamily="sans-serif", fontweight="bold",
                          color=S.PAPER, zorder=5)
    left = [a + b for a, b in zip(left, values)]

for i, row in enumerate(rows):
    ax_total.text(row["total"] + max_total * 0.022, i,
                  f"{row['total']:.2f}s     {row['prompt_tokens']:,} in / "
                  f"{row['generated_tokens']} out",
                  ha="left", va="center", fontsize=S.SZ_NOTE,
                  fontfamily="sans-serif", color=S.INK_SOFT, zorder=5)

ax_total.set_xlim(0, max_total * 1.52)
ax_total.set_yticks(y)
ax_total.set_yticklabels([f"Answer {i}" for i in range(1, 5)])
ax_total.invert_yaxis()
ax_total.set_xlabel("Elapsed time (seconds)", fontsize=S.SZ_AXIS, labelpad=6)
ax_total.legend(loc="upper left", bbox_to_anchor=(0.0, -0.185), ncol=4,
                fontsize=S.SZ_NOTE, columnspacing=1.4, handlelength=1.1,
                handletextpad=0.45, labelcolor=S.INK_SOFT)

# Sits under the short bar inside the cache-hit band, clear of every other mark.
ax_total.text(0, 3.62,
              "PROMPT-CACHE HIT — prefill 0.23 s, not a typical cold answer",
              ha="left", va="center", fontsize=S.SZ_NOTE,
              fontfamily="sans-serif", fontweight="bold", color=S.PURPLE,
              zorder=7)

# ---- Panel B: retrieval sub-stages, milliseconds, log scale ------------------
S.axes(ax_detail, grid="x")
S.panel_heading(ax_detail, "B", "Retrieval, expanded",
                "Milliseconds, log scale, no averaging", y=1.07)

SUBSTAGES = [("Embed query", "embed_query"), ("BM25", "bm25"),
             ("FAISS search", "faiss_search"), ("RRF fusion", "rrf_fusion"),
             ("Build context", "build_context")]
ANSWER_COLOUR = [S.BLUE, S.VERMILION, S.TEAL, S.PURPLE]
BAR_H = 0.18
FLOOR = 1e-3  # log axes have no zero; bars are drawn from this left edge

for stage_index, (stage_label, key) in enumerate(SUBSTAGES):
    for answer_index, row in enumerate(rows):
        value_ms = row[key] * 1000.0
        offset = (answer_index - 1.5) * BAR_H
        ax_detail.barh(stage_index + offset, value_ms - FLOOR, left=FLOOR,
                       height=BAR_H * 0.8, color=ANSWER_COLOUR[answer_index],
                       edgecolor=S.PAPER, linewidth=0.35, zorder=3)
    if stage_index:
        ax_detail.axhline(stage_index - 0.5, color=S.GRID, lw=0.7, zorder=1)

ax_detail.set_xscale("log")
ax_detail.set_xlim(FLOOR, 2e4)
ax_detail.set_ylim(len(SUBSTAGES) - 0.5, -0.5)
ax_detail.set_yticks(range(len(SUBSTAGES)))
ax_detail.set_yticklabels([label for label, _ in SUBSTAGES])
ax_detail.set_xticks([1e-2, 1e-1, 1, 10, 100, 1000, 1e4])
ax_detail.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: ("0.01" if abs(v - 1e-2) < 1e-9 else
                  "0.1" if abs(v - 1e-1) < 1e-9 else f"{v:,.0f}")))
ax_detail.set_xlabel("Elapsed time (milliseconds, log scale)",
                     fontsize=S.SZ_AXIS, labelpad=6)
ax_detail.legend(
    handles=[Line2D([], [], color=ANSWER_COLOUR[i], lw=5,
                    label=f"Answer {i + 1}") for i in range(4)],
    loc="upper left", bbox_to_anchor=(0.0, -0.185), ncol=4, fontsize=S.SZ_NOTE,
    columnspacing=1.2, handlelength=1.0, handletextpad=0.45,
    labelcolor=S.INK_SOFT)

tiny_ms = max(r["faiss_search"] + r["rrf_fusion"] + r["build_context"]
              for r in rows) * 1000
fig.text(0.088, 0.070,
         f"Embedding the query is nearly all of retrieval: FAISS search, RRF fusion and "
         f"context building together never exceed {tiny_ms:.1f} ms.",
         ha="left", va="bottom", fontsize=S.SZ_NOTE, color=S.INK_FAINT)
fig.text(0.088, 0.028,
         f"{chip}, {cores} cores, {ram} GB RAM, macOS, embedding on CPU, "
         f"generator {model} via llama.cpp, n = {len(rows)} answers.",
         ha="left", va="bottom", fontsize=S.SZ_NOTE, color=S.INK_FAINT)

path = S.save(fig, OUT / "chart 04 - fig_latency_breakdown.png")
for i, row in enumerate(rows, 1):
    print(f"answer {i}: total={row['total']:.2f}s retrieval={row['retrieval']:.2f}s "
          f"prefill={row['prefill']:.2f}s decode={row['decode']:.2f}s "
          f"other={row['other']:.2f}s"
          + ("  [prompt-cache hit]" if row["cache_hit"] else ""))
print("retrieval sub-stages (ms): " + json.dumps(
    {label: [round(r[key] * 1000, 4) for r in rows] for label, key in SUBSTAGES}))
print("wrote", path)
