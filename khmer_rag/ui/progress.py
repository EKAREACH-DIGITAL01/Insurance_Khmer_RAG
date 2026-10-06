"""Progress UI components."""

from khmer_rag.config import MODEL

PHASES = [
    "encoding_query",
    "searching_faiss",
    "searching_bm25",
    "fusing_results",
    "building_context",
    "constructing_prompt",
    "preparing_response",
    "generating_response",
    "completed",
]

PHASE_LABELS = {
    "encoding_query": (
        "<b>Encoding Query</b> - <i>BGE-M3 is converting the question into an embedding vector</i>"
    ),
    "searching_faiss": (
        "<b>Searching FAISS Index</b> - "
        "<i>FAISS is searching the vector database for relevant chunks</i>"
    ),
    "searching_bm25": (
        "<b>Searching BM25 Index</b> - "
        "<i>Lexical keyword search is running alongside semantic search</i>"
    ),
    "fusing_results": (
        "<b>Fusing Rankings</b> - <i>Semantic and keyword rankings are being combined</i>"
    ),
    "building_context": (
        "<b>Building Retrieval Context</b> - "
        "<i>Retrieved chunks are being assembled into context</i>"
    ),
    "constructing_prompt": (
        "<b>Constructing Prompt</b> - <i>Context, instructions, and question are being merged</i>"
    ),
    "preparing_response": (
        "<b>Preparing Response</b> - "
        f"<i>{MODEL} is processing the prompt and preparing the first token</i>"
    ),
    "generating_response": (
        "<b>Generating Response</b> - "
        f"<i>{MODEL} is generating the answer from retrieved context</i>"
    ),
    "completed": ("<b>Completed</b>"),
}


def render_ladder_html(current_idx, phase_elapsed):
    rows = []
    for i, key in enumerate(PHASES):
        label_html = PHASE_LABELS[key]
        elapsed = phase_elapsed.get(key)
        time_str = f"{elapsed:.4f}s" if elapsed is not None else ""
        if i < current_idx or (key == "completed" and i == current_idx):
            icon = '<span class="kc-step-icon kc-step-done">&#10003;</span>'
            row_cls = "kc-step-row"
        elif i == current_idx:
            icon = '<span class="kc-step-icon kc-step-active"></span>'
            row_cls = "kc-step-row"
        else:
            icon = '<span class="kc-step-icon kc-step-pending"></span>'
            row_cls = "kc-step-row kc-step-row-pending"
        rows.append(
            f'<div class="{row_cls}">'
            f"{icon}"
            f'<span class="kc-step-label">{label_html}</span>'
            f'<span class="kc-step-time">{time_str}</span>'
            f"</div>"
        )
    return f'<div class="kc-ladder">{"".join(rows)}</div>'
