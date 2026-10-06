"""Diagnostics UI components."""

import html

import streamlit as st

from khmer_rag import storage as history
from khmer_rag.config import LLAMA_SERVER_URL
from khmer_rag.sources import DOCUMENT_TYPE_KH
from khmer_rag.ui.session import open_chat


def console_block(title, rows):
    """Renders one pipeline stage block. The title includes a real
    kc-pipeline-dot element (not a CSS-only pseudo-element) so its
    position on the rail is set by a deterministic negative margin (see
    .kc-pipeline-dot CSS) rather than a guessed absolute offset. This
    couples console_block to the pipeline rail in render_diagnostics_page
    -- fine today since this is its only call site, but worth revisiting
    if console_block is ever reused outside that railed container."""
    row_html = "".join(
        f'<div class="kc-row"><span class="kc-row-label">{html.escape(str(k))}</span>'
        f'<span class="kc-row-value">{html.escape(str(v))}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="kc-console-title kc-pipeline-title">'
        f'<span class="kc-pipeline-dot"></span>&gt; {html.escape(title)}'
        f"</div>"
        f'<div class="kc-console-block">{row_html}</div>',
        unsafe_allow_html=True,
    )


def render_provenance_badge(has_faiss, has_bm25):
    """A small circular badge showing which retrieval engine(s) surfaced
    this chunk. The mapping is a nod to physical seals (ត្រា) stamped on
    Cambodian legal documents: a chunk gets "stamped" by FAISS, by BM25,
    or by both. When both rankers agree, the ring and center dot split
    between the IRC gold and green."""
    GOLD, GREEN = "#B19F50", "#02913F"
    OUTER_RING = "#3A4A41"  # quiet neutral ring, independent of provenance color

    if has_faiss and has_bm25:
        inner = (
            f'<path d="M 3 10 A 7 7 0 0 1 17 10" fill="none" stroke="{GOLD}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
            f'<path d="M 3 10 A 7 7 0 0 0 17 10" fill="none" stroke="{GREEN}" '
            'stroke-width="2.1" stroke-linecap="round"/>'
        )
        # Center dot split the same way as the ring so "both" reads as a
        # literal half-and-half blend of the two engines.
        dot = (
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 1 12.8 10 Z" fill="{GOLD}"/>'
            f'<path d="M 7.2 10 A 2.8 2.8 0 0 0 12.8 10 Z" fill="{GREEN}"/>'
        )
    elif has_faiss:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{GOLD}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{GOLD}"/>'
    elif has_bm25:
        inner = f'<circle cx="10" cy="10" r="7" fill="none" stroke="{GREEN}" stroke-width="2.1"/>'
        dot = f'<circle cx="10" cy="10" r="2.8" fill="{GREEN}"/>'
    else:
        inner, dot = "", ""

    return (
        '<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="10" cy="10" r="9" fill="none" stroke="{OUTER_RING}" stroke-width="1"/>'
        f"{inner}{dot}"
        "</svg>"
    )


def render_diag_source_cards(results, score_lookup):
    # Score bar is normalized relative to the TOP result in this specific
    # list, not assumed to live on a fixed 0-1 range. This list now holds
    # RRF fused scores (see reciprocal_rank_fusion), which are small
    # numbers on a completely different scale than the old raw FAISS
    # cosine scores this card was originally built for -- relative
    # normalization works correctly for either scale without caring which
    # one it's given.
    max_score = max((float(score) for score, _ in results), default=0.0) or 1.0
    cards = []
    for rank, (score, chunk) in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        doc_name = html.escape(metadata.get("document_name", "Unknown Document"))
        chunk_id = html.escape(chunk["chunk_id"])
        pct = max(0.0, min(1.0, float(score) / max_score)) * 100
        delay = (rank - 1) * 0.05

        # A chunk may have been surfaced by FAISS, by BM25, or by both --
        # an em-dash means that particular ranker didn't return this chunk
        # in its own top-K at all, which is itself useful diagnostic
        # information (e.g. a chunk with FAISS "—" and a real BM25 score
        # means lexical search alone found it; dense search missed it).
        raw_scores = score_lookup.get(chunk["chunk_id"], {})
        faiss_val = raw_scores.get("faiss")
        bm25_val = raw_scores.get("bm25")
        faiss_str = f"{faiss_val:.4f}" if faiss_val is not None else "—"
        bm25_str = f"{bm25_val:.4f}" if bm25_val is not None else "—"
        badge_svg = render_provenance_badge(faiss_val is not None, bm25_val is not None)

        score_rows = "".join(
            f'<div class="kc-diag-src-score-row{extra_class}">'
            f'<span class="kc-diag-src-score-key">{label}</span>'
            f'<span class="kc-diag-src-score-val">{value}</span>'
            f"</div>"
            for label, value, extra_class in [
                ("FAISS", faiss_str, ""),
                ("BM25", bm25_str, ""),
                ("RRF", f"{score:.4f}", " kc-diag-src-score-row--rrf"),
            ]
        )

        card = (
            f'<div class="kc-diag-src-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="kc-diag-src-top">'
            f'<span class="kc-diag-src-rank">{rank:02d}</span>'
            f'<div class="kc-diag-src-main">'
            f'<div class="kc-diag-src-name-row">{badge_svg}'
            f'<div class="kc-diag-src-name">{doc_name}</div></div>'
            f'<div class="kc-diag-src-id">{chunk_id}</div>'
            f"</div>"
            f'<div class="kc-diag-src-scores">{score_rows}</div>'
            f"</div>"
            f'<div class="kc-diag-src-bar"><div class="kc-diag-src-fill" '
            f'style="--w:{pct:.1f}%; animation-delay:{delay:.2f}s"></div></div>'
            f"</div>"
        )
        cards.append(card)
    st.markdown(f'<div class="kc-diag-src-list">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_source_detail(rank, score, chunk):
    metadata = chunk.get("metadata", {})
    doc_name = metadata.get("document_name", "N/A")
    rows = [
        ("fused score (rrf)", f"{float(score):.4f}"),
        ("document", doc_name),
        (
            "type",
            metadata.get(
                "document_type_kh",
                DOCUMENT_TYPE_KH.get(
                    metadata.get("document_type"), metadata.get("document_type", "N/A")
                ),
            ),
        ),
        ("release date", metadata.get("release_date", "N/A")),
        ("source url", metadata.get("source_url", "N/A")),
        ("chapter", metadata.get("chapter_title", metadata.get("chapter", "N/A"))),
        ("section", metadata.get("section_title", metadata.get("section", "N/A"))),
        ("legal unit", metadata.get("legal_unit", "N/A")),
        ("chunk id", chunk["chunk_id"]),
    ]
    grid = "".join(
        f'<div class="k">{html.escape(str(k))}</div><div class="v">{html.escape(str(v))}</div>'
        for k, v in rows
    )
    text = html.escape(chunk["text"])
    with st.expander(f"Rank {rank}  —  {doc_name}"):
        st.markdown(
            f'<div class="kc-detail-grid">{grid}</div><div class="kc-detail-text">{text}</div>',
            unsafe_allow_html=True,
        )


def render_diagnostics_page(message_id):
    # The message is looked for in the open conversation first, then in the
    # database -- so this page renders whether it was reached by clicking the
    # info icon or by opening its URL in a fresh session.
    msg = next(
        (m for m in st.session_state["messages"] if m["id"] == message_id),
        None,
    ) or history.load_message(message_id)

    if st.button("← Back to chat", key="back_to_chat"):
        open_chat()
        st.rerun()

    if msg is None:
        st.markdown(
            '<div class="kc-diag-title">No diagnostics found for this message.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="kc-diag-header">'
        f'<div class="kc-diag-title">DIAGNOSTICS — <span class="kc-diag-question">{html.escape(msg["question"])}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    d = msg["diag"]

    def tok(value):
        return f"{value:,}" if value is not None else "N/A"

    col1, col2 = st.columns([1.25, 2.15])
    with col1:
        with st.container(key="kc_diag_pipeline"):
            console_block(
                "SYSTEM",
                [
                    ("generator model", d["model"]),
                    ("generator endpoint", LLAMA_SERVER_URL),
                    ("retriever model", d["embedding_model"]),
                    ("retriever device", d["embedding_device"]),
                    ("retrieval mode", "hybrid (FAISS + BM25, RRF)"),
                    ("top k", d["top_k"]),
                ],
            )
            console_block(
                "RETRIEVAL",
                [
                    ("embed time", f"{d['embed_time']:.4f}s"),
                    ("faiss time", f"{d['search_time']:.4f}s"),
                    ("bm25 time", f"{d['bm25_time']:.4f}s"),
                    ("fusion time", f"{d['fusion_time']:.4f}s"),
                    (
                        "total retrieval time",
                        f"{d['retrieval_time'] + d['hybrid_time']:.4f}s",
                    ),
                    ("best score (faiss, gate)", f"{msg['best_score']:.4f}"),
                    ("min score (faiss)", f"{d['min_score']:.4f}"),
                    ("spread score (faiss)", f"{d['spread_score']:.4f}"),
                ],
            )
            console_block(
                "CONTEXT",
                [
                    ("context length", f"{d['context_length']} chars"),
                    ("context token", tok(d["context_tokens"])),
                    ("context build time", f"{d['context_time']:.4f}s"),
                    ("prompt length", f"{d['prompt_length']} chars"),
                    ("prompt token", tok(d["prompt_tokens"])),
                ],
            )
            console_block(
                "GENERATION",
                [
                    ("first token", f"{d['first_token_time']:.4f}s"),
                    ("generation time", f"{d['generation_time']:.4f}s"),
                    (
                        "tokens/sec (decode)",
                        f"{d['generated_tokens'] / max(d['generation_time'] - d['first_token_time'], 0.001):.4f}",
                    ),
                    ("total time", f"{d['total_time']:.4f}s"),
                    ("output token", tok(d["generated_tokens"])),
                    ("generated chars", d["generated_chars"]),
                ],
            )
    with col2:
        st.markdown(
            f'<div class="kc-console-title">&gt; CHUNKS (TOP {len(msg["all_results"])}, FUSED RANKING)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kc-diag-legend">'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(True, False)}<span>FAISS only</span></div>'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(False, True)}<span>BM25 only</span></div>'
            f'<div class="kc-diag-legend-item">{render_provenance_badge(True, True)}<span>both (confirmed)</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        render_diag_source_cards(msg["all_results"], msg.get("score_lookup", {}))

    st.markdown(
        '<div class="kc-console-title" style="margin-top:18px">&gt; FULL SOURCE DETAILS</div>',
        unsafe_allow_html=True,
    )
    for rank, (score, chunk) in enumerate(msg["all_results"], start=1):
        render_source_detail(rank, score, chunk)
