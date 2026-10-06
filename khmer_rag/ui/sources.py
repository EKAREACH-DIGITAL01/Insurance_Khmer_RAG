"""Sources UI components."""

import html
import os
import shutil

import streamlit as st

from khmer_rag.config import PROJECT_ROOT
from khmer_rag.sources import (
    SOURCE_PDF_DIRS,
    document_id_for_chunk,
)


@st.cache_resource(show_spinner=False)
def publish_source_pdfs():
    """Publish legal PDFs through Streamlit's static directory.

    Prefer hard links, falling back to copies across filesystems. Replace
    stale links only inside static/documents and skip unavailable sources."""
    static_dir = os.path.join(PROJECT_ROOT, "static", "documents")
    os.makedirs(static_dir, exist_ok=True)
    urls = {}
    for doc_type, folder in SOURCE_PDF_DIRS.items():
        src_dir = os.path.join(PROJECT_ROOT, "data", "01_source_documents", folder)
        if not os.path.isdir(src_dir):
            continue
        for filename in os.listdir(src_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            doc_id = os.path.splitext(filename)[0]
            src = os.path.join(src_dir, filename)
            dst = os.path.join(static_dir, filename)
            try:
                # A link that no longer points at the same inode as the source
                # is stale -- the PDF behind it was replaced -- and would keep
                # serving the old document. Relink rather than skip.
                #
                # The guard is not paranoia: this is the one place in the app
                # that deletes a file, and it sits a couple of path joins away
                # from the irreplaceable source corpus. If `dst` ever resolved
                # outside static/ -- a symlinked static/documents, a doctored
                # filename -- os.remove would take the original PDF instead of
                # the link. Verify where the path really points before unlinking.
                if os.path.exists(dst) and os.stat(dst).st_ino != os.stat(src).st_ino:
                    if os.path.realpath(dst).startswith(os.path.realpath(static_dir) + os.sep):
                        os.remove(dst)
                    else:
                        continue
                if not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
                urls[doc_id] = f"app/static/documents/{filename}"
            except OSError:
                continue
    return urls


def render_used_sources(used_sources):
    """Render one escaped link per source document, with text for missing URLs."""
    seen = []
    for score, chunk in used_sources:
        metadata = chunk.get("metadata", {})
        name = metadata.get("document_name", "Unknown Document")
        source_url = metadata.get("source_url")
        doc_id = document_id_for_chunk(chunk["chunk_id"])
        if not any(d == doc_id for d, _, _ in seen):
            seen.append((doc_id, name, source_url))

    if not seen:
        return

    pdf_urls = publish_source_pdfs()
    items = []
    for doc_id, name, source_url in seen:
        label = html.escape(name)
        url = source_url or pdf_urls.get(doc_id)
        # A document with no PDF on disk still gets listed -- it is a real
        # source of the answer -- just without a link to follow.
        items.append(
            f'<li><a href="{html.escape(url)}" target="_blank" rel="noopener">{label}</a></li>'
            if url
            else f"<li>{label}</li>"
        )

    st.markdown(
        f'<div class="kc-sources-title">មើលប្រភពឯកសារ</div>'
        f'<ul class="kc-sources-list">{"".join(items)}</ul>',
        unsafe_allow_html=True,
    )
