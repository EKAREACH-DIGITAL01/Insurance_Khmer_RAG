#!/usr/bin/env python3
"""Quality audit for the production Khmer Insurance RAG index.

This complements the retrieval smoke tests. It checks corpus shape and common
failure modes that make Khmer RAG answers look unprofessional: oversized website
chunks, missing website categories, UI/HTML artifacts in indexed text, and known
OCR/generation typos.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_DIR = "BAAI-BGE-M3"
INDEX_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "khmer_insurance.index"
METADATA_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "khmer_insurance_metadata.json"
MODEL_INFO_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "embedding_model_info.json"

MIN_TOTAL_CHUNKS = 3230
MIN_WEBSITE_CHUNKS = 1800
MIN_DOCUMENTS = 850
MAX_WEBSITE_CHUNK_CHARS = 3000
MIN_CATEGORY_CHUNKS = {
    "home": 500,
    "statistics": 140,
    "publication": 230,
    "public_service": 175,
    "insurance_institutions": 20,
    "knowledge": 70,
}
REQUIRED_DOCUMENT_IDS = {
    "website_homepage_management_structure": "management structure",
    "website_homepage_gross_premium_statistics": "homepage gross premium statistics",
    "website_publication_catalog": "publication catalog",
    "website_b95a99c055bd": "public-service Khmer catalog",
    "website_df65e9cbeca4": "public-service English catalog",
    "website_1fab36e16db2": "insurance institutions Khmer directory",
    "website_bbaee43f1602": "insurance institutions English directory",
    "website_file_ceab89ff2b0c": "Bou Chanphirou profile PDF",
}
WEBSITE_DERIVED_CATEGORIES = {
    "announcement",
    "events",
    "home",
    "insurance_institutions",
    "knowledge",
    "law_regulation",
    "news",
    "publication",
    "public_service",
    "social_programs",
    "statistics",
    "website",
}
UI_ARTIFACT_PATTERN = re.compile(
    r"(?:data-tip=|aria-label=|<button|</button|<svg|</svg|<div|</div|<ul|</ul|<li|</li)",
    re.IGNORECASE,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def check(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status} {message}")
    if not condition:
        failures.append(message)


def warn(condition: bool, message: str) -> None:
    status = "WARN" if condition else "PASS"
    print(f"{status} {message}")


def chunk_category(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    return metadata.get("document_type") or metadata.get("category") or "unknown"


def document_id(chunk: dict) -> str:
    return chunk.get("metadata", {}).get("document_id", "")


def chunk_text(chunk: dict) -> str:
    return "\n".join(
        value
        for value in (
            chunk.get("text") or "",
            chunk.get("embedding_text") or "",
        )
        if value
    )


def is_website_chunk(chunk: dict) -> bool:
    return chunk_category(chunk) in WEBSITE_DERIVED_CATEGORIES


def main() -> int:
    failures: list[str] = []

    check(INDEX_PATH.exists(), f"index exists: {INDEX_PATH.relative_to(PROJECT_ROOT)}", failures)
    check(METADATA_PATH.exists(), f"metadata exists: {METADATA_PATH.relative_to(PROJECT_ROOT)}", failures)
    check(MODEL_INFO_PATH.exists(), f"model info exists: {MODEL_INFO_PATH.relative_to(PROJECT_ROOT)}", failures)
    if failures:
        return 1

    try:
        import faiss
    except Exception as exc:
        print(f"FAIL faiss import failed: {exc}")
        return 1

    index = faiss.read_index(str(INDEX_PATH))
    chunks = load_json(METADATA_PATH)
    model_info = load_json(MODEL_INFO_PATH)
    category_counts = Counter(chunk_category(chunk) for chunk in chunks)
    document_ids = {document_id(chunk) for chunk in chunks}
    website_chunks = [chunk for chunk in chunks if is_website_chunk(chunk)]
    website_lengths = [len(chunk.get("text") or "") for chunk in website_chunks]

    print("\nCorpus Shape")
    check(len(chunks) >= MIN_TOTAL_CHUNKS, f"chunk count {len(chunks)} >= {MIN_TOTAL_CHUNKS}", failures)
    check(index.ntotal == len(chunks), f"FAISS vectors {index.ntotal} match metadata chunks {len(chunks)}", failures)
    check(
        int(model_info.get("chunk_count", -1)) == len(chunks),
        f"model-info chunk_count {model_info.get('chunk_count')} matches metadata chunks {len(chunks)}",
        failures,
    )
    check(len(document_ids) >= MIN_DOCUMENTS, f"document count {len(document_ids)} >= {MIN_DOCUMENTS}", failures)
    check(len(website_chunks) >= MIN_WEBSITE_CHUNKS, f"website chunks {len(website_chunks)} >= {MIN_WEBSITE_CHUNKS}", failures)
    check(
        max(website_lengths or [0]) <= MAX_WEBSITE_CHUNK_CHARS,
        f"largest website chunk {max(website_lengths or [0])} <= {MAX_WEBSITE_CHUNK_CHARS} chars",
        failures,
    )

    print("\nCategory Coverage")
    for category, minimum in MIN_CATEGORY_CHUNKS.items():
        check(
            category_counts.get(category, 0) >= minimum,
            f"{category}: {category_counts.get(category, 0)} chunks >= {minimum}",
            failures,
        )
    print("INFO category counts:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")

    print("\nCritical Sources")
    for doc_id, label in REQUIRED_DOCUMENT_IDS.items():
        check(doc_id in document_ids, f"{label} present ({doc_id})", failures)

    print("\nText Hygiene")
    typo_hits = [chunk.get("chunk_id") for chunk in chunks if "ធានរបង" in chunk_text(chunk)]
    artifact_hits = [
        chunk.get("chunk_id")
        for chunk in chunks
        if UI_ARTIFACT_PATTERN.search(chunk_text(chunk))
    ]
    check(not typo_hits, "known typo 'ធានរបង' is absent from indexed text", failures)
    check(not artifact_hits, "HTML/UI action artifacts are absent from indexed text", failures)
    if artifact_hits:
        print("INFO first artifact hits:")
        for chunk_id in artifact_hits[:10]:
            print(f"  - {chunk_id}")

    source_root = PROJECT_ROOT / "data" / "01_source_documents"
    if source_root.exists():
        print("PASS raw source folders are present locally for future OCR refreshes")
    else:
        warn(
            True,
            "raw source folders are not present locally; this is okay for a clone but not for refreshing OCR",
        )

    if failures:
        print(f"\nAudit failed with {len(failures)} issue(s).")
        return 1
    print("\nRAG quality audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
