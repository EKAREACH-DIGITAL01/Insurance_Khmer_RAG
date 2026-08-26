#!/usr/bin/env python3
"""Reclassify existing IRC website corpus documents into section folders.

Use this after changing the website category map, or when the local crawl/OCR
cache already exists and you want to avoid re-scraping/re-OCRing. It only moves
website-derived documents whose IDs start with ``website``; the original student
PDF corpus folders are left untouched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES, infer_website_category


MERGED_ROOT = Path("data/05_merged_documents")
CLEANED_ROOT = Path("data/06_cleaned_documents")
CHUNKED_ROOT = Path("data/07_chunked_documents")
MERGED_CHUNKS_ROOT = Path("data/08_merged_chunked_documents")
METADATA_PATH = Path("data/document_metadata.json")

LEGACY_CATEGORY = "website"
ALL_WEBSITE_FOLDERS = [LEGACY_CATEGORY] + WEBSITE_DERIVED_CATEGORIES
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPECIAL_DOC_CATEGORIES = {
    "website_homepage_management_structure": "home",
    "website_homepage_gross_premium_statistics": "home",
    "website_publication_catalog": "publication",
    # These PDFs are listed under the IRC publication page. Some object-file
    # URLs do not carry a parent article/category once cached locally, so keep
    # them in publication instead of letting the fallback put them under home.
    "website_file_0a0b637917e6": "publication",
    "website_file_2f70838051e2": "publication",
    "website_file_39f076910114": "publication",
    "website_file_646db1aa65d3": "publication",
    "website_file_723cfd5541c1": "publication",
    "website_file_a2a9f95557bc": "publication",
    "website_file_c4a172db1586": "publication",
    "website_file_db047ce07365": "publication",
    "website_file_ea0a6e39a7f9": "publication",
}


def load_json(path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def first_title_line(text):
    for line in (text or "").splitlines():
        title = line.strip()
        if title:
            return title
    return ""


def is_management_image_doc(doc):
    source_url = doc.get("source_url") or ""
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix.lower()
    return "/files/cv/" in parsed.path.lower() and suffix in IMAGE_EXTENSIONS


def collect_website_documents():
    docs = {}
    source_paths = {}
    for folder in ALL_WEBSITE_FOLDERS:
        root = MERGED_ROOT / folder
        if not root.exists():
            continue
        for path in sorted(root.glob("website*.json")):
            doc = load_json(path, {})
            doc_id = doc.get("document_id") or path.stem
            if not doc_id.startswith("website"):
                continue
            if is_management_image_doc(doc):
                continue
            current = docs.get(doc_id)
            current_text_len = len((current or {}).get("merged_text") or "")
            new_text_len = len(doc.get("merged_text") or "")
            if current is None or new_text_len >= current_text_len:
                docs[doc_id] = doc
                source_paths[doc_id] = path
    return docs, source_paths


def clear_stale_generated_outputs():
    removed = 0
    for root in (MERGED_ROOT, CLEANED_ROOT):
        for folder in ALL_WEBSITE_FOLDERS:
            category_root = root / folder
            if not category_root.exists():
                continue
            for path in category_root.glob("website*.json"):
                path.unlink()
                removed += 1

    for folder in ALL_WEBSITE_FOLDERS:
        category_root = CHUNKED_ROOT / folder
        if not category_root.exists():
            continue
        for path in category_root.glob("website*_chunks.json"):
            path.unlink()
            removed += 1

    for folder in ALL_WEBSITE_FOLDERS:
        path = MERGED_CHUNKS_ROOT / f"{folder}_chunks.json"
        if path.exists():
            path.unlink()
            removed += 1

    return removed


def infer_doc_category(doc_id, doc):
    if doc_id in SPECIAL_DOC_CATEGORIES:
        return SPECIAL_DOC_CATEGORIES[doc_id]
    text = doc.get("merged_text") or ""
    return infer_website_category(
        source_url=doc.get("source_url"),
        parent_url=doc.get("parent_url"),
        title=first_title_line(text),
        text=text,
    )


def update_metadata(metadata, doc_id, doc, category):
    existing = metadata.get(doc_id, {})
    text = doc.get("merged_text") or ""
    title = existing.get("document_name") or first_title_line(text) or doc_id
    scraped_at = doc.get("scraped_at") or ""
    release_date = existing.get("release_date") or scraped_at[:10]
    updated = dict(existing)
    updated.update(
        {
            "document_name": title,
            "document_type": category,
            "release_date": release_date,
            "source_url": doc.get("source_url") or existing.get("source_url"),
            "source_section": category,
        }
    )
    metadata[doc_id] = updated


def main():
    docs, source_paths = collect_website_documents()
    if not docs:
        raise SystemExit("No website-derived merged documents found to reclassify.")

    metadata = load_json(METADATA_PATH, {})
    removed = clear_stale_generated_outputs()

    counts = Counter()
    for doc_id, doc in sorted(docs.items()):
        category = infer_doc_category(doc_id, doc)
        doc["document_id"] = doc_id
        doc["category"] = category
        save_json(MERGED_ROOT / category / f"{doc_id}.json", doc)
        update_metadata(metadata, doc_id, doc, category)
        counts[category] += 1

    save_json(METADATA_PATH, metadata)

    print("\n========== WEBSITE CATEGORY RECLASSIFICATION ==========")
    print(f"Documents collected : {len(docs)}")
    print(f"Generated files removed: {removed}")
    print("Category counts:")
    for category, count in counts.most_common():
        print(f"  {category:24s} {count}")
    print(f"Metadata updated   : {METADATA_PATH}")
    print("Original student PDF categories were not modified.")


if __name__ == "__main__":
    main()
