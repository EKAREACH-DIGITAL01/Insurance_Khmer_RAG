#!/usr/bin/env python3
"""Arrange raw IRC website downloads into data/01_source_documents categories.

The website scraper keeps raw downloads in data/00_irc_website_scrape/files,
which is intentionally gitignored. This script creates a review-friendly view
of those same raw files under data/01_source_documents/<website-category>/.

By default files are hard-linked, so the 01 folder looks complete without
duplicating hundreds of megabytes on disk. If hard links are not available, the
script falls back to copying.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES, infer_website_category


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_ROOT = PROJECT_ROOT / "data/00_irc_website_scrape/files"
PAGE_ROOT = PROJECT_ROOT / "data/00_irc_website_scrape/pages"
MERGED_DOC_ROOT = PROJECT_ROOT / "data/05_merged_documents"
SOURCE_ROOT = PROJECT_ROOT / "data/01_source_documents"
MANIFEST_PATH = SOURCE_ROOT / "irc_raw_sources_manifest.json"

RAW_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}

DIRECT_SECTION_URLS = {
    "insurance_institutions": [
        "https://irc.gov.kh/insurance-institutions",
        "https://www.irc.gov.kh/insurance-institutions",
        "https://irc.gov.kh/en/insurance-institutions",
        "https://www.irc.gov.kh/en/insurance-institutions",
    ],
    "public_service": [
        "https://irc.gov.kh/public-service",
        "https://www.irc.gov.kh/public-service",
        "https://irc.gov.kh/en/public-service",
        "https://www.irc.gov.kh/en/public-service",
        "https://www.irc.gov.kh/agent/registration-menu",
        "https://www.irc.gov.kh/en/agent/registration-menu",
        "https://www.irc.gov.kh/complaint",
        "https://www.irc.gov.kh/en/complaint",
    ],
}


@dataclass
class AssetRecord:
    source_path: Path
    category: str
    title: str
    source_url: str = ""
    parent_url: str = ""
    document_id: str = ""
    reason: str = ""


def safe_cache_filename(url: str) -> str:
    """Match the downloader's filename normalization in 00_scrape_irc_website."""
    name = Path(urlparse(url).path).name or "file"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "file"


def clean_label(value: str, fallback: str = "source") -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = re.sub(r"[\\/:*?\"<>|\0]+", " - ", value)
    value = re.sub(r"\s+", " ", value).strip(" .-_")
    if not value:
        value = fallback
    if len(value) > 90:
        value = value[:87].rstrip() + "..."
    return value


def read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def is_raw_asset(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in RAW_EXTENSIONS


def title_from_text(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("Source file:"):
            continue
        return line
    return fallback


def choose_title(candidates: list[str], fallback: str) -> str:
    clean = [c.strip() for c in candidates if c and c.strip()]
    if not clean:
        return fallback
    non_generic = [
        c
        for c in clean
        if c.casefold()
        not in {
            "news",
            "event",
            "publication",
            "prakas",
            "laws",
            "guidance",
            "monthly statistics",
            "quarterly statistics",
            "annually statistics",
        }
    ]
    return (non_generic or clean)[0]


def collect_page_link_records(cache: dict[str, Path]) -> dict[str, AssetRecord]:
    category_votes: dict[str, Counter[str]] = defaultdict(Counter)
    article_category_votes: dict[str, Counter[str]] = defaultdict(Counter)
    titles: dict[str, list[str]] = defaultdict(list)
    article_titles: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    urls: dict[str, list[str]] = defaultdict(list)

    for page_path in sorted(PAGE_ROOT.glob("*.json")):
        page = read_json(page_path)
        if not page:
            continue
        page_url = page.get("url", "")
        page_title = page.get("title", "")
        page_text = page.get("text", "")
        category = infer_website_category(
            source_url=page_url,
            title=page_title,
            text=page_text,
        )
        parsed_page_url = urlparse(page_url)
        is_article_page = parsed_page_url.path.endswith("/article") and "post=" in parsed_page_url.query
        for link in page.get("links") or []:
            href = link.get("url") if isinstance(link, dict) else str(link)
            filename = safe_cache_filename(href)
            if filename not in cache:
                continue
            category_votes[filename][category] += 1
            titles[filename].append(page_title)
            if is_article_page:
                article_category_votes[filename][category] += 1
                article_titles[filename].append(page_title)
            parents[filename].append(page_url)
            urls[filename].append(href)

    records = {}
    for filename, votes in category_votes.items():
        if article_category_votes.get(filename):
            votes = article_category_votes[filename]
        category = votes.most_common(1)[0][0]
        title_candidates = article_titles[filename] or titles[filename]
        records[filename] = AssetRecord(
            source_path=cache[filename],
            category=category,
            title=choose_title(title_candidates, filename),
            source_url=urls[filename][0] if urls[filename] else "",
            parent_url=parents[filename][0] if parents[filename] else "",
            reason="linked from cached page",
        )
    return records


def overlay_institution_logo_records(
    records: dict[str, AssetRecord], cache: dict[str, Path]
) -> None:
    """Use the structured institutions directory to give logo files useful names."""
    logo_line_pattern = re.compile(r"^- (?P<name>.+?) \| Type: (?P<type>.+?) \| Logo: (?P<url>https?://\S+)")
    for doc_path in sorted((MERGED_DOC_ROOT / "insurance_institutions").glob("*.json")):
        doc = read_json(doc_path)
        if not doc:
            continue
        text = doc.get("merged_text") or doc.get("text") or ""
        for line in text.splitlines():
            match = logo_line_pattern.match(line.strip())
            if not match:
                continue
            logo_url = match.group("url")
            filename = safe_cache_filename(logo_url)
            if filename not in cache:
                continue
            institution_name = clean_label(match.group("name"), filename)
            institution_type = clean_label(match.group("type"), "institution")
            records[filename] = AssetRecord(
                source_path=cache[filename],
                category="insurance_institutions",
                title=f"{institution_name} - {institution_type}",
                source_url=logo_url,
                parent_url=doc.get("source_url", ""),
                document_id=doc.get("document_id", ""),
                reason="structured insurance institution logo",
            )


def overlay_merged_document_records(
    records: dict[str, AssetRecord], cache: dict[str, Path]
) -> None:
    for doc_path in sorted(MERGED_DOC_ROOT.glob("*/*.json")):
        if not doc_path.name.startswith("website_file"):
            continue
        doc = read_json(doc_path)
        if not doc:
            continue
        source_url = doc.get("source_url", "")
        filename = safe_cache_filename(source_url)
        if filename not in cache:
            continue
        category = doc.get("category") or doc_path.parent.name
        if category not in WEBSITE_DERIVED_CATEGORIES:
            category = infer_website_category(
                source_url=source_url,
                parent_url=doc.get("parent_url"),
                text=doc.get("merged_text"),
            )
        existing_record = records.get(filename)
        if existing_record and category == "home" and existing_record.category != "home":
            category = existing_record.category
        records[filename] = AssetRecord(
            source_path=cache[filename],
            category=category,
            title=title_from_text(doc.get("merged_text", ""), filename),
            source_url=source_url,
            parent_url=doc.get("parent_url", ""),
            document_id=doc.get("document_id", ""),
            reason="merged website file document",
        )


def fill_unmapped_records(records: dict[str, AssetRecord], cache: dict[str, Path]) -> None:
    for filename, source_path in cache.items():
        if filename in records:
            continue
        category = infer_website_category(source_url=filename, title=filename, fallback="home")
        records[filename] = AssetRecord(
            source_path=source_path,
            category=category,
            title=filename,
            reason="fallback category from filename",
        )


def prepare_category_dirs(clear: bool) -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    for category in WEBSITE_DERIVED_CATEGORIES:
        target = SOURCE_ROOT / category
        target.mkdir(parents=True, exist_ok=True)
        if clear:
            for child in target.iterdir():
                if child.name in {".gitkeep", "README.md"}:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        (target / ".gitkeep").touch()


def unique_destination(target_dir: Path, desired_name: str) -> Path:
    candidate = target_dir / desired_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(2, 10_000):
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique destination for {desired_name}")


def destination_name(index: int, record: AssetRecord) -> str:
    title = clean_label(record.title, record.source_path.stem)
    source_name = clean_label(record.source_path.name, record.source_path.name)
    if record.document_id:
        prefix = record.document_id
    else:
        prefix = f"raw_{index:04d}"
    name = f"{prefix}__{title}__{source_name}"
    if len(name) > 210:
        keep = 210 - len(prefix) - len(source_name) - 4
        title = clean_label(title[: max(20, keep)], record.source_path.stem)
        name = f"{prefix}__{title}__{source_name}"
    return name


def link_or_copy(source: Path, destination: Path, hardlink: bool) -> str:
    if destination.exists():
        return "exists"
    if hardlink:
        try:
            os.link(source, destination)
            return "hard_linked"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copied"


def write_readmes(counts: Counter[str]) -> None:
    for category in WEBSITE_DERIVED_CATEGORIES:
        lines = [
            f"# {category}",
            "",
            "Generated local view of raw files from data/00_irc_website_scrape/files.",
            f"Raw assets currently arranged here: {counts.get(category, 0)}.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            ".venv/bin/python scripts/21_arrange_irc_raw_sources.py --clear",
            "```",
        ]
        if category in DIRECT_SECTION_URLS:
            lines.extend(
                [
                    "",
                    "Direct website routes to scrape for this section:",
                    "",
                    *[f"- {url}" for url in DIRECT_SECTION_URLS[category]],
                ]
            )
        (SOURCE_ROOT / category / "README.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def arrange(clear: bool, hardlink: bool) -> dict:
    if not RAW_CACHE_ROOT.exists():
        raise FileNotFoundError(f"Missing raw cache folder: {RAW_CACHE_ROOT}")

    cache = {path.name: path for path in sorted(RAW_CACHE_ROOT.iterdir()) if is_raw_asset(path)}
    prepare_category_dirs(clear=clear)

    records = collect_page_link_records(cache)
    overlay_merged_document_records(records, cache)
    overlay_institution_logo_records(records, cache)
    fill_unmapped_records(records, cache)

    manifest_records = []
    counts: Counter[str] = Counter()
    actions: Counter[str] = Counter()

    for index, record in enumerate(sorted(records.values(), key=lambda item: item.source_path.name), 1):
        category = record.category if record.category in WEBSITE_DERIVED_CATEGORIES else "home"
        target_dir = SOURCE_ROOT / category
        target_path = unique_destination(target_dir, destination_name(index, record))
        action = link_or_copy(record.source_path, target_path, hardlink=hardlink)
        counts[category] += 1
        actions[action] += 1
        manifest_records.append(
            {
                "category": category,
                "source_file": str(record.source_path.relative_to(PROJECT_ROOT)),
                "arranged_file": str(target_path.relative_to(PROJECT_ROOT)),
                "title": record.title,
                "source_url": record.source_url,
                "parent_url": record.parent_url,
                "document_id": record.document_id,
                "reason": record.reason,
                "action": action,
                "size_bytes": record.source_path.stat().st_size,
            }
        )

    write_readmes(counts)
    manifest = {
        "source_cache": str(RAW_CACHE_ROOT.relative_to(PROJECT_ROOT)),
        "target_root": str(SOURCE_ROOT.relative_to(PROJECT_ROOT)),
        "total_raw_assets": len(cache),
        "total_arranged": len(manifest_records),
        "counts_by_category": dict(sorted(counts.items())),
        "actions": dict(sorted(actions.items())),
        "records": manifest_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arrange cached IRC website raw PDFs/images into source category folders."
    )
    parser.add_argument("--clear", action="store_true", help="Clear generated files in website category folders first.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hard links.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = arrange(clear=args.clear, hardlink=not args.copy)
    print(f"Arranged raw assets: {manifest['total_arranged']}")
    for category in WEBSITE_DERIVED_CATEGORIES:
        print(f"  {category}: {manifest['counts_by_category'].get(category, 0)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
