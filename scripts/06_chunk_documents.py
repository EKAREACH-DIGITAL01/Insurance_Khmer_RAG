import re
import json
import time
from pathlib import Path

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES

input_root = Path("data/06_cleaned_documents")
output_root = Path("data/07_chunked_documents")

output_root.mkdir(parents=True, exist_ok=True)

# Find all JSONs recursively
json_files = sorted(input_root.rglob("*.json"))

# Document-level metadata
with open("data/document_metadata.json", "r", encoding="utf-8") as f:
    DOCUMENT_METADATA = json.load(f)

CATEGORY_CONFIG = {
    "guidance":               {"chunking_method": "guidance_heading"},
    "laws":                   {"chunking_method": "legal_unit"},
    "prakas":                 {"chunking_method": "legal_unit"},
    "sub_decrees":            {"chunking_method": "legal_unit"},
    "home":                   {"chunking_method": "homepage_blocks", "max_chars": 1400},
    "statistics":             {"chunking_method": "statistics_table_window", "max_chars": 2200},
    "publication":            {"chunking_method": "publication_section_window", "max_chars": 1700},
    "public_service":         {"chunking_method": "public_service_steps", "max_chars": 1400},
    "insurance_institutions": {"chunking_method": "institution_directory", "max_chars": 1400},
    "knowledge":              {"chunking_method": "knowledge_article", "max_chars": 1400},
    "news":                   {"chunking_method": "news_article", "max_chars": 1400},
    "events":                 {"chunking_method": "event_article", "max_chars": 1400},
    "social_programs":        {"chunking_method": "social_program_article", "max_chars": 1400},
    "announcement":           {"chunking_method": "announcement_notice", "max_chars": 1400},
    "law_regulation":         {"chunking_method": "website_law_regulation", "max_chars": 1600},
    "website":                {"chunking_method": "legacy_website_paragraph_window", "max_chars": 1400},
}

LEGAL_CATEGORIES = {"laws", "prakas", "sub_decrees"}
WEBSITE_CHUNKING_CATEGORIES = set(WEBSITE_DERIVED_CATEGORIES) | {"website"}

# The IRC publication page contains Khmer and English versions of the same
# reports/plans. The chatbot is Khmer-only, and duplicated English annual
# reports make retrieval noisier for Khmer demo questions, so we keep the raw
# files/OCR cache but skip these English publication PDFs in the embedded index.
SKIP_PUBLICATION_DOCUMENT_IDS = {
    "website_file_646db1aa65d3",  # Annual Report 2021 on Cambodia Insurance Market
    "website_file_723cfd5541c1",  # Cambodia Strategic Development Plan 2025-2030
    "website_file_ea0a6e39a7f9",  # Annual Report 2025 on Cambodia Insurance Market
}

processed_files = 0
skipped_files = 0
total_chunks = 0

start_time = time.time()


# Chunk guidance documents by normalized heading hierarchy (<H1>-<H4>)
# Each heading becomes one chunk and inherits its parent H1 context
def guidance_chunking(text):

    heading_pattern = r"(?m)(?=^<H[1-4]>\s)"
    sections = re.split(heading_pattern, text)
    chunks = []
    parent_h1 = None

    for section in sections:
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()
        first_line = lines[0].strip()
        match = re.match(r"^<H([1-4])>\s+([0-9\.]+)\s+(.*)$", first_line)

        if not match:
            # Preserve document introduction before the first heading
            chunks.append(
                {
                    "content"       : section.strip(),
                    "heading_level" : 0,
                    "heading_id"    : None,
                    "heading_title" : None,
                    "parent_h1"     : None,
                }
            )
            continue

        level = int(match.group(1))
        heading_id = match.group(2)
        heading_title = match.group(3).strip()

        if level == 1:
            parent_h1 = heading_title

        content = "\n".join(lines[1:]).strip()

        if not content:
            continue

        chunks.append(
            {
                "content"       : content,
                "heading_level" : level,
                "heading_id"    : heading_id,
                "heading_title" : heading_title,
                "parent_h1"     : parent_h1,
            }
        )

    return chunks


# Chunk legal documents at the legal-unit level
# (មាត្រា for laws/sub-decrees, ប្រការ for prakas)
def legal_unit_chunking(text):

    chapter_pattern = (
        r"(?=" r"(?:" r"^ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+" r"|" r"^Chapter\s+[0-9]+" r")" r")"
    )

    article_pattern = (
        r"(?="
        r"(?:"
        r"^មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+"
        r"|"
        r"^ប្រការ\s+[0-9០១២៣៤៥៦៧៨៩]+"
        r")"
        r")"
    )

    chapter_sections = re.split(chapter_pattern, text, flags=re.MULTILINE)

    chunks = []

    for chapter_text in chapter_sections:
        chapter_text = chapter_text.strip()

        if not chapter_text:
            continue

        # Extract chapter identifier and human-readable chapter title
        chapter_match = re.search(r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)", chapter_text, flags=re.MULTILINE)

        chapter_name = None
        chapter_title = None

        if chapter_match:
            chapter_name = chapter_match.group(1)
            lines = chapter_text.splitlines()

            for i, line in enumerate(lines):
                if line.strip() == chapter_name:
                    if i + 1 < len(lines):
                        chapter_title = lines[i + 1].strip()

                    break

        section_matches = list(re.finditer(r"^(ផ្នែកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)", chapter_text, flags=re.MULTILINE))

        articles = re.split(article_pattern, chapter_text, flags=re.MULTILINE)

        for article in articles:
            article = article.strip()

            if not article:
                continue

            article_start = chapter_text.find(article)

            section_name = None
            section_title = None

            # Determine which section the current legal unit belongs to
            for section_match in section_matches:
                if section_match.start() < article_start:
                    section_name = section_match.group(1)
                    section_lines = chapter_text[section_match.start() :].splitlines()

                    if len(section_lines) > 1:
                        section_title = section_lines[1].strip()

                else:
                    break

            article_match = re.search(
                r"^(មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+" r"|" r"ប្រការ\s+[0-9០១២៣៤៥៦៧៨៩]+)",
                article,
                flags=re.MULTILINE,
            )

            if not article_match:
                continue

            article_name = article_match.group(1)

            # Remove trailing next section header
            article = re.sub(
                r"\n\s*" r"(ផ្នែកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)" r"\s*\n.*$",
                "",
                article,
                flags=re.DOTALL,
            ).strip()

            chunks.append(
                {
                    "content"       : article,
                    "chapter"       : chapter_name,
                    "chapter_title" : chapter_title,
                    "section"       : section_name,
                    "section_title" : section_title,
                    "legal_unit"    : article_name,
                }
            )

    return chunks


def hard_wrap_text(text, max_chars):
    """Last-resort split for a single oversized line.

    Most Khmer website pages arrive as useful newline-separated records. A few
    OCR/table fragments arrive as one very long line; keeping those as one chunk
    makes retrieval less precise and can swamp the generation prompt. This
    fallback wraps on whitespace where possible, then on character count.
    """
    pieces = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars * 0.55:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_oversized_paragraph(paragraph, max_chars):
    """Split one long paragraph into line/sentence windows.

    The IRC website often renders catalogs as a single block separated by
    single newlines, not blank lines. The old paragraph-only chunker therefore
    produced 14k-27k character chunks for public-service and institution
    directory pages. This keeps natural list lines together while enforcing the
    per-category max_chars target used by the rest of the pipeline.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[។៕.!?])\s+", paragraph)
        if part.strip()
    ]
    if len(sentence_parts) > 1:
        return sentence_parts

    return hard_wrap_text(paragraph, max_chars)


def paragraph_window_chunking(text, max_chars=1400):
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paragraphs = []
    for paragraph in raw_paragraphs:
        paragraphs.extend(split_oversized_paragraph(paragraph, max_chars))

    chunks = []
    current = []
    current_size = 0

    for paragraph in paragraphs:
        if current and current_size + len(paragraph) > max_chars:
            chunks.append({"content": "\n\n".join(current)})
            current = []
            current_size = 0
        if len(paragraph) > max_chars and not current:
            for piece in hard_wrap_text(paragraph, max_chars):
                chunks.append({"content": piece})
            current_size = 0
            continue
        current.append(paragraph)
        current_size += len(paragraph)

    if current:
        chunks.append({"content": "\n\n".join(current)})

    return chunks


KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
PUBLICATION_PAGE_MARKER_RE = re.compile(r"^\[Page\s+[0-9]+\]$")
PUBLICATION_HEADING_RE = re.compile(
    r"^(?:"
    r"ផ្នែកទី\s*[0-9" + KHMER_DIGITS + r"]+\s*[៖:].+"
    r"|ជំពូកទី\s*[0-9" + KHMER_DIGITS + r"]+\s*[៖:].+"
    r"|[0-9" + KHMER_DIGITS + r"]+(?:\.[0-9" + KHMER_DIGITS + r"]+){1,3}\.?\s+.+"
    r")$"
)
PUBLICATION_NUMBERED_HEADING_RE = re.compile(
    r"^([0-9" + KHMER_DIGITS + r"]+(?:\.[0-9" + KHMER_DIGITS + r"]+){1,3})\.?\s+(.+)$"
)


def strip_publication_heading_markup(line):
    line = line.strip()
    line = re.sub(r"^\*+\s*", "", line)
    line = re.sub(r"\s*\*+$", "", line)
    return line.strip()


def is_table_of_contents_line(line):
    return "....." in line or line.count(".") >= 8


def is_publication_heading(line):
    line = strip_publication_heading_markup(line)
    if not line or is_table_of_contents_line(line):
        return False
    return bool(PUBLICATION_HEADING_RE.match(line))


def publication_heading_level(line):
    line = strip_publication_heading_markup(line)
    if line.startswith(("ផ្នែកទី", "ជំពូកទី")):
        return 1
    match = PUBLICATION_NUMBERED_HEADING_RE.match(line)
    if match:
        # ៤.១ -> level 2, ៤.១.១ -> level 3.
        return min(match.group(1).count(".") + 1, 4)
    return 2


def clean_publication_heading_title(line):
    line = strip_publication_heading_markup(line)
    return re.sub(r"\s+", " ", line).strip()


def split_publication_section(lines, heading_path, max_chars):
    prefix_lines = []
    body_lines = []
    body_started = False

    for line in lines:
        clean = clean_publication_heading_title(line)
        if (
            not body_started
            and (PUBLICATION_PAGE_MARKER_RE.match(clean) or is_publication_heading(clean))
        ):
            if clean not in prefix_lines:
                prefix_lines.append(clean)
            continue
        body_started = True
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if not body:
        return []

    prefix = "\n".join(prefix_lines).strip()
    full_text = f"{prefix}\n\n{body}".strip() if prefix else body
    section_title = heading_path[-1] if heading_path else None
    chunk_base = {
        "heading_level": len(heading_path) if heading_path else 0,
        "heading_title": section_title,
        "section": section_title,
        "section_title": section_title,
        "heading_path": " > ".join(heading_path) if heading_path else None,
    }

    if len(full_text) <= max_chars:
        return [{**chunk_base, "content": full_text}]

    prefix_for_piece = prefix or "\n".join(heading_path).strip()
    piece_max = max(500, max_chars - len(prefix_for_piece) - 2)
    pieces = paragraph_window_chunking(body, max_chars=piece_max)
    chunks = []
    for piece in pieces:
        piece_text = piece["content"].strip()
        if prefix_for_piece and not piece_text.startswith(prefix_for_piece):
            piece_text = f"{prefix_for_piece}\n\n{piece_text}".strip()
        chunks.append({**chunk_base, "content": piece_text})
    return chunks


def publication_section_chunking(text, max_chars=1700):
    """Chunk IRC publication PDFs by report/plan section headings.

    Annual reports and strategic plans are numerically structured. A generic
    paragraph window can mix the tail of one table with the heading and first
    facts of the next section, which is exactly how the 2025 "៣៩ companies"
    fact became fragile. This scanner starts new chunks at publication
    headings and repeats the active heading path inside every long split.
    """
    chunks = []
    current = []
    heading_stack = []
    last_page_marker = None

    def flush_current():
        nonlocal current
        if not current:
            return
        chunks.extend(split_publication_section(current, heading_stack, max_chars))
        current = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            if current and current[-1] != "":
                current.append("")
            continue

        clean_line = clean_publication_heading_title(line)
        if PUBLICATION_PAGE_MARKER_RE.match(clean_line):
            last_page_marker = clean_line
            if current:
                current.append(clean_line)
            continue

        if is_publication_heading(clean_line):
            flush_current()
            level = publication_heading_level(clean_line)
            heading_stack = heading_stack[: max(0, level - 1)]
            heading_stack.append(clean_line)
            current = []
            if last_page_marker:
                current.append(last_page_marker)
            current.extend(heading_stack)
            continue

        current.append(line)

    flush_current()

    if chunks:
        return chunks
    return paragraph_window_chunking(text, max_chars=max_chars)


def chunk_document(text, category, config):

    if category == "guidance":
        return guidance_chunking(text)

    if category in LEGAL_CATEGORIES:
        return legal_unit_chunking(text)

    if category == "publication":
        return publication_section_chunking(text, max_chars=config.get("max_chars", 1700))

    if category in WEBSITE_CHUNKING_CATEGORIES:
        return paragraph_window_chunking(text, max_chars=config.get("max_chars", 1400))

    return []


for input_path in json_files:
    relative_path = input_path.relative_to(input_root)
    output_filename = input_path.stem + "_chunks.json"
    output_path = output_root / relative_path.parent / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nChunking: {relative_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_text = data.get("cleaned_text", "").strip()

    if not cleaned_text:
        print(f"No cleaned text found: {relative_path}")
        skipped_files += 1
        continue

    category = relative_path.parts[0]

    if category == "publication" and data["document_id"] in SKIP_PUBLICATION_DOCUMENT_IDS:
        print(f"Skipping English publication duplicate: {relative_path}")
        skipped_files += 1
        continue

    config = CATEGORY_CONFIG.get(category)

    if not config:
        print(f"Unsupported category: {category}")
        skipped_files += 1
        continue

    chunking_method = config["chunking_method"]
    split_chunks = chunk_document(cleaned_text, category, config)

    # Merge document-level metadata into every chunk
    doc_meta = DOCUMENT_METADATA.get(data["document_id"], {})
    release_date = doc_meta.get("release_date")

    chunks = []

    for i, chunk_data in enumerate(split_chunks):
        if category in LEGAL_CATEGORIES:
            text = chunk_data["content"]
            metadata = {
                "document_id"       : data["document_id"],
                "document_name"     : doc_meta.get("document_name"),
                "document_type"     : doc_meta.get("document_type"),
                "release_date"      : release_date,
                "category"          : category,
                "chapter"           : chunk_data.get("chapter"),
                "chapter_title"     : chunk_data.get("chapter_title"),
                "section"           : chunk_data.get("section"),
                "section_title"     : chunk_data.get("section_title"),
                "legal_unit"        : chunk_data.get("legal_unit"),
                "chunk_size"        : len(text),
                "chunking_method"   : chunking_method,
            }

        elif category == "guidance":
            text = chunk_data["content"]
            metadata = {
                "document_id"       : data["document_id"],
                "document_name"     : doc_meta.get("document_name"),
                "document_type"     : doc_meta.get("document_type"),
                "release_date"      : release_date,
                "category"          : category,
                "heading_level"     : chunk_data.get("heading_level"),
                "heading_id"        : chunk_data.get("heading_id"),
                "heading_title"     : chunk_data.get("heading_title"),
                "parent_h1"         : chunk_data.get("parent_h1"),
                "chunk_size"        : len(text),
                "chunking_method"   : chunking_method,
            }

        else:
            text = chunk_data["content"]
            metadata = {
                "document_id"       : data["document_id"],
                "document_name"     : doc_meta.get("document_name"),
                "document_type"     : doc_meta.get("document_type", category),
                "release_date"      : release_date,
                "category"          : category,
                "source_url"        : data.get("source_url") or doc_meta.get("source_url"),
                "source_section"    : doc_meta.get("source_section", category),
                "chunk_size"        : len(text),
                "chunking_method"   : chunking_method,
                "heading_level"     : chunk_data.get("heading_level"),
                "heading_title"     : chunk_data.get("heading_title"),
                "heading_path"      : chunk_data.get("heading_path"),
                "section"           : chunk_data.get("section"),
                "section_title"     : chunk_data.get("section_title"),
            }

        # Final chunk structure stored for embedding and retrieval
        chunk = {
            "chunk_id": f"{data['document_id']}_chunk_{i}",
            "content": text,
            "metadata": metadata,
        }

        chunks.append(chunk)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    processed_files += 1
    total_chunks += len(chunks)

    print(f"Created {len(chunks)} chunks.")

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Processed Files : {processed_files:,}")
print(f"Skipped Files   : {skipped_files:,}")
print(f"Total Chunks    : {total_chunks:,}")
print(f"Total Runtime   : {elapsed_time:.2f} seconds")
