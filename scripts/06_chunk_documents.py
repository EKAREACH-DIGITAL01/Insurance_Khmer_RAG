from pathlib import Path
import re
import json
import time

input_root = Path("data/06_cleaned_documents")
output_root = Path("data/07_chunked_documents")

output_root.mkdir(parents=True, exist_ok=True)

with open("data/document_metadata.json", "r", encoding="utf-8") as f:
    DOCUMENT_METADATA = json.load(f)

CATEGORY_CONFIG = {
    "guidance": {"chunking_method": "guidance_section"},
    "laws": {"chunking_method": "legal_article"},
    "prakas": {"chunking_method": "legal_article"},
    "sub_decrees": {"chunking_method": "legal_article"},
}

LEGAL_CATEGORIES = {"laws", "prakas", "sub_decrees"}

json_files = sorted(input_root.rglob("*.json"))

processed_files = 0
skipped_files = 0
total_chunks = 0

start_time = time.time()


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

            # intro before first heading
            chunks.append(
                {
                    "content": section.strip(),
                    "heading_level": 0,
                    "heading_id": None,
                    "heading_title": None,
                    "parent_h1": None,
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
                "content": content,
                "heading_level": level,
                "heading_id": heading_id,
                "heading_title": heading_title,
                "parent_h1": parent_h1,
            }
        )

    return chunks


def legal_article_chunking(text):

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

        chapter_match = re.search(
            r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)", chapter_text, flags=re.MULTILINE
        )

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

        section_matches = list(
            re.finditer(
                r"^(ផ្នែកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)", chapter_text, flags=re.MULTILINE
            )
        )

        articles = re.split(article_pattern, chapter_text, flags=re.MULTILINE)

        for article in articles:

            article = article.strip()

            if not article:
                continue

            article_start = chapter_text.find(article)

            section_name = None
            section_title = None

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
                    "content": article,
                    "chapter": chapter_name,
                    "chapter_title": chapter_title,
                    "section": section_name,
                    "section_title": section_title,
                    "legal_unit": article_name,
                }
            )

    return chunks


def chunk_document(text, category):

    if category == "guidance":

        return guidance_chunking(text)

    elif category in LEGAL_CATEGORIES:

        return legal_article_chunking(text)

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

    config = CATEGORY_CONFIG[category]

    chunking_method = config["chunking_method"]

    split_chunks = chunk_document(cleaned_text, category)

    chunks = []

    for i, chunk_data in enumerate(split_chunks):

        doc_meta = DOCUMENT_METADATA.get(data["document_id"], {})
        release_date = doc_meta.get("release_date")

        if category in LEGAL_CATEGORIES:

            text = chunk_data["content"]

            metadata = {
                "document_id": data["document_id"],
                "document_name": doc_meta.get("document_name"),
                "document_type": doc_meta.get("document_type"),
                "release_date": release_date,
                "category": category,
                "chapter": chunk_data.get("chapter"),
                "chapter_title": chunk_data.get("chapter_title"),
                "section": chunk_data.get("section"),
                "section_title": chunk_data.get("section_title"),
                "legal_unit": chunk_data.get("legal_unit"),
                "chunk_size": len(text),
                "chunking_method": chunking_method,
            }

        else:

            text = chunk_data["content"]

            metadata = {
                "document_id": data["document_id"],
                "document_name": doc_meta.get("document_name"),
                "document_type": doc_meta.get("document_type"),
                "release_date": release_date,
                "category": category,
                "heading_level": chunk_data.get("heading_level"),
                "heading_id": chunk_data.get("heading_id"),
                "heading_title": chunk_data.get("heading_title"),
                "parent_h1": chunk_data.get("parent_h1"),
                "chunk_size": len(text),
                "chunking_method": chunking_method,
            }

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

print(f"Processed Files : " f"{processed_files}")

print(f"Skipped Files   : " f"{skipped_files}")

print(f"Total Chunks    : " f"{total_chunks}")

print(f"Total Runtime   : " f"{elapsed_time:.2f} seconds")

print("Document chunking completed successfully.")
