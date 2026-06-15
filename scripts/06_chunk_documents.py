import re
import json
from pathlib import Path
import time

input_root = Path("data/06_cleaned_documents")
output_root = Path("data/07_chunked_documents")

output_root.mkdir(
    parents=True,
    exist_ok=True
)

CATEGORY_CONFIG = {
    "guidance": {
        "chunk_size": None,
        "min_chunk_size": 0,
        "chunking_method": "guidance_section",
        "merge_small_chunks": False
    },
     "laws": {
        "chunk_size": None,
        "min_chunk_size": 0,
        "chunking_method": "legal_article",
        "merge_small_chunks": False
    },
    "prakas": {
        "chunk_size": None,
        "min_chunk_size": 0,
        "chunking_method": "legal_article",
        "merge_small_chunks": False
    },
    "sub_decrees": {
        "chunk_size": None,
        "min_chunk_size": 0,
        "chunking_method": "legal_article",
        "merge_small_chunks": False
    }
}

json_files = sorted(
    input_root.rglob("*.json")
)

processed_files = 0
skipped_files = 0
total_chunks = 0

start_time = time.time()

def extract_guidance_section_title(text):

    lines = text.strip().splitlines()

    if not lines:
        return None

    first_line = lines[0].strip()

    if re.match(
        r"^(?:"
        r"[កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ][\.\)]"
        r"|"
        r"[០១២៣៤៥៦៧៨៩]+[\.\)]"
        r")",
        first_line
    ):
        return first_line

    return None

def guidance_chunking(text, chunk_size):

    text = text.replace("**", "")

    heading_pattern = (
        r"(?m)(?=^"
        r"(?:"
        r"[កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ][\.\)]"
        r"|"
        r"[០១២៣៤៥៦៧៨៩]+[\.\)]"
        r")"
        r"\s*"
        r")"
    )

    sections = re.split(
        heading_pattern,
        text
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        section_title = extract_guidance_section_title(
            section
        )

        chunks.append(
            {
                "content": section,
                "section_title": section_title
            }
        )

    return chunks

def legal_article_chunking(
    text,
    chunk_size=None
):

    chapter_pattern = (
        r"(?="
        r"(?:"
        r"^ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+"
        r"|"
        r"^Chapter\s+[0-9]+"
        r")"
        r")"
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

    chapter_sections = re.split(
        chapter_pattern,
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for chapter_text in chapter_sections:

        chapter_text = chapter_text.strip()

        if not chapter_text:
            continue

        chapter_match = re.search(
            r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)",
            chapter_text
        )

        chapter_name = None
        section_name = None

        if chapter_match:

            chapter_name = (
                chapter_match.group(1)
            )

        section_matches = list(
            re.finditer(
                r"^(ផ្នែកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)",
                chapter_text,
                flags=re.MULTILINE
            )
        )

        articles = re.split(
            article_pattern,
            chapter_text,
            flags=re.MULTILINE
        )

        for article in articles:

            article = article.strip()

            if not article:
                continue

            article_start = chapter_text.find(article)

            section_name = None

            for section_match in section_matches:

                if section_match.start() < article_start:

                    section_name = section_match.group(1)

                else:

                    break

            article_match = re.search(
                r"^(មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+"
                r"|"
                r"ប្រការ\s+[0-9០១២៣៤៥៦៧៨៩]+)",
                article,
                flags=re.MULTILINE
            )

            if not article_match:
                continue

            article_name = (
                article_match.group(1)
            )

            # Remove trailing next section header
            article = re.sub(
                r"\n\s*"
                r"(ផ្នែកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)"
                r"\s*\n.*$",
                "",
                article,
                flags=re.DOTALL
            ).strip()

            chunks.append(
                {
                    "content": article,
                    "chapter": chapter_name,
                    "section": section_name,
                    "legal_unit": article_name
                }
            )

    return chunks

def merge_small_chunks(
    chunks,
    min_size
):

    if not chunks:
        return chunks

    merged = []

    for chunk in chunks:

        if (
            merged
            and len(chunk) < min_size
        ):

            merged[-1] += "\n" + chunk

        else:

            merged.append(chunk)

    return merged

def chunk_document(
    text,
    category,
    chunk_size
):

    if category == "guidance":

        return guidance_chunking(
            text,
            chunk_size
        )

    elif category in ["laws", "sub_decrees", "prakas"]:

        return legal_article_chunking(
            text,
            chunk_size
        )

    return []


for input_path in json_files:

    relative_path = (
        input_path.relative_to(
            input_root
        )
    )

    output_filename = (
        input_path.stem
        + "_chunks.json"
    )

    output_path = (
        output_root
        / relative_path.parent
        / output_filename
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nChunking: "
        f"{relative_path}"
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    cleaned_text = (
        data.get(
            "cleaned_text",
            ""
        ).strip()
    )

    if not cleaned_text:

        print(
            f"No cleaned text found: "
            f"{relative_path}"
        )

        skipped_files += 1
        continue

    category = (
        relative_path.parts[0]
    )

    config = CATEGORY_CONFIG[
        category
    ]

    chunk_size = config["chunk_size"]
    min_chunk_size = config["min_chunk_size"]

    chunking_method = config[
        "chunking_method"
    ]

    merge_small = config[
        "merge_small_chunks"
    ]

    split_chunks = (
        chunk_document(
            cleaned_text,
            category,
            chunk_size
        )
    )

    if merge_small:

        split_chunks = merge_small_chunks(
            split_chunks,
            min_chunk_size
        )

    chunks = []

    for i, chunk_data in enumerate(
        split_chunks
    ):

        if category in ["laws", "sub_decrees", "prakas"]:

            text = chunk_data["content"]

            metadata = {
                "document_id": data.get(
                    "document_id"
                ),
                "category": category,
                "chapter": chunk_data.get(
                    "chapter"
                ),
                "section": chunk_data.get(
                    "section"
                ),
                "legal_unit": chunk_data.get(
                    "legal_unit"
                ),
                "chunk_size": len(text),
                "chunking_method": chunking_method
            }

        else:

            text = chunk_data["content"]

            metadata = {
                "document_id": data.get(
                    "document_id"
                ),
                "category": category,
                "section_title": chunk_data.get(
                    "section_title"
                ),
                "chunk_size": len(text),
                "chunking_method": chunking_method
            }

        chunk = {
            "chunk_id": (
                f"{data['document_id']}"
                f"_chunk_{i}"
            ),
            "content": text,
            "metadata": metadata
        }

        chunks.append(chunk)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    processed_files += 1
    total_chunks += len(
        chunks
    )

    print(
        f"Created "
        f"{len(chunks)} chunks."
    )

elapsed_time = (
    time.time()
    - start_time
)

print(
    "\n========== SUMMARY =========="
)

print(
    f"Processed Files : "
    f"{processed_files}"
)

print(
    f"Skipped Files   : "
    f"{skipped_files}"
)

print(
    f"Total Chunks    : "
    f"{total_chunks}"
)

print(
    f"Total Runtime   : "
    f"{elapsed_time:.2f} seconds"
)

print(
    "Document chunking completed successfully."
)