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
        "chunk_size": 400,
        "min_chunk_size": 120,
        "chunking_method": "guidance_section",
        "merge_small_chunks": True
    },
    "prakas": {
        "chunk_size": 400,
        "min_chunk_size": 120,
        "chunking_method": "recursive",
        "merge_small_chunks": True
    },
    "sub_decrees": {
        "chunk_size": 400,
        "min_chunk_size": 120,
        "chunking_method": "recursive",
        "merge_small_chunks": True
    },
    "laws": {
        "chunk_size": None,
        "min_chunk_size": 0,
        "chunking_method": "law_article",
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

def guidance_chunking(
    text,
    chunk_size
):

    text = text.replace(
        "**",
        ""
    )

    section_pattern = (
        r"(?m)(?=^[០១២៣៤៥៦៧៨៩]+"
        r"(?:\.[០១២៣៤៥៦៧៨៩]+)*"
        r"[\.\)]?\s+)"
    )

    sections = re.split(
        section_pattern,
        text
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if len(section) <= chunk_size:

            chunks.append(section)

        else:

            chunks.extend(
                recursive_split_text(
                    section,
                    chunk_size
                )
            )

    return chunks

def law_chunking(
    text,
    chunk_size=None
):

    chapter_pattern = (
        r"(?="
        r"(?:"
        r"ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+"
        r"|"
        r"Chapter\s+[0-9]+"
        r")"
        r")"
    )

    article_pattern = (
        r"(?="
        r"(?:"
        r"មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+"
        r"|"
        r"Article\s*[0-9]+"
        r")"
        r")"
    )

    chapter_sections = re.split(
        chapter_pattern,
        text,
        flags=re.IGNORECASE
    )

    chunks = []

    for chapter_text in chapter_sections:

        chapter_text = chapter_text.strip()

        if not chapter_text:
            continue

        chapter_match = re.search(
            r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+|Chapter\s+[0-9]+)",
            chapter_text,
            flags=re.IGNORECASE
        )

        chapter_name = None

        if chapter_match:

            chapter_name = (
                chapter_match.group(1)
            )

        articles = re.split(
            article_pattern,
            chapter_text,
            flags=re.IGNORECASE
        )

        for article in articles:

            article = article.strip()

            if not article:
                continue

            article_match = re.search(
                r"(មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+|Article\s*[0-9]+)",
                article,
                flags=re.IGNORECASE
            )

            if not article_match:
                continue

            article_name = (
                article_match.group(1)
            )

            chunks.append(
                {
                    "content": article,
                    "chapter": chapter_name,
                    "article": article_name
                }
            )

    return chunks


def recursive_split_text(
    text,
    chunk_size
):

    separators = [
        "\n\n",
        "។",
        "\n",
        " ",
        ""
    ]

    return split_with_separators(
        text,
        separators,
        chunk_size
    )

def split_with_separators(
    text,
    separators,
    chunk_size
):

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]

    if separator == "":
        return [
            text[i:i + chunk_size]
            for i in range(
                0,
                len(text),
                chunk_size
            )
        ]

    pieces = text.split(separator)

    chunks = []
    current_chunk = ""

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        if separator in ["។", "."]:
            piece = piece + separator

        if current_chunk:

            if separator in ["\n\n", "\n"]:
                possible_chunk = (
                    current_chunk
                    + separator
                    + piece
                )
            else:
                possible_chunk = (
                    current_chunk
                    + " "
                    + piece
                )

        else:
            possible_chunk = piece

        if len(possible_chunk) <= chunk_size:

            current_chunk = possible_chunk

        else:

            if current_chunk:
                chunks.append(current_chunk)

            if len(piece) > chunk_size:

                smaller_chunks = (
                    split_with_separators(
                        piece,
                        separators[1:],
                        chunk_size
                    )
                )

                chunks.extend(
                    smaller_chunks
                )

                current_chunk = ""

            else:
                current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

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

    elif category == "laws":

        return law_chunking(
            text,
            chunk_size
        )

    return recursive_split_text(
        text,
        chunk_size
    )


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
            "merged_text",
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

        if category == "laws":

            text = chunk_data["content"]

            metadata = {
                "document_id": data.get(
                    "document_id"
                ),
                "category": category,
                "chapter": chunk_data.get(
                    "chapter"
                ),
                "article": chunk_data.get(
                    "article"
                ),
                "chunk_size": len(text),
                "chunking_method": chunking_method
            }

        else:

            text = chunk_data

            metadata = {
                "document_id": data.get(
                    "document_id"
                ),
                "category": category,
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
    "Recursive chunking completed successfully."
)