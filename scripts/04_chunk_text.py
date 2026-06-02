import json
from pathlib import Path
import time

input_root = Path("data/extracted_text/cleaned")
output_root = Path("data/chunked_documents")

output_root.mkdir(
    parents=True,
    exist_ok=True
)

CHUNK_SIZE = 300

json_files = sorted(
    input_root.rglob("*.json")
)

processed_files = 0
skipped_files = 0
total_chunks = 0

start_time = time.time()


def recursive_split_text(text, chunk_size=300):

    separators = [
        "\n\n",
        "\n",
        "។",
        ".",
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

    split_chunks = (
        recursive_split_text(
            cleaned_text,
            CHUNK_SIZE
        )
    )

    chunks = []

    for i, text in enumerate(
        split_chunks
    ):

        chunk = {
            "chunk_id": i,
            "content": text,
            "metadata": {
                "chunk_id": i,
                "document_id": data.get(
                    "document_id"
                ),
                "page": data.get(
                    "page"
                ),
                "source_doc": data.get(
                    "document_id"
                ),
                "chunk_size": len(
                    text
                ),
                "chunking_method":
                    "recursive",
                "tables_converted":
                    False
            },
            "chunk_size": len(text)
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