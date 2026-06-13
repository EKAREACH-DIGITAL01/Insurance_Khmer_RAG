import json
import re
from pathlib import Path
from collections import defaultdict
import time

input_root = Path(
    "data/extracted_text/cleaned"
)

output_root = Path(
    "data/merged_documents"
)

output_root.mkdir(
    parents=True,
    exist_ok=True
)

json_files = sorted(
    input_root.rglob("*.json")
)

start_time = time.time()

documents = defaultdict(list)

# -----------------------------
# LOAD ALL PAGES
# -----------------------------

for input_path in json_files:

    relative_path = input_path.relative_to(
        input_root
    )

    category = relative_path.parts[0]

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    document_id = data["document_id"]

    page = data["page"]

    cleaned_text = data.get(
        "cleaned_text",
        ""
    )

    documents[
        (category, document_id)
    ].append(
        {
            "page": page,
            "text": cleaned_text
        }
    )

# -----------------------------
# MERGE PAGES
# -----------------------------
def clean_law_merged_document(text):

    signature_markers = [
        r"ថ្ងៃ\s+.*?ព\.ស\.\d+",
        r"ធ្វើនៅព្រះបរមរាជវាំង",
        r"Royal Palace",
    ]

    positions = []

    for marker in signature_markers:

        match = re.search(
            marker,
            text,
            flags=re.IGNORECASE
        )

        if match:

            positions.append(
                match.start()
            )

    if positions:

        text = text[
            :min(positions)
        ]

    return text.strip()

merged_count = 0

for (
    category,
    document_id
), pages in documents.items():

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    merged_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    # Remove law preamble before first chapter/article
    if category == "laws":

        chapter_match = re.search(
            r"ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+",
            merged_text
        )

        article_match = re.search(
            r"មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+",
            merged_text
        )

        if chapter_match:

            merged_text = merged_text[
                chapter_match.start():
            ]

        elif article_match:

            merged_text = merged_text[
                article_match.start():
            ]

        merged_text = clean_law_merged_document(
            merged_text
        )

    output_folder = (
        output_root
        / category
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / f"{document_id}.json"
    )

    output_data = {
        "document_id": document_id,
        "category": category,
        "total_pages": len(
            pages
        ),
        "merged_text": merged_text
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    merged_count += 1

    print(
        f"Merged: "
        f"{category}/{document_id} "
        f"({len(pages)} pages)"
    )

# -----------------------------
# SUMMARY
# -----------------------------

elapsed_time = (
    time.time()
    - start_time
)

print(
    "\n========== SUMMARY =========="
)

print(
    f"Documents Merged : "
    f"{merged_count}"
)

print(
    f"Runtime : "
    f"{elapsed_time:.2f} seconds"
)

print(
    "Document merging completed successfully."
)