import json
import re
from pathlib import Path
import time

input_root = Path(
    "data/05_merged_documents"
)

output_root = Path(
    "data/06_cleaned_documents"
)

output_root.mkdir(
    parents=True,
    exist_ok=True
)

json_files = sorted(
    input_root.rglob("*.json")
)

processed_files = 0
skipped_files = 0

start_time = time.time()


# =====================================================
# LAW DOCUMENT CLEANING
# =====================================================

def clean_law_document(text):

    chapter_match = re.search(
        r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+|Chapter\s+[0-9]+)",
        text,
        flags=re.IGNORECASE
    )

    article_match = re.search(
        r"(មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+|Article\s+[0-9]+)",
        text,
        flags=re.IGNORECASE
    )

    if chapter_match:

        text = text[
            chapter_match.start():
        ]

    elif article_match:

        text = text[
            article_match.start():
        ]

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


def clean_sub_decree_document(text):

    chapter_match = re.search(
        r"ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+",
        text
    )

    article_match = re.search(
        r"មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+",
        text
    )

    if chapter_match:

        text = text[
            chapter_match.start():
        ]

    elif article_match:

        text = text[
            article_match.start():
        ]

    else:

      start_match = re.search(
          r"HEREBY\s+DECIDES",
          text,
          flags=re.IGNORECASE
      )

      if start_match:

          text = text[
              start_match.end():
          ]

    signature_markers = [
        r"ថ្ងៃ.*?ព\.ស\.[0-9០១២៣៤៥៦៧៨៩]+",
        r"ធ្វើនៅរាជធានីភ្នំពេញ",
        r"នាយករដ្ឋមន្ត្រី",
        r"បានយកសេចក្តីគោរព",
        r"កន្លែងទទួល",

        r"Phnom Penh,\s*\d+.*?\d{4}",
        r"Have Proposed Signature to",
        r"Royal Gazette",
        r"Chronological documents",
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

    appendix_match = re.search(
        r"^\s*ឧបសម្ព័ន្ធទី\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$",
        text,
        flags=re.MULTILINE
    )

    if appendix_match:

        positions.append(
            appendix_match.start()
        )

    if positions:

        text = text[:min(positions)]

    return text.strip()


# =====================================================
# PROCESS FILES
# =====================================================

for input_path in json_files:

    relative_path = (
        input_path.relative_to(
            input_root
        )
    )

    output_path = (
        output_root
        / relative_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nCleaning document: "
        f"{relative_path}"
    )

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        text = data.get(
            "merged_text",
            ""
        )

        category = data.get(
            "category",
            ""
        )

        if category == "laws":

            text = clean_law_document(
                text
            )

        elif category == "sub_decrees":

            text = clean_sub_decree_document(
                text
            )

        output_data = {
            "document_id": data[
                "document_id"
            ],
            "category": category,
            "total_pages": data.get(
                "total_pages",
                0
            ),
            "cleaned_text": text
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

        processed_files += 1

        print(
            f"Completed: "
            f"{relative_path}"
        )

    except Exception as e:

        skipped_files += 1

        print(
            f"Failed: "
            f"{relative_path}"
        )

        print(e)


# =====================================================
# SUMMARY
# =====================================================

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
    f"Runtime         : "
    f"{elapsed_time:.2f} seconds"
)

print(
    "Document cleaning completed successfully."
)