import json
import re
from pathlib import Path
import time

# Root input folder
input_root = Path(
    "data/extracted_text/raw"
)

# Root output folder
output_root = Path(
    "data/extracted_text/cleaned"
)

output_root.mkdir(
    parents=True,
    exist_ok=True
)

# Find ALL JSON files recursively
json_files = sorted(
    input_root.rglob("*.json")
)

# Counters
processed_files = 0
skipped_files = 0

# Runtime
start_time = time.time()

for input_path in json_files:

    relative_path = input_path.relative_to(
        input_root
    )

    # Preserve folder structure
    output_path = output_root / relative_path

    # Create parent folders
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # File name only
    file_name = input_path.name

    # Skip existing cleaned file
    if output_path.exists():

        print(
            f"Skipping already cleaned: "
            f"{relative_path}"
        )

        skipped_files += 1
        continue

    print(f"\nCleaning: {relative_path}")

    # Load raw JSON
    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    raw_text = data["raw_text"]

    # -----------------------------
    # TEXT CLEANING
    # -----------------------------

    cleaned_text = raw_text

    cleaned_text = re.sub(
    r"(\S+)-\n(\S+)",
    r"\1\2",
    cleaned_text
    )

    # Remove page numbers
    cleaned_text = re.sub(
        r"ទំព័រទី\s*[0-9០១២៣៤៥៦៧៨៩]+\s*នៃ\s*[0-9០១២៣៤៥៦៧៨៩]+",
        "",
        cleaned_text
    )

    # Footer Removal
    footer_patterns = [
        r"^.*Building\s*#.*$",
        r"^.*www\.irc\.gov\.kh.*$",
        r"^.*@irc\.gov\.kh.*$",
        r"^071\s*\d+.*$",
        r"^អគារលេខ.*$",
        r"^Khan Russey Keo.*$",
    ]

    for pattern in footer_patterns:

        cleaned_text = re.sub(
            pattern,
            "",
            cleaned_text,
            flags=re.MULTILINE
        )

    # OCR spacing normalization
    cleaned_text = re.sub(
        r"\.\s+",
        ".",
        cleaned_text
    )

    # Remove URLs
    cleaned_text = re.sub(
        r"www\.[^\s]+",
        "",
        cleaned_text
    )

    # Remove email addresses
    cleaned_text = re.sub(
        r"[\w\.-]+@[\w\.-]+",
        "",
        cleaned_text
    )

    # Remove phone numbers
    cleaned_text = re.sub(
        r"\b\d{3}\s\d{3}\s\d{4}\b",
        "",
        cleaned_text
    )

    # Remove markdown artifacts
    cleaned_text = cleaned_text.replace(
        "**",
        ""
    )

    # Normalize excessive line breaks
    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text
    )

    # Remove excessive spaces/tabs
    cleaned_text = re.sub(
        r"[ \t]+",
        " ",
        cleaned_text
    )

    # Remove trailing spaces
    cleaned_text = re.sub(
        r" +\n",
        "\n",
        cleaned_text
    )

    # Strip outer whitespace
    cleaned_text = cleaned_text.strip()

    # -----------------------------
    # SAVE CLEANED JSON
    # -----------------------------

    output_data = {
        "document_id": data["document_id"],
        "page": data["page"],
        "cleaned_text": cleaned_text
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

    print(f"Completed: {relative_path}")

# Total runtime
elapsed_time = (
    time.time() - start_time
)

print("\n========== SUMMARY ==========")

print(
    f"Processed Files : "
    f"{processed_files}"
)

print(
    f"Skipped Files   : "
    f"{skipped_files}"
)

print(
    f"Total Runtime   : "
    f"{elapsed_time:.2f} seconds"
)

print(
    "Text cleaning completed successfully."
)