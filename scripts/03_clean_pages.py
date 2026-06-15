import json
import re
from pathlib import Path
import time

# Root input folder
input_root = Path(
    "data/03_raw_pages"
)

# Root output folder
output_root = Path(
    "data/04_cleaned_pages"
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


def general_cleaning(text):

    # Add hyphenated line break fix
    text = re.sub(
        r"(\S+)-\n(\S+)",
        r"\1\2",
        text
    )

    # Remove page numbers
    text = re.sub(
        r"ទំព័រទី\s*[0-9០១២៣៤៥៦៧៨៩]+\s*នៃ\s*[0-9០១២៣៤៥៦៧៨៩]+",
        "",
        text
    )

    footer_patterns = [
        r"^.*Building\s*#.*$",
        r"^.*www\.irc\.gov\.kh.*$",
        r"^.*@irc\.gov\.kh.*$",
        r"^071\s*\d+.*$",
        r"^អគារលេខ.*$",
        r"^Khan Russey Keo.*$",
    ]

    for pattern in footer_patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.MULTILINE
        )

    text = re.sub(
        r"\.\s+",
        ".",
        text
    )

    text = re.sub(
        r"www\.[^\s]+",
        "",
        text
    )

    text = re.sub(
        r"[\w\.-]+@[\w\.-]+",
        "",
        text
    )

    text = re.sub(
        r"\b\d{3}\s\d{3}\s\d{4}\b",
        "",
        text
    )

    text = text.replace(
        "**",
        ""
    )

    return text


def clean_laws(text):

    text = re.sub(
        r"^\d+/\d+$",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^[0-9០១២៣៤៥៦៧៨៩]+/[0-9០១២៣៤៥៦៧៨៩]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"ROYAUME DU CAMBODGE",
        "",
        text
    )

    text = re.sub(
        r"LE MINISTRE DU PALAIS ROYAL",
        "",
        text
    )

    text = re.sub(
        r"\(Seal text:.*?\)",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"(?m)^មាត្រា\s*([0-9០១២៣៤៥៦៧៨៩]+)\s*(?:\.\.|\.|-)+\s*",
        r"មាត្រា \1 - ",
        text
    )

    # Add space after numbered legal list markers
    text = re.sub(
        r"(^|\n)([0-9០១២៣៤៥៦៧៨៩]+)[\.\-]\s*",
        r"\1\2. ",
        text
    )

    # Add space after Khmer letter list markers
    text = re.sub(
        r"(^|\n)([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])[\.\-]\s*",
        r"\1\2. ",
        text
    )

    return text


def clean_guidance(text):

    return text


def clean_prakas(text):

    # Remove page markers like ទំព័រ ២ នៃទំព័រ ៤
    text = re.sub(
        r"^\s*ទំព័រ\s*[0-9០១២៣៤៥៦៧៨៩]+\s*នៃ\s*(?:ទំព័រ\s*)?[0-9០១២៣៤៥៦៧៨៩]+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^\s*ទំព័រទី\s*[0-9០១២៣៤៥៦៧៨៩]+\s*នៃ\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove OCR markdown headers
    text = re.sub(
        r"(?m)^#\s*",
        "",
        text
    )

    # Fix article ranges
    text = re.sub(
        r"\n(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\s*-\s*(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)",
        r" \1 ដល់ \2",
        text
    )

    # Fix article references split after "ក្នុង"
    text = re.sub(
        r"ក្នុង\s*\n\s*(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\s*-\s*នៃ",
        r"ក្នុង \1 នៃ",
        text
    )

    # Fix referenced articles split across lines
    text = re.sub(
        r"(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\n(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\s*-",
        r"\1 \2",
        text
    )

    # Remove appendix page numbers
    text = re.sub(
        r"^\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove escaped bullets
    text = text.replace(
        "\\-",
        "-"
    )

    # Normalize article formatting
    text = re.sub(
        r"(?m)^ប្រការ\s*([0-9០១២៣៤៥៦៧៨៩]+)\s*(?:\.\-|\. -|\.|\.{2}|-|.-)+\s*",
        r"ប្រការ \1 - ",
        text
    )

    # Fix article references broken across lines
    text = re.sub(
        r"(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\n(ប្រការ\s*[0-9០១២៣៤៥៦៧៨៩]+)\s*-\s*និង",
        r"\1 \2 និង",
        text
    )

    # Normalize chapter formatting
    text = re.sub(
        r"ជំពូកទី\s*([0-9០១២៣៤៥៦៧៨៩]+)",
        r"ជំពូកទី \1",
        text
    )

    # Normalize section formatting
    text = re.sub(
        r"ផ្នែកទី\s*([0-9០១២៣៤៥៦៧៨៩]+)",
        r"ផ្នែកទី \1",
        text
    )

    # Add space after numbered list markers
    text = re.sub(
        r"(^|\n)([0-9០១២៣៤៥៦៧៨៩]+)[\.\-]\s*",
        r"\1\2. ",
        text
    )

    # Add space after Khmer letter list markers
    text = re.sub(
        r"(^|\n)([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])[\.\-]\s*",
        r"\1\2. ",
        text
    )

    return text


def clean_sub_decrees(text):

    # Remove page markers like 1/16
    text = re.sub(
        r"^[0-9០១២៣៤៥៦៧៨៩]+/[0-9០១២៣៤៥៦៧៨៩]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove page markers like ទំព័រ ២
    text = re.sub(
        r"^\s*ទំព័រ\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove page markers like - ១ - or - ២ -
    text = re.sub(
        r"\s*-\s*[0-9០១២៣៤៥៦៧៨៩]+\s*-\s*",
        "\n",
        text
    )

    # Remove backlash infront of bullet points
    text = text.replace(
        "\\-",
        "-"
    )

    # Normalize article formatting
    text = re.sub(
        r"(?m)^មាត្រា\s*([0-9០១២៣៤៥៦៧៨៩]+)\s*(?:\.\.|\.|-|\. -|.-)*\s*",
        r"មាត្រា \1 - ",
        text
    )

    # Normalize chapter formatting
    text = re.sub(
        r"ជំពូកទី\s*([0-9០១២៣៤៥៦៧៨៩]+)",
        r"ជំពូកទី \1",
        text
    )

    # Normalize section formatting
    text = re.sub(
        r"ផ្នែកទី\s*([0-9០១២៣៤៥៦៧៨៩]+)",
        r"ផ្នែកទី \1",
        text
    )

    # Fix article-reference ranges wrongly looking like article headers
    text = re.sub(
        r"\n(មាត្រា\s*[0-9០១២៣៤៥៦៧៨៩]+)\s*-\s*(មាត្រា\s*[0-9០១២៣៤៥៦៧៨៩]+)",
        r" \1 ដល់ \2",
        text
    )

    # Add space after numbered legal list markers
    text = re.sub(
        r"(^|\n)([0-9០១២៣៤៥៦៧៨៩]+)[\.\-]\s*",
        r"\1\2. ",
        text
    )

    # Add space after Khmer letter list markers
    text = re.sub(
        r"(^|\n)([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])[\.\-]\s*",
        r"\1\2. ",
        text
    )

    return text


def final_normalization(text):

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r" +\n",
        "\n",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()

    return text


for input_path in json_files:

    relative_path = input_path.relative_to(
        input_root
    )

    output_path = output_root / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():

        print(
            f"Skipping already cleaned: "
            f"{relative_path}"
        )

        skipped_files += 1
        continue

    print(f"\nCleaning: {relative_path}")

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    raw_text = data["raw_text"]

    category = relative_path.parts[0]

    cleaned_text = general_cleaning(
        raw_text
    )

    if category == "laws":

        cleaned_text = clean_laws(
            cleaned_text
        )

    elif category == "guidance":

        cleaned_text = clean_guidance(
            cleaned_text
        )

    elif category == "prakas":

        cleaned_text = clean_prakas(
            cleaned_text
        )

    elif category == "sub_decrees":

        cleaned_text = clean_sub_decrees(
            cleaned_text
        )

    cleaned_text = final_normalization(
        cleaned_text
    )

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