import json
import re
import time
from pathlib import Path

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES

input_root = Path("data/05_merged_documents")
output_root = Path("data/06_cleaned_documents")

output_root.mkdir(parents=True, exist_ok=True)

WEBSITE_CLEANING_CATEGORIES = set(WEBSITE_DERIVED_CATEGORIES) | {"website"}

# Find all JSONs recursively
json_files = sorted(input_root.rglob("*.json"))

processed_files = 0
skipped_files = 0

start_time = time.time()

# Maps abbreviation to its meaning
ABBREVIATION_MAP = {
    "ន.ធ.ក.": "និយ័តករធានារ៉ាប់រងកម្ពុជា",
    "អ.ស.ហ.": "អាជ្ញាធរសេវាហិរញ្ញវត្ថុមិនមែនធនាគារ",
    "ន.គ.ស.": "និយ័តករគណនេយ្យនិងសវនកម្ម",
    "ន.ប.ធ.": "និយ័តករបរធនបាលកិច្ច",
    "ន.ស.ស.": "និយ័តករសន្តិសុខសង្គម",
    "ន.អ.ប.": "និយ័តករអាជីវកម្មអចលនវត្ថុនិងបញ្ចាំ",
    "វ.គ.ស.ក.": "វិទ្យាស្ថានគណនេយ្យករជំនាញ និងសវនករកម្ពុជា",
    "ស.ធ.ក.": "សមាគមធានារ៉ាប់រងកម្ពុជា",
    "ន.ម.ក.": "និយ័តករមូលបត្រកម្ពុជា",
}

# Maps guidance documents to their heading hierarchy pattern
PATTERN_MAP = {
    "guidance_01": "A",
    "guidance_02": "B",
    "guidance_03": "C",
    "guidance_04": "D",
    "guidance_05": "E",
    "guidance_06": "F",
    "guidance_07": "A",
    "guidance_08": "G",
    "guidance_09": "A",
    "guidance_10": "A",
    "guidance_11": "J",
    "guidance_12": "A",
    "guidance_13": "C",
    "guidance_14": "H",
    "guidance_15": "B",
    "guidance_16": "I",
}


def clean_guidance_document(text):

    # Remove kingdom header
    text = re.sub(
        r"^\s*ព្រះរាជាណាចក្រកម្ពុជា\s*"
        r"\n\s*ជាតិ\s+សាសនា\s+ព្រះមហាក្សត្រ\s*"
        r"\n?(?:[-*—–]+\s*)?",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove regulator header
    text = re.sub(r"(?m)^\s*និយ័តករធានារ៉ាប់រងកម្ពុជា\s*$", "", text)

    # Remove date block before title
    text = re.sub(r"(?s)^.*?(?=\n\s*(?:គោលការណ៍ណែនាំ|សេចក្តីណែនាំ)\s*\n\s*ស្តីពី)", "", text)

    text = re.sub(r"(?m)^\s*INSURANCE REGULATOR OF CAMBODIA\s*$", "", text)

    # Remove document number line
    text = re.sub(r"(?m)^\s*លេខ.*$", "", text)

    # Remove signature section ONLY near document end
    signature_patterns = [
        r"\n\s*ចម្លងជូន[:៖]",
        r"\n\s*អគ្គនាយក\s*\n",
        r"\n\s*រាជធានីភ្នំពេញ\s*ថ្ងៃទី",
        r"\n\s*ថ្ងៃ\s*.*?ព\.ស\.",
    ]

    # Remove appendix and everything after it
    APPENDIX_MARKERS = [
        "ឧបសម្ព័ន្ធនៃគោលការណ៍ណែនាំលេខ",
        "ឧបសម្ព័ន្ធស្តីពី ដំណើរការនៃបរិស្ថានសាកល្បងបច្ចេកវិទ្យាធានារ៉ាប់រង",
    ]

    for marker in APPENDIX_MARKERS:
        pos = text.find(marker)

        if pos != -1:
            text = text[:pos]
            break

    cut_positions = []

    for pattern in signature_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            # Only cut if marker is in last 40% of document
            if match.start() > len(text) * 0.6:
                cut_positions.append(match.start())

    if cut_positions:
        text = text[: min(cut_positions)]

    return text.strip()


def normalize_guidance_headings(text, document_id):

    pattern = PATTERN_MAP.get(document_id)

    # Level 1: num. # Level 2: num.num.
    if pattern == "A":
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H2> \3", text)
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H1> \2", text)

    # Level 1: alphabet. # Level 2: num.
    elif pattern == "B":

        def replace_h1(match):
            title = match.group(2)
            return f"<H1> {title}"

        text = re.sub(r"(?m)^([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])\.\s*(.+)$", replace_h1, text)
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H2> \2", text)

    # Level 1: alphabet.
    elif pattern == "C":
        text = re.sub(r"(?m)^([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])\.\s*(.+)$", r"<H1> \2", text)

    # Level 1: alphabet. # Level 2: num. # Level 3: num.num.
    elif pattern == "D":
        text = re.sub(r"(?m)^([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])\.\s*(.+)$", r"<H1> \2", text)
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H3> \3", text)
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H2> \2", text)

    # Level 1: num. # Level 2: num.num. # Level 3: num.num.num. # Level 4: num.num.num.num.
    elif pattern == "E":
        text = re.sub(
            r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$",
            r"<H4> \5",
            text,
        )
        text = re.sub(
            r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$",
            r"<H3> \4",
            text,
        )
        text = re.sub(
            r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H2> \3", text
        )
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H1> \2", text)

    # Level 1: num)
    elif pattern == "F":
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\)\s*(.+)$", r"<H1> \2", text)

    # No heading structure (single-page document)
    elif pattern == "G":
        pass

    # Level 1: num.
    elif pattern == "H":
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H1> \2", text)

    # Level 1: num.-
    elif pattern == "I":
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.-\s*(.+)$", r"<H1> \2", text)

    # Level 1: num. # Level 2: alphabet.
    elif pattern == "J":
        text = re.sub(r"(?m)^([០១២៣៤៥៦៧៨៩]+)\.\s*(.+)$", r"<H1> \2", text)
        text = re.sub(r"(?m)^([កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ])\.\s*(.+)$", r"<H2> \2", text)

    text = rebuild_guidance_heading_ids(text)

    return text


# Rebuild sequential heading IDs after heading normalization
def rebuild_guidance_heading_ids(text):

    lines = text.splitlines()

    h1 = 0
    h2 = 0
    h3 = 0
    h4 = 0

    rebuilt_lines = []

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^<H([1-4])>\s*(.*)$", stripped)

        if not match:
            rebuilt_lines.append(line)
            continue

        level = int(match.group(1))
        title = match.group(2).strip()

        if level == 1:
            h1 += 1
            h2 = 0
            h3 = 0
            h4 = 0
            heading_id = f"{h1}."

        elif level == 2:
            h2 += 1
            h3 = 0
            h4 = 0
            heading_id = f"{h1}.{h2}."

        elif level == 3:
            h3 += 1
            h4 = 0
            heading_id = f"{h1}.{h2}.{h3}."

        elif level == 4:
            h4 += 1
            heading_id = f"{h1}.{h2}.{h3}.{h4}."

        rebuilt_lines.append(f"<H{level}> {heading_id} {title}".strip())

    return "\n".join(rebuilt_lines)


def clean_law_document(text):

    chapter_match = re.search(r"(ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+)", text, flags=re.IGNORECASE)
    article_match = re.search(r"(មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+)", text, flags=re.IGNORECASE)

    if chapter_match:
        text = text[chapter_match.start() :]

    elif article_match:
        text = text[article_match.start() :]

    signature_markers = [
        r"ថ្ងៃ\s+.*?ព\.ស\.\d+",
        r"ធ្វើនៅព្រះបរមរាជវាំង",
        r"Royal Palace",
    ]

    positions = []

    for marker in signature_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)

        if match:
            positions.append(match.start())

    if positions:
        text = text[: min(positions)]

    return text.strip()


def clean_sub_decree_document(text):

    chapter_match = re.search(r"ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+", text)
    article_match = re.search(r"មាត្រា\s+[0-9០១២៣៤៥៦៧៨៩]+", text)

    if chapter_match:
        text = text[chapter_match.start() :]

    elif article_match:
        text = text[article_match.start() :]

    signature_markers = [
        r"ថ្ងៃ.*?ព\.ស\.[0-9០១២៣៤៥៦៧៨៩]+",
        r"ធ្វើនៅរាជធានីភ្នំពេញ",
        r"នាយករដ្ឋមន្ត្រី",
        r"បានយកសេចក្តីគោរព",
        r"កន្លែងទទួល",
    ]

    positions = []

    for marker in signature_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)

        if match:
            positions.append(match.start())

    appendix_match = re.search(
        r"^\s*ឧបសម្ព័ន្ធទី\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$", text, flags=re.MULTILINE
    )

    if appendix_match:
        positions.append(appendix_match.start())

    if positions:
        text = text[: min(positions)]

    return text.strip()


def clean_prakas_document(text):

    # Remove page markers left after merge
    text = re.sub(r"(?m)^\s*ទំព័រ\s*[0-9០១២៣៤៥៦៧៨៩]+\s*$", "", text)

    # Remove markdown heading marks
    text = re.sub(r"(?m)^#+\s*", "", text)

    chapter_match = re.search(r"ជំពូកទី\s*[0-9០១២៣៤៥៦៧៨៩]+", text)
    article_match = re.search(r"ប្រការ\s+[0-9០១២៣៤៥៦៧៨៩]+", text)

    if chapter_match:
        text = text[chapter_match.start() :]

    elif article_match:
        text = text[article_match.start() :]

    signature_markers = [
        r"ថ្ងៃ.*?ព\.ស\.[0-9០១២៣៤៥៦៧៨៩]+",
        r"ធ្វើនៅរាជធានីភ្នំពេញ",
        r"ឧបនាយករដ្ឋមន្ត្រី",
        r"កន្លែងទទួល",
        r"រាជកិច្ច",
        r"ឯកសារ\s+កាលប្បវត្តិ",
    ]

    positions = []

    for marker in signature_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)

        if match:
            positions.append(match.start())

    appendix_match = re.search(r"^\s*ឧបសម្ព័ន្ធ", text, flags=re.MULTILINE)

    if appendix_match:
        positions.append(appendix_match.start())

    if positions:
        text = text[: min(positions)]

    return text.strip()


# Expand common regulatory abbreviations to improve retrieval quality
def expand_abbreviations(text):

    for abbr, full_form in ABBREVIATION_MAP.items():
        text = text.replace(abbr, f"{abbr} ({full_form})")

    return text


def clean_website_document(text):
    # Gemini can emit presentation-only HTML around centered captions/page
    # numbers in scanned PDFs. Keep the visible text, drop the markup, so the
    # retriever and generator never index UI-looking tokens such as <div>.
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(?:div|p|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"(?is)<[^>\n]+>", "", text)
    text = re.sub(r"(?m)^\s*(Image|Button:.*|FOLLOW US)\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Process and clean every merged document
for input_path in json_files:
    relative_path = input_path.relative_to(input_root)
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCleaning document: {relative_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("merged_text", "")
        category = data.get("category", "")

        if category == "laws":
            text = clean_law_document(text)

        elif category == "sub_decrees":
            text = clean_sub_decree_document(text)

        elif category == "prakas":
            text = clean_prakas_document(text)

        elif category == "guidance":
            text = clean_guidance_document(text)
            text = normalize_guidance_headings(text, data["document_id"])

            if data["document_id"] == "guidance_02":
                # Remove OCR-extracted markdown table artifacts
                text = re.sub(r"(?m)^\|.*\|\s*$", "", text)
                # Collapse empty lines left after table removal
                text = re.sub(r"\n{3,}", "\n\n", text)

            elif data["document_id"] == "guidance_03":
                # Remove OCR-extracted markdown table artifacts
                text = re.sub(r"(?m)^\|.*\|\s*$", "", text)
                # Collapse empty lines left after table removal
                text = re.sub(r"\n{3,}", "\n\n", text)

            elif data["document_id"] == "guidance_06":
                # Remove invalid trailing content inside section 1 to 2
                text = re.sub(
                    r"(?s)(<H1>\s*1\..*?)(?=\n<H1>\s*2\.)",
                    lambda m: re.sub(r"\s*តាមលក្ខខណ្ឌ\s*ដូចខាងក្រោម៖", "។", m.group(1)),
                    text,
                )
                text = re.sub(
                    r"(?s)(<H1>\s*1\..*?\។\n).*?(?=\n<H1>\s*2\.)", r"\1", text
                )
                # Normalize excessive blank lines
                text = re.sub(r"\n{3,}", "\n\n", text)

            elif data["document_id"] == "guidance_13":
                # Remove final page that begins with the signature date block
                pos = text.find("\n\nថ្ងៃ អង្គារ ១២ រោច ខែ មិគសិរ ឆ្នាំ ខាល ចត្វាស័ក ព.ស.២៥៦៦")

                if pos != -1:
                    text = text[:pos]

                text = re.sub(r"\n{3,}", "\n\n", text)

            elif data["document_id"] == "guidance_15":
                # Remove invalid content
                text = re.sub(
                    r"(?s)(ត្រូវមានរយៈពេល៨ម៉ោង។\n\n).*?(?=\n<H2>\s*1\.2\.)", r"\1", text
                )
                # Normalize excessive blank lines
                text = re.sub(r"\n{3,}", "\n\n", text)

        elif category in WEBSITE_CLEANING_CATEGORIES:
            text = clean_website_document(text)

        text = expand_abbreviations(text)

        output_data = {
            "document_id"   : data["document_id"],
            "category"      : category,
            "total_pages"   : data.get("total_pages", 0),
            "cleaned_text"  : text,
        }

        if data.get("source_url"):
            output_data["source_url"] = data["source_url"]
        if data.get("scraped_at"):
            output_data["scraped_at"] = data["scraped_at"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        processed_files += 1

        print(f"Completed: {relative_path}")

    except Exception as e:
        skipped_files += 1

        print(f"Failed: {relative_path}")
        print(e)

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Processed Files : {processed_files}")
print(f"Skipped Files   : {skipped_files}")
print(f"Runtime         : {elapsed_time:.2f} seconds")
print("Document cleaning completed successfully.")
