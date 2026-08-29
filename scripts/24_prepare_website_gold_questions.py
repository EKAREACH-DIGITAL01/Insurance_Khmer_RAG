import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ----------------------------------------------------------------------------
# Website Gold-Question Seed Builder
# ----------------------------------------------------------------------------
# The student's original gold set covers the PDF-only legal categories:
# guidance, laws, prakas and sub_decrees. The IRC website expansion added
# additional categories that also need retrieval tests. This script creates
# source-grounded seed questions for those website categories from the active
# BGE-M3 metadata.
#
# These seed sets deliberately start at qid_026. In this project, qid_001-025
# means "core, hand-checked generator benchmark". Starting at 026 lets retrieval
# evaluation use the website categories immediately without pretending the new
# answers have already gone through the same human-validation pass.
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "vector_db" / "BAAI-BGE-M3" / "khmer_insurance_metadata.json"
EVALUATION_ROOT = PROJECT_ROOT / "data" / "09_evaluation"
REPORT_PATH = PROJECT_ROOT / "reports" / "website_gold_questions_seed.md"

START_QID = 26
TARGET_PER_CATEGORY = 10
MAX_CONTEXT_CHARS = 2400
MAX_ANSWER_CHARS = 1200

WEBSITE_CATEGORIES = [
    "announcement",
    "events",
    "home",
    "insurance_institutions",
    "knowledge",
    "law_regulation",
    "news",
    "public_service",
    "publication",
    "social_programs",
    "statistics",
]

LEGACY_CATEGORIES = {"guidance", "laws", "prakas", "sub_decrees"}

PUBLICATION_DOCUMENT_IDS = {
    "website_file_0a0b637917e6",
    "website_file_2f70838051e2",
    "website_file_39f076910114",
    "website_file_646db1aa65d3",
    "website_file_723cfd5541c1",
    "website_file_a2a9f95557bc",
    "website_file_c4a172db1586",
    "website_file_db047ce07365",
    "website_file_ea0a6e39a7f9",
}

GENERIC_TITLES = {
    "Announcement",
    "Event",
    "News",
    "Education",
    "Knowledge Sharing",
    "Educational Broadcasting",
    "Publication",
    "Monthly Statistics",
    "PUBLIC SERVICE",
    "INSURANCE REGULATOR OF CAMBODIA",
    "**INSURANCE REGULATOR OF CAMBODIA**",
    "**និយ័តករធានារ៉ាប់រងកម្ពុជា**",
    "ចំណេះដឹងពី",
    "ចំណេះដឹងពីធានារ៉ាប់រង",
    "ធានារ៉ាប់រង",
    "Insurance Institutions",
    "ព័ត៌មានថ្មីៗ",
    "កម្មវិធីសង្គម",
    "ការចែករំលែកចំណេះដឹង",
    "ការអប់រំផ្សព្វផ្សាយ",
    "និយ័តករធានារ៉ាប់រងកម្ពុជា",
    "និយ័តករធានារ៉ាប់រងកម្ពុជា - Insurance Regulator of Cambodia (ន.ធ.ក. (និយ័តករធានារ៉ាប់រងកម្ពុជា) - IRC)",
    "សេចក្ដីជូនដំណឹង",
    "សេចក្តីជូនដំណឹង",
    "ការបោះពុម្ពផ្សាយ",
    "ស្ថិតិប្រចាំខែ",
    "ស្ថិតិប្រចាំត្រីមាស",
    "ស្ថិតិប្រចាំឆ្នាំ",
}

CURATED_QUESTIONS = {
    "home": {
        "website_homepage_management_structure_chunk_0": "តើថ្នាក់ដឹកនាំ ន.ធ.ក. មានរណាខ្លះ?",
        "website_homepage_gross_premium_statistics_chunk_0": "តើទំព័រដើម ន.ធ.ក. បង្ហាញចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ២០២៥ប៉ុន្មាន?",
        "website_homepage_gross_premium_statistics_chunk_1": "យោងតាមទំព័រដើម ន.ធ.ក. តើបុព្វលាភធានារ៉ាប់រងដុលសរុបពីឆ្នាំ២០១៨ ដល់ឆ្នាំ២០២៣ មានចំនួនដូចម្តេច?",
    },
    "insurance_institutions": {
        "website_1fab36e16db2_chunk_0": "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយសមាគម និងក្រុមហ៊ុនធានារ៉ាប់រងទូទៅអ្វីខ្លះ?",
        "website_1fab36e16db2_chunk_1": "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយឈ្មោះក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចណាខ្លះ?",
        "website_1fab36e16db2_chunk_2": "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយឈ្មោះក្រុមហ៊ុនជើងសាធានារ៉ាប់រងណាខ្លះនៅចុងបញ្ជី?",
        "website_1fab36e16db2_chunk_3": "តើ Insurance Institutions directory បង្ហាញសមាគម និងក្រុមហ៊ុនធានារ៉ាប់រងទូទៅដំបូងៗណាខ្លះ?",
        "website_1fab36e16db2_chunk_4": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនធានារ៉ាប់រងទូទៅប្រភេទ GENERAL_INSURANCE ណាខ្លះនៅផ្នែកកណ្ដាល?",
        "website_1fab36e16db2_chunk_5": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ និងក្រុមហ៊ុនជើងសាធានារ៉ាប់រងណាខ្លះនៅផ្នែកបន្ត?",
        "website_1fab36e16db2_chunk_6": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនជើងសាធានារ៉ាប់រងប្រភេទ INSURANCE_BROKER ណាខ្លះនៅផ្នែកកណ្ដាល?",
        "website_1fab36e16db2_chunk_7": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនជើងសាធានារ៉ាប់រងណាខ្លះនៅចុងបញ្ជី?",
        "website_1fab36e16db2_chunk_8": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិតដំបូងៗណាខ្លះ?",
        "website_1fab36e16db2_chunk_9": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត និងក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូចណាខ្លះ?",
        "website_1fab36e16db2_chunk_10": "តើ Insurance Institutions directory បង្ហាញក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច និងក្រុមហ៊ុនធានារ៉ាប់រងបន្តណាខ្លះ?",
    },
    "publication": {
        "website_file_db047ce07365_chunk_13": "តើឆ្នាំ២០២៥ មានគ្រឹះស្ថានធានារ៉ាប់រងប៉ុន្មាននៅប្រទេសកម្ពុជា ហើយបែងចែកជាប្រភេទអ្វីខ្លះ?",
        "website_file_db047ce07365_chunk_14": "យោងតាមរបាយការណ៍ប្រចាំឆ្នាំ២០២៥ តើអន្តរការីធានារ៉ាប់រងបានកើនឡើងយ៉ាងដូចម្តេច?",
        "website_file_db047ce07365_chunk_15": "គិតត្រឹមចុងឆ្នាំ២០២៥ តើ ន.ធ.ក. បានផ្តល់អាជ្ញាបណ្ណសេវាគាំទ្រអាជីវកម្មធានារ៉ាប់រងចំនួនប៉ុន្មាន និងមានប្រភេទអ្វីខ្លះ?",
        "website_file_db047ce07365_chunk_16": "យោងតាមរបាយការណ៍ប្រចាំឆ្នាំ២០២៥ តើសូចនាករគន្លឹះសម្រាប់វាយតម្លៃវិស័យធានារ៉ាប់រងមានអ្វីខ្លះ?",
        "website_file_db047ce07365_chunk_17": "យោងតាមរបាយការណ៍ប្រចាំឆ្នាំ២០២៥ តើបុព្វលាភធានារ៉ាប់រងដុលកើនឡើងពីឆ្នាំ២០២១ ដល់ឆ្នាំ២០២៥ យ៉ាងដូចម្តេច?",
        "website_file_39f076910114_chunk_7": "តើផែនការយុទ្ធសាស្ត្រអភិវឌ្ឍន៍វិស័យធានារ៉ាប់រងកម្ពុជា ២០២៥-២០៣០ ត្រូវបានរៀបចំឡើងក្នុងបរិបទអ្វី?",
        "website_file_c4a172db1586_chunk_87": "តើសទ្ទានុក្រមធានារ៉ាប់រងពន្យល់អំពីអ្នកត្រូវបានធានារ៉ាប់រង អ្នកទទួលផល និងអ្នកប្រើប្រាស់យ៉ាងដូចម្តេច?",
        "website_file_c4a172db1586_chunk_63": "តើសទ្ទានុក្រមធានារ៉ាប់រងពន្យល់អំពីផែនការបន្តការងារ ពិន័យអន្តរកាល និងភ្នាក់ងារធានារ៉ាប់រងយ៉ាងដូចម្តេច?",
        "website_file_c4a172db1586_chunk_64": "តើសទ្ទានុក្រមធានារ៉ាប់រងពន្យល់អំពីភ្នាក់ងារផ្តល់ចំណាត់ថ្នាក់ឥណទាន ភាពគ្រប់គ្រាន់នៃដើមទុន និងភាគីពាក់ព័ន្ធយ៉ាងដូចម្តេច?",
        "website_publication_catalog_chunk_1": "តើការបោះពុម្ពផ្សាយផ្សេងទៀតរបស់ ន.ធ.ក. មានឯកសារអ្វីខ្លះ?",
    },
}

NAV_LINES = {
    "បរិស្ថានសាកល្បង",
    "ការបោះពុម្ពផ្សាយ",
    "គ្រឹះស្ថានធានារ៉ាប់រង",
    "ចុះបញ្ជីភ្នាក់ងារធានារ៉ាប់រង",
    "ក្រុមហ៊ុនផ្ដល់/ស្នើ",
    "ដាក់បណ្តឹង",
    "ច្បាប់និងបទប្បញ្ញត្តិ",
    "និយ័តករ​ធានារ៉ាប់រង​កម្ពុជា",
    "ច្បាប់",
    "អនុក្រឹត្យ",
    "ប្រកាស",
    "សេចក្ដី​ណែនាំ",
    "ផ្សេងទៀត",
    "ស្ថិតិ",
    "ប្រចាំខែ",
    "ប្រចាំត្រីមាស",
    "ប្រចាំឆ្នាំ",
    "សេវា​សាធារណៈ",
    "និយ័តករធានារ៉ាប់រងកម្ពុជា",
    "អំពី​​យើង",
    "អំពីយើង",
    "ទំនាក់ទំនង",
    "ឯកសារ",
    "ទាញយកឯកសារ",
    "ឯកសារ ទាញយកឯកសារ",
    "យើង​ទទួល",
    "យើងទទួល",
    "ប្រព័ន្ធ​ផ្សព្វផ្សាយ​សង្គម",
    "Sandbox",
    "Publication",
    "Insurance Institutions",
    "Insurance Agent Registration",
    "Company Submission/Request",
    "Complaint",
    "LAW & REGULATION",
    "Laws",
    "Sub Decrees",
    "Prakas",
    "Guidance",
    "Other",
    "STATISTICS",
    "Monthly",
    "Quarterly",
    "Annually",
    "PUBLIC SERVICE",
    "INSURANCE REGULATOR OF CAMBODIA",
    "INSURANCE REGULATOR",
    "of CAMBODIA",
    "INSURANCE REGULATOR of CAMBODIA",
    "ចំណេះដឹងពី",
    "ចំណេះដឹងពីធានារ៉ាប់រង",
    "ធានារ៉ាប់រង",
    "About Us",
    "Contact Us",
    "File",
    "Download",
    "File Download",
    "View Other",
    "We accept",
    ":",
    "---",
}

NAV_SUBSTRINGS = [
    "UNSC",
    "AML & PF",
    "រក្សាសិទ្ធិដោយនិយ័តករធានារ៉ាប់រងកម្ពុជា",
    "ប្រព័ន្ធ​ផ្សព្វផ្សាយ​សង្គម",
    "gad@irc.gov.kh",
    "071 258 0896",
    "St. 598",
    "ផ្លូវ ៥៩៨",
    "Khan Russey Keo",
    "រាជធានីភ្នំពេញ កម្ពុជា",
    "Insurance Regulator of Cambodia",
    "©2022",
    "All rights reserved",
    "Content type:",
]

KHMER_RE = re.compile(r"[ក-៿]")
UUID_PDF_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}\.pdf$", re.I)
PDF_SUFFIX_RE = re.compile(r"\s+-\s+[0-9a-f]{8}-[0-9a-f-]{27,}\.pdf$", re.I)
DATE_LINE_RE = re.compile(
    r"^([០-៩0-9]{1,2}\s+)?"
    r"(មករា|កុម្ភៈ|មីនា|មេសា|ឧសភា|មិថុនា|កក្កដា|សីហា|កញ្ញា|តុលា|វិច្ឆិកា|ធ្នូ|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+[០-៩0-9]{4}.*",
    re.I,
)


def load_chunks():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing active metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}"
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def chunk_category(chunk):
    md = chunk.get("metadata", {})
    return md.get("category") or md.get("document_type") or "unknown"


def inferred_gold_category(chunk):
    """Return the category that should own this evaluation seed.

    A few Gemini-extracted PDFs were initially indexed under `home` because they
    are object-storage URLs without their parent article route. The gold set
    should still evaluate them under their content category.
    """
    md = chunk.get("metadata", {})
    current = chunk_category(chunk)
    doc_id = md.get("document_id") or ""
    source_url = md.get("source_url") or ""
    name = md.get("document_name") or ""
    text = raw_text(chunk)
    haystack = f"{name}\n{source_url}\n{text[:1600]}"

    if current in LEGACY_CATEGORIES:
        return current
    if doc_id in PUBLICATION_DOCUMENT_IDS or doc_id == "website_publication_catalog":
        return "publication"
    if doc_id == "website_1fab36e16db2":
        return "insurance_institutions"
    if doc_id.startswith("website_homepage_"):
        return "home"
    if "/public-service/" in source_url:
        return "public_service"
    if "object.irc.gov.kh/irc-public/upload/" in source_url:
        if "ចំណេះដឹងពីធានារ៉ាប់រង" in text or re.search(r"(លេខ|No\.)\s*[៖:.]?\s*០?២[៤៥៦]", haystack):
            return "knowledge"
        if re.search(r"របាយការណ៍ទីផ្សារ|ទិន្នន័យទីផ្សារ|Insurance Market|Gross Premium|Premium Income", haystack, re.I):
            return "statistics"
    return current


def raw_text(chunk):
    return chunk.get("text") or chunk.get("content") or ""


def has_khmer(text):
    return bool(KHMER_RE.search(text or ""))


def khmer_count(text):
    return sum(1 for ch in text or "" if KHMER_RE.match(ch))


def is_noise_line(line):
    s = line.strip()
    normalized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s).strip(" *_")
    if not s:
        return True
    if s in NAV_LINES or normalized in NAV_LINES or normalized in GENERIC_TITLES:
        return True
    if s.startswith(("Source file:", "OCR method:", "Logo:", "Content type:")):
        return True
    if s.startswith(("Category:", "Publish date:", "Download file:")):
        return True
    if s.startswith("[Page ") and s.endswith("]"):
        return True
    if s.startswith(("http://", "https://")):
        return True
    if any(part in s for part in NAV_SUBSTRINGS):
        return True
    if re.fullmatch(r"[\|\-\:\s]+", normalized):
        return True
    if normalized.startswith("💡") and len(normalized) < 40:
        return True
    if "សហការឧបត្ថម្ភដោយ" in normalized:
        return True
    if re.fullmatch(r"\(?\s*[0-9]+(?:\.[0-9]+)?\s*MB\s*\)?", s, re.I):
        return True
    if re.fullmatch(r"[០-៩0-9]+", s):
        return True
    if DATE_LINE_RE.match(normalized):
        return True
    if s.startswith("Building #"):
        return True
    if s.startswith("អគារលេខ") and len(s) < 150:
        return True
    if UUID_PDF_RE.match(s):
        return True
    return False


def collapse_ws(text):
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", text)).strip()


def cleaned_lines(text):
    lines = []
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        s = s.replace("**", "")
        s = s.strip("*_")
        s = s.lstrip("#").strip()
        s = strip_uuid_suffix(s)
        if not is_noise_line(s):
            lines.append(s)
    return lines


def clean_text(text):
    return collapse_ws("\n".join(cleaned_lines(text)))


def meaningful_line_count(text):
    return len(cleaned_lines(text))


def is_page_shell(chunk):
    text = expected_answer(chunk)
    if len(text) < 180:
        return True
    shell_terms = [
        "Download file:",
        "Content type:",
        "Category:",
        "Publish date:",
        "Insurance Agent Registration",
        "Company Submission/Request",
        "UNSC Consolidated List",
    ]
    raw = raw_text(chunk)
    if sum(raw.count(term) for term in shell_terms) >= 4 and meaningful_line_count(raw) < 6:
        return True
    return False


def trim_text(text, limit):
    text = collapse_ws(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut.rstrip("។,;:") + "។"


def strip_uuid_suffix(title):
    title = PDF_SUFFIX_RE.sub("", title)
    title = re.sub(r"\s+[0-9a-f]{8}-[0-9a-f-]{27,}\.pdf$", "", title, flags=re.I)
    return title.strip()


def title_from_content(chunk):
    lines = cleaned_lines(raw_text(chunk))
    for idx, line in enumerate(lines):
        s = strip_uuid_suffix(line)
        s = s.strip("*_")
        s = s.lstrip("#").strip()
        if s in GENERIC_TITLES:
            continue
        if UUID_PDF_RE.match(s):
            continue
        if s.lower().endswith(".pdf") and not has_khmer(s):
            continue
        if s == "ទិន្នន័យទីផ្សារធានារ៉ាប់រងកម្ពុជា":
            for next_line in lines[idx + 1 : idx + 4]:
                if "ត្រីមាស" in next_line or "ប្រចាំ" in next_line:
                    return trim_text(f"{s} {next_line}", 170)
        if has_khmer(s) and 8 <= len(s) <= 170:
            return s
    return None


def derive_title(chunk):
    md = chunk.get("metadata", {})
    heading = md.get("heading_path") or md.get("heading_title") or md.get("section_title")
    if heading and has_khmer(heading):
        return trim_text(heading, 170)

    name = strip_uuid_suffix(md.get("document_name") or "")
    if name and name not in GENERIC_TITLES and not UUID_PDF_RE.match(name):
        return trim_text(name, 170)

    from_body = title_from_content(chunk)
    if from_body:
        return trim_text(from_body, 170)

    return trim_text(name or md.get("document_id") or chunk["chunk_id"], 170)


def expected_answer(chunk):
    text = clean_text(raw_text(chunk))
    return trim_text(text, MAX_ANSWER_CHARS)


def context_text(chunk):
    return trim_text(clean_text(raw_text(chunk)), MAX_CONTEXT_CHARS)


def is_usable_candidate(category, chunk):
    md = chunk.get("metadata", {})
    text = expected_answer(chunk)
    chunk_id = chunk["chunk_id"]
    doc_id = md.get("document_id", "")
    source_url = md.get("source_url") or ""
    title = derive_title(chunk)

    if chunk_id in CURATED_QUESTIONS.get(category, {}):
        return True
    if len(text) < 100 or khmer_count(text) < 25:
        return False
    if category in {"publication", "statistics", "public_service"}:
        # Prefer downloaded PDFs/forms over category-list shells.
        if not doc_id.startswith("website_file_"):
            return False
    if category == "insurance_institutions":
        return doc_id == "website_1fab36e16db2"
    if category == "home":
        # The homepage scrape also discovered linked PDFs. Keep the "home"
        # gold set focused on the actual homepage management/statistics blocks.
        return doc_id.startswith("website_homepage_")
    if category == "knowledge":
        if not doc_id.startswith("website_file_"):
            return False
        if "Source file:" not in raw_text(chunk):
            return False
        if title.lower().startswith("no."):
            return False
        if (md.get("document_name") or "") in GENERIC_TITLES:
            return False
        if title in GENERIC_TITLES:
            return False
    if category in {"announcement", "events"} and "/article/list" in source_url:
        return False
    if category in {"announcement", "events"} and is_page_shell(chunk):
        return False
    if category in {"news", "social_programs"} and "/article/list" in source_url:
        return False
    if category in {"news", "events", "social_programs", "law_regulation"}:
        if title in GENERIC_TITLES and khmer_count(text) < 120:
            return False
        if is_page_shell(chunk):
            return False
    return True


def candidate_score(category, chunk):
    md = chunk.get("metadata", {})
    text = expected_answer(chunk)
    title = derive_title(chunk)
    method = md.get("chunking_method") or ""
    doc_id = md.get("document_id") or ""

    score = khmer_count(text) + min(len(text), 1600) // 2

    if title in GENERIC_TITLES:
        score -= 250
    if doc_id.startswith("website_file_"):
        score += 180
    if method in {
        "publication_section_window",
        "statistics_table_window",
        "public_service_steps",
        "institution_directory",
        "homepage_blocks",
        "knowledge_article",
        "news_article",
        "event_article",
        "social_program_article",
    }:
        score += 120
    if category == "home" and doc_id.startswith("website_homepage_"):
        score += 500
    if category == "publication" and doc_id == "website_file_db047ce07365":
        score += 450
    if category == "statistics" and "២០២៦" in text:
        score += 200
    if category == "public_service" and "ឯកសារតម្រូវ" in text:
        score += 250
    if category == "insurance_institutions" and "ក្រុមហ៊ុន" in text:
        score += 250
    if "data-tip=" in text or "aria-label=" in text:
        score -= 300

    return score


def question_for(category, chunk):
    override = CURATED_QUESTIONS.get(category, {}).get(chunk["chunk_id"])
    if override:
        return override

    title = derive_title(chunk)
    text = expected_answer(chunk)

    if category == "announcement":
        return f"តើសេចក្តីជូនដំណឹង «{title}» មានខ្លឹមសារសំខាន់អ្វីខ្លះ?"

    if category == "events":
        return f"តើព្រឹត្តិការណ៍ «{title}» បានបង្ហាញព័ត៌មានសំខាន់អ្វីខ្លះ?"

    if category == "home":
        if chunk["chunk_id"].startswith("website_homepage_management_structure"):
            return "តើថ្នាក់ដឹកនាំ ន.ធ.ក. មានរណាខ្លះ?"
        if "បុព្វលាភ" in text and "២០២៥" in text:
            return "តើទំព័រដើម ន.ធ.ក. បង្ហាញចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ២០២៥ប៉ុន្មាន?"
        return f"តើទំព័រដើម ឬឯកសារ «{title}» បង្ហាញព័ត៌មានសំខាន់អ្វីខ្លះ?"

    if category == "insurance_institutions":
        if "សមាគម" in text and "ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ" in text:
            return "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយសមាគម និងក្រុមហ៊ុនធានារ៉ាប់រងទូទៅអ្វីខ្លះ?"
        if "ឡាយហ្វ៍" in text or "ធានារ៉ាប់រងអាយុជីវិត" in text:
            return "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយឈ្មោះក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិតណាខ្លះ?"
        if "ប្រូឃើរ" in text or "BROKER" in text:
            return "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងរាយឈ្មោះក្រុមហ៊ុនជើងសាធានារ៉ាប់រងណាខ្លះ?"
        if "Type:" in text:
            return "តើបញ្ជី Insurance Institutions របស់ ន.ធ.ក. បង្ហាញឈ្មោះគ្រឹះស្ថាន និងប្រភេទអ្វីខ្លះ?"
        return "តើទំព័រគ្រឹះស្ថានធានារ៉ាប់រងបង្ហាញបញ្ជីគ្រឹះស្ថានណាខ្លះនៅក្នុងផ្នែកនេះ?"

    if category == "knowledge":
        if "មានន័យ" in title or "ជាអ្វី" in title or "Who is" in title or "What is" in title:
            return f"តើ «{title}» មានន័យដូចម្តេចក្នុងវិស័យធានារ៉ាប់រង?"
        return f"តើអត្ថបទចំណេះដឹង «{title}» ពន្យល់អំពីអ្វីខ្លះ?"

    if category == "law_regulation":
        if title in {"Prakas", "ប្រកាស", "Guidance", "សេចក្ដី​ណែនាំ", "Sub Decrees", "អនុក្រឹត្យ", "Laws", "ច្បាប់"}:
            return f"តើផ្នែកច្បាប់និងបទប្បញ្ញត្តិរាយបញ្ជីឯកសារអ្វីខ្លះក្នុងផ្នែក «{title}»?"
        return f"តើឯកសារ «{title}» ក្នុងផ្នែកច្បាប់និងបទប្បញ្ញត្តិបង្ហាញអំពីអ្វីខ្លះ?"

    if category == "news":
        return f"តើព័ត៌មាន «{title}» និយាយអំពីអ្វីខ្លះ?"

    if category == "public_service":
        return f"តើសេវា «{title}» ត្រូវការឯកសារអ្វីខ្លះ?"

    if category == "publication":
        if "គ្រឹះស្ថានធានារ៉ាប់រង" in text and "២០២៥" in text:
            return "តើឆ្នាំ២០២៥ មានគ្រឹះស្ថានធានារ៉ាប់រងប៉ុន្មាននៅប្រទេសកម្ពុជា?"
        if "បុព្វលាភ" in text and "២០២៥" in text:
            return f"យោងតាម «{title}» តើទិន្នន័យបុព្វលាភធានារ៉ាប់រងសំខាន់ៗបង្ហាញដូចម្តេច?"
        return f"យោងតាម «{title}» តើខ្លឹមសារសំខាន់ក្នុងផ្នែកនេះបង្ហាញអ្វីខ្លះ?"

    if category == "social_programs":
        return f"តើកម្មវិធីសង្គម «{title}» មានគោលបំណង និងព័ត៌មានសំខាន់អ្វីខ្លះ?"

    if category == "statistics":
        if "ត្រីមាស" in title:
            return f"យោងតាម «{title}» តើទិន្នន័យទីផ្សារធានារ៉ាប់រងត្រីមាសបង្ហាញអ្វីខ្លះ?"
        if "ប្រចាំខែ" in title or "ខែ" in title:
            return f"យោងតាម «{title}» តើបុព្វលាភដុល និងស្ថិតិសំខាន់ៗប្រចាំខែបង្ហាញដូចម្តេច?"
        return f"យោងតាម «{title}» តើស្ថិតិទីផ្សារធានារ៉ាប់រងកម្ពុជាបង្ហាញអ្វីខ្លះ?"

    return f"តើឯកសារ «{title}» បង្ហាញព័ត៌មានសំខាន់អ្វីខ្លះ?"


def max_chunks_per_doc(category):
    if category == "insurance_institutions":
        return 10
    if category == "publication":
        return 3
    if category in {"home", "statistics"}:
        return 2
    return 1


def select_chunks(category, chunks):
    candidates = [c for c in chunks if is_usable_candidate(category, c)]
    candidates.sort(key=lambda c: candidate_score(category, c), reverse=True)

    selected = []
    per_doc = Counter()
    questions = set()

    curated_ids = CURATED_QUESTIONS.get(category, {})
    lookup = {c["chunk_id"]: c for c in candidates}
    ordered_candidates = [
        lookup[chunk_id]
        for chunk_id in curated_ids
        if chunk_id in lookup
    ]
    ordered_candidates.extend(
        c for c in candidates if c["chunk_id"] not in curated_ids
    )

    for chunk in ordered_candidates:
        md = chunk.get("metadata", {})
        doc_id = md.get("document_id") or chunk["chunk_id"]
        is_curated = chunk["chunk_id"] in curated_ids
        if not is_curated and per_doc[doc_id] >= max_chunks_per_doc(category):
            continue
        q = question_for(category, chunk)
        if q in questions:
            continue
        selected.append(chunk)
        per_doc[doc_id] += 1
        questions.add(q)
        if len(selected) >= TARGET_PER_CATEGORY:
            break

    return selected


def build_gold_entry(category, chunk, seq):
    qid = START_QID + seq
    return {
        "question_id": f"{category}_qid_{qid:03d}",
        "question": question_for(category, chunk),
        "expected_chunks": [chunk["chunk_id"]],
        "context": context_text(chunk),
        "expected_answer": expected_answer(chunk),
    }


def write_gold_file(category, entries):
    out_dir = EVALUATION_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gold_questions.json"
    if not entries:
        if out_path.exists():
            out_path.unlink()
        return out_path
    out_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def write_report(summary_rows):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Website Gold-Question Seed Sets",
        "",
        "These seed sets extend the student's original legal gold questions to the IRC website categories.",
        "",
        "- Existing student categories kept unchanged: `guidance`, `laws`, `prakas`, `sub_decrees`.",
        "- New website-category questions start at `qid_026`, so they are treated as retrieval/human-evaluation seeds rather than hand-validated generator-core questions.",
        "- Every question points to a real current chunk ID from `vector_db/BAAI-BGE-M3/khmer_insurance_metadata.json`.",
        "",
        "| Category | Questions | Notes |",
        "| --- | ---: | --- |",
    ]
    for category, count, note in summary_rows:
        lines.append(f"| `{category}` | {count} | {note} |")
    lines.append("")
    lines.append(
        "Before promoting any website seed question into the core generator benchmark, have a reviewer check the source chunk and move it into the `qid_001`-`qid_025` range for that category."
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    chunks = load_chunks()
    by_category = defaultdict(list)
    for chunk in chunks:
        category = inferred_gold_category(chunk)
        if category in LEGACY_CATEGORIES:
            continue
        by_category[category].append(chunk)

    summary_rows = []
    total = 0

    for category in WEBSITE_CATEGORIES:
        selected = select_chunks(category, by_category.get(category, []))
        entries = [
            build_gold_entry(category, chunk, seq)
            for seq, chunk in enumerate(selected)
        ]
        write_gold_file(category, entries)
        total += len(entries)

        note = "source-grounded seed set"
        if len(entries) < TARGET_PER_CATEGORY:
            note = f"only {len(entries)} usable chunks passed filters"
        if len(entries) == 0:
            note = "no source-rich chunks passed filters; re-scrape article bodies before adding gold questions"
        if category == "announcement":
            note += "; scrape is thin and should be human-reviewed first"
        summary_rows.append((category, len(entries), note))

    write_report(summary_rows)

    print(f"Wrote {total} website gold questions across {len(WEBSITE_CATEGORIES)} categories.")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
