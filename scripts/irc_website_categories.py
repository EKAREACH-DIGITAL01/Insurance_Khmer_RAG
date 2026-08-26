"""Category helpers for IRC website-derived corpus documents.

The original student corpus keeps legal source PDFs in laws/prakas/sub_decrees/
guidance. Website crawl output uses a separate category namespace so pages and
website PDFs can be chunked by the section they came from without colliding with
those original source-PDF folders.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


WEBSITE_DERIVED_CATEGORIES = [
    "home",
    "statistics",
    "publication",
    "public_service",
    "insurance_institutions",
    "knowledge",
    "news",
    "events",
    "social_programs",
    "announcement",
    "law_regulation",
]

WEBSITE_CATEGORY_LABEL_KH = {
    "home": "ទំព័រដើម",
    "statistics": "ស្ថិតិ",
    "publication": "ការបោះពុម្ពផ្សាយ",
    "public_service": "សេវាសាធារណៈ",
    "insurance_institutions": "គ្រឹះស្ថានធានារ៉ាប់រង",
    "knowledge": "ចំណេះដឹង",
    "news": "ព័ត៌មាន",
    "events": "ព្រឹត្តិការណ៍",
    "social_programs": "កម្មវិធីសង្គម",
    "announcement": "សេចក្តីជូនដំណឹង",
    "law_regulation": "ច្បាប់និងបទប្បញ្ញត្តិ",
    "website": "ទំព័រគេហទំព័រ",
}

ARTICLE_CATEGORY_ALIASES = {
    "statistics-monthly": "statistics",
    "statistics-quarterly": "statistics",
    "statistics-annually": "statistics",
    "monthly-statistics": "statistics",
    "quarterly-statistics": "statistics",
    "annually-statistics": "statistics",
    "publication": "publication",
    "publications": "publication",
    "knowledge-sharing": "knowledge",
    "educational-broadcasting": "knowledge",
    "education": "knowledge",
    "knowledge": "knowledge",
    "news": "news",
    "event": "events",
    "events": "events",
    "social-programs": "social_programs",
    "social_programs": "social_programs",
    "announcement": "announcement",
    "announcements": "announcement",
    "laws": "law_regulation",
    "law": "law_regulation",
    "sub-decrees": "law_regulation",
    "sub_decrees": "law_regulation",
    "prakas": "law_regulation",
    "guidance": "law_regulation",
    "law-regulation": "law_regulation",
    "law_regulation": "law_regulation",
}

PATH_CATEGORY_ALIASES = {
    "public-service": "public_service",
    "public_service": "public_service",
    "insurance-agent-registration": "public_service",
    "company-submission-request": "public_service",
    "complaint": "public_service",
    "sandbox": "public_service",
    "insurance-institutions": "insurance_institutions",
    "insurance-institution": "insurance_institutions",
    "insurance_institutions": "insurance_institutions",
    "institution": "insurance_institutions",
    "institutions": "insurance_institutions",
}

TITLE_CATEGORY_HINTS = [
    (re.compile(r"\b(monthly|quarterly|annual|annually)\s+statistics\b", re.I), "statistics"),
    (re.compile(r"\binsurance\s+market\s+(?:report|data)\b", re.I), "statistics"),
    (re.compile(r"\bmarket\s+data\b", re.I), "statistics"),
    (re.compile(r"ស្ថិតិ|របាយការណ៍ទីផ្សារធានារ៉ាប់រង|ទិន្នន័យទីផ្សារ"), "statistics"),
    (
        re.compile(
            r"\bpublication|annual report|strategic development plan|glossary\b",
            re.I,
        ),
        "publication",
    ),
    (
        re.compile(
            r"ការបោះពុម្ពផ្សាយ|របាយការណ៍ប្រចាំឆ្នាំ|"
            r"ផែនការយុទ្ធសាស្ត្រអភិវឌ្ឍន៍វិស័យធានារ៉ាប់រង|"
            r"របាយការណ៍តាមដាន.*ផែនការយុទ្ធសាស្ត្រ|"
            r"សទ្ទានុក្រមធានារ៉ាប់រង"
        ),
        "publication",
    ),
    (re.compile(r"\bpublic\s+service|insurance agent registration|company submission|complaint|sandbox\b", re.I), "public_service"),
    (re.compile(r"សេវាសាធារណៈ|ចុះបញ្ជីភ្នាក់ងារ|ដាក់បណ្តឹង|បរិស្ថានសាកល្បង"), "public_service"),
    (re.compile(r"\binsurance institutions?\b", re.I), "insurance_institutions"),
    (re.compile(r"គ្រឹះស្ថានធានារ៉ាប់រង"), "insurance_institutions"),
    (re.compile(r"\beducational broadcasting|knowledge sharing|education\b", re.I), "knowledge"),
    (re.compile(r"ការអប់រំផ្សព្វផ្សាយ|ចំណេះដឹង|កម្រិតវប្បធម៌"), "knowledge"),
    (re.compile(r"\bnews\b", re.I), "news"),
    (re.compile(r"ព័ត៌មាន"), "news"),
    (re.compile(r"\bevent\b", re.I), "events"),
    (re.compile(r"ព្រឹត្តិការណ៍"), "events"),
    (re.compile(r"\bsocial programs?\b", re.I), "social_programs"),
    (re.compile(r"កម្មវិធីសង្គម"), "social_programs"),
    (re.compile(r"\bannouncement\b", re.I), "announcement"),
    (re.compile(r"សេចក្តីជូនដំណឹង|សេចក្ដីជូនដំណឹង"), "announcement"),
    (re.compile(r"\blaws?|sub decrees?|prakas|guidance|law & regulation\b", re.I), "law_regulation"),
    (re.compile(r"ច្បាប់|អនុក្រឹត្យ|ប្រកាស|សេចក្ដីណែនាំ|សេចក្ដី​ណែនាំ"), "law_regulation"),
]


def normalize_article_category(raw_category: str | None) -> str | None:
    if not raw_category:
        return None
    slug = raw_category.strip().lower().replace("_", "-")
    return ARTICLE_CATEGORY_ALIASES.get(slug) or PATH_CATEGORY_ALIASES.get(slug)


def category_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    query_category = parse_qs(parsed.query).get("category", [None])[0]
    normalized = normalize_article_category(query_category)
    if normalized:
        return normalized

    path = parsed.path.lower().strip("/")
    path_parts = [part for part in path.split("/") if part]
    for part in path_parts:
        normalized = normalize_article_category(part)
        if normalized:
            return normalized

    for part in path_parts:
        if part in PATH_CATEGORY_ALIASES:
            return PATH_CATEGORY_ALIASES[part]

    if "/files/cv/" in parsed.path.lower():
        return "home"

    return None


def category_from_text(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        match = re.match(r"^\s*Category:\s*(.+?)\s*$", line)
        if match:
            return normalize_article_category(match.group(1))
    return None


def title_hint_category(title_or_text: str | None) -> str | None:
    if not title_or_text:
        return None
    title = " ".join(str(title_or_text).split())
    for pattern, category in TITLE_CATEGORY_HINTS:
        if pattern.search(title):
            return category
    return None


def first_content_lines(text: str | None, max_lines: int = 5) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def is_home_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    return path in {"", "en", "kh"} and not parsed.query


def infer_website_category(
    *,
    source_url: str | None = None,
    parent_url: str | None = None,
    title: str | None = None,
    text: str | None = None,
    fallback: str = "home",
) -> str:
    """Infer the IRC website section for a page or linked file.

    Priority is intentionally conservative: explicit URL/category metadata wins,
    then article metadata embedded by the scraper, then a short title/header
    hint. Full body text is not scanned because every IRC page repeats the same
    navbar labels and that would misclassify ordinary article pages.
    """

    for candidate in (
        category_from_url(source_url),
        category_from_url(parent_url),
        category_from_text(text),
        title_hint_category(title),
        title_hint_category(first_content_lines(text)),
    ):
        if candidate in WEBSITE_DERIVED_CATEGORIES:
            return candidate

    if is_home_url(source_url) or is_home_url(parent_url):
        return "home"

    return fallback if fallback in WEBSITE_DERIVED_CATEGORIES else "home"
