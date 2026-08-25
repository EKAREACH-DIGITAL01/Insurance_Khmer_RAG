import argparse
import html as html_lib
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from irc_website_categories import (
    WEBSITE_DERIVED_CATEGORIES,
    infer_website_category,
)

try:
    import requests
except ImportError:
    requests = None


BASE_URL = "https://www.irc.gov.kh/"
USER_AGENT = "Khmer-RAG-Insurance-Chatbot dataset scraper (+research corpus build)"

SCRAPE_ROOT = Path("data/00_irc_website_scrape")
PAGE_ROOT = SCRAPE_ROOT / "pages"
FILE_ROOT = SCRAPE_ROOT / "files"
MERGED_DOC_ROOT = Path("data/05_merged_documents")
CLEANED_DOC_ROOT = Path("data/06_cleaned_documents")
CHUNKED_DOC_ROOT = Path("data/07_chunked_documents")
WEBSITE_DOC_ROOT = MERGED_DOC_ROOT / "website"
CLEANED_WEBSITE_DOC_ROOT = CLEANED_DOC_ROOT / "website"
CHUNKED_WEBSITE_DOC_ROOT = CHUNKED_DOC_ROOT / "website"
SOURCE_PDF_ROOT = Path("data/01_source_documents")
METADATA_PATH = Path("data/document_metadata.json")
ORIGINAL_SOURCE_PDF_CATEGORIES = ("laws", "sub_decrees", "prakas", "guidance")

LIST_CATEGORIES = [
    "laws",
    "sub-decrees",
    "prakas",
    "guidance",
    "publication",
    "statistics-monthly",
    "statistics-quarterly",
    "statistics-annually",
    "news",
    "event",
    "education",
    "announcement",
    "educational-broadcasting",
    "knowledge-sharing",
    "social-programs",
]
DEFAULT_SEEDS = [
    "https://www.irc.gov.kh/",
    "https://www.irc.gov.kh/en",
    "https://www.irc.gov.kh/public-service",
    "https://www.irc.gov.kh/en/public-service",
    "https://www.irc.gov.kh/insurance-institutions",
    "https://www.irc.gov.kh/en/insurance-institutions",
    "https://www.irc.gov.kh/agent/registration-menu",
    "https://www.irc.gov.kh/en/agent/registration-menu",
    "https://www.irc.gov.kh/complaint",
    "https://www.irc.gov.kh/en/complaint",
]
for category in LIST_CATEGORIES:
    DEFAULT_SEEDS.append(f"https://www.irc.gov.kh/article/list?category={category}&page=1")
    DEFAULT_SEEDS.append(f"https://www.irc.gov.kh/en/article/list?category={category}&page=1")

INTERNAL_HOSTS = {"irc.gov.kh", "www.irc.gov.kh"}
FILE_HOSTS = INTERNAL_HOSTS | {"object.irc.gov.kh"}
FILE_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".zip",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_EXTENSIONS = {".css", ".js", ".ico", ".svg", ".woff", ".woff2", ".ttf", ".map"}
DROP_TEXT = {
    "image",
    "english",
    "ភាសាខ្មែរ",
    "ទំព័រដើម",
    "home",
    "follow us",
}

INSTITUTION_TYPE_LABELS = {
    "ASSOCIATION": "សមាគម / Association",
    "GENERAL_INSURANCE": "ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ / General insurance company",
    "LIFE_INSURANCE": "ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត / Life insurance company",
    "MICRO_INSURANCE": "ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច / Micro insurance company",
    "REINSURANCE": "ក្រុមហ៊ុនធានារ៉ាប់រងបន្ត / Reinsurance company",
    "INSURANCE_BROKER": "ក្រុមហ៊ុនជើងសាធានារ៉ាប់រង / Insurance broker",
}

_SOURCE_PDF_HASHES = None

MANAGEMENT_STRUCTURE_TEXT = """
Management Structure
Official homepage management structure for the Insurance Regulator of Cambodia (IRC / ន.ធ.ក.).
NBFSA BOARD
H.E Bou Chanphirou - Director General - Insurance Regulator of Cambodia
Dr. Hor Sovathana - Deputy Director General
Ms. Tan Linna - Deputy Director General
Mr. Var Vichea - Director - Department of License and Legal Affairs
Ms. Ngan Leakhena - Director - Department of General Affairs
Mr. Touch Mengleng - Director - Department of Market Development
Department of Research, Training and Cooperation
Department of Supervision

រចនាសម្ព័ន្ធថ្នាក់ដឹកនាំ
ថ្នាក់ដឹកនាំ ន.ធ.ក. រួមមាន៖
ឯកឧត្តម ប៊ូ ច័ន្ទភីរូ - អគ្គនាយក - និយ័តករធានារ៉ាប់រងកម្ពុជា
លោកស្រីបណ្ឌិត ហោ សុវឌ្ឍនា - អគ្គនាយករង
លោកស្រី តាន់ លីនណា - អគ្គនាយករង
លោក វ៉ា វិជ្ជា - ប្រធាននាយកដ្ឋានគ្រប់គ្រងអាជ្ញាបណ្ណ និងកិច្ចការគតិយុត្ត
លោកស្រី ង៉ាន់ លក្ខិណា - ប្រធាននាយកដ្ឋានកិច្ចការទូទៅ
លោក ទូច ម៉េងឡេង - ប្រធាននាយកដ្ឋានអភិវឌ្ឍន៍ទីផ្សារ
នាយកដ្ឋានស្រាវជ្រាវ បណ្តុះបណ្តាល និងសហប្រតិបត្តិការ
នាយកដ្ឋានត្រួតពិនិត្យ
""".strip()

MANAGEMENT_ASSET_PATHS = [
    "/files/cv/Bou Chanphirou/H.E Bou Chanphirou_ Updataed Biography.pdf",
    "/files/cv/Bou Chanphirou/BouChanphirou-min.jpg",
    "/files/cv/Hor Sovathana/CV_Sovathana_IRC Website_2022_ENG.pdf",
    "/files/cv/Hor Sovathana/CV_Sovathana_IRC_2021_ENG-images/0001-min.jpg",
    "/files/cv/Hor Sovathana/CV_Sovathana_IRC_2021_ENG-images/0002-min.jpg",
    "/files/cv/Hor Sovathana/CV_Sovathana_IRC_2021_KH-images/0001-min.jpg",
    "/files/cv/Hor Sovathana/CV_Sovathana_IRC_2021_KH-images/0002-min.jpg",
    "/files/cv/Ngan Leakhena/Ngan Leakhena-Bio.jpg",
    "/files/cv/Tan Linna/CV Tan Linna .pdf",
    "/files/cv/Tan Linna/CV Tan Linna -images/0001-min.jpg",
    "/files/cv/TouchMengleng/Mengleng Touch - Resume - English.pdf",
    "/files/cv/TouchMengleng/Mengleng Touch - Resume - English-images/0001-min.jpg",
    "/files/cv/TouchMengleng/Mengleng Touch - Resume - English-images/0002-min.jpg",
    "/files/cv/TouchMengleng/ទូច ម៉េងឡេង -​ ប្រវត្តិរូបសង្ខេប.pdf",
    "/files/cv/TouchMengleng/ទូច ម៉េងឡេង -_ ប្រវត្តិរូបសង្ខេប-images/0001-min.jpg",
    "/files/cv/TouchMengleng/ទូច ម៉េងឡេង -_ ប្រវត្តិរូបសង្ខេប-images/0002-min.jpg",
    "/files/cv/Va Vichea/Curriculum Vitae អង់គ្លេស.pdf",
    "/files/cv/Va Vichea/Curriculum Vitae អង់គ្លេស-images/0001-min.jpg",
    "/files/cv/Va Vichea/Curriculum Vitae អង់គ្លេស-images/0002-min.jpg",
    "/files/cv/Va Vichea/Curriculum Vitae ខ្មែរ.pdf",
    "/files/cv/Va Vichea/Curriculum Vitae ខ្មែរ-images/0001-min.jpg",
    "/files/cv/Va Vichea/Curriculum Vitae ខ្មែរ-images/0002-min.jpg",
    "/files/cv/Va Vichea/Curriculum Vitae ខ្មែរ-images/0003-min.jpg",
]

GROSS_PREMIUM_STATISTICS_TEXT = """
Total Gross Premium
បុព្វលាភធានារ៉ាប់រងដុល
Official homepage graph data for the Insurance Regulator of Cambodia (IRC / ន.ធ.ក.).
This is official insurance statistics, not a product premium quote or personal premium calculation.

ឆ្នាំ២០២៥ / 2025: General Insurance 151,933,652 USD; Life Insurance 204,216,341 USD; Micro insurance 10,898,912 USD; Total Gross Premium 367,048,905 USD.
ចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ២០២៥ គឺ 367,048,905 ដុល្លារ អាមេរិក។

ឆ្នាំ២០២៤ / 2024: General Insurance 147,636,549 USD; Life Insurance 200,407,724 USD; Micro insurance 8,872,543 USD; Total Gross Premium 356,916,816 USD.
ចំនួនបុព្វលាភធានារ៉ាប់រងដុលសរុបសម្រាប់ឆ្នាំ២០២៤ គឺ 356,916,816 ដុល្លារ អាមេរិក។

ឆ្នាំ២០២៣ / 2023: General Insurance 150,727,745 USD; Life Insurance 193,539,426 USD; Micro insurance 0 USD; Total Gross Premium 344,267,171 USD.
ឆ្នាំ២០២២ / 2022: General Insurance 132,340,704 USD; Life Insurance 193,808,543 USD; Micro insurance 5,716,368 USD; Total Gross Premium 331,865,615 USD.
ឆ្នាំ២០២១ / 2021: General Insurance 122,965,797 USD; Life Insurance 170,458,869 USD; Micro insurance 4,129,267 USD; Total Gross Premium 297,553,933 USD.
ឆ្នាំ២០២០ / 2020: General Insurance 114,485,277 USD; Life Insurance 153,374,081 USD; Micro insurance 5,034,998 USD; Total Gross Premium 272,894,356 USD.
ឆ្នាំ២០១៩ / 2019: General Insurance 101,383,939 USD; Life Insurance 144,400,835 USD; Micro insurance 7,059,819 USD; Total Gross Premium 252,844,593 USD.
ឆ្នាំ២០១៨ / 2018: General Insurance 86,640,270 USD; Life Insurance 100,770,354 USD; Micro insurance 9,259,753 USD; Total Gross Premium 196,670,377 USD.
""".strip()


class StdlibResponse:
    def __init__(self, raw, stream=False):
        self.raw = raw
        self.status_code = raw.status
        self.headers = raw.headers
        self._body = None if stream else raw.read()

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    @property
    def text(self):
        body = self._body if self._body is not None else self.raw.read()
        charset = self.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        while True:
            chunk = self.raw.read(chunk_size)
            if not chunk:
                break
            yield chunk
        self.raw.close()


class StdlibSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=30, stream=False):
        request = Request(url, headers=self.headers)
        return StdlibResponse(urlopen(request, timeout=timeout), stream=stream)


def make_session():
    if requests:
        session = requests.Session()
    else:
        session = StdlibSession()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(scheme=scheme, netloc=netloc, path=path).geturl()


def stable_id(prefix, value):
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def website_category_root(root, category):
    category = category if category in WEBSITE_DERIVED_CATEGORIES else "home"
    return root / category


def website_pipeline_path(root, category, document_id, suffix=".json"):
    return website_category_root(root, category) / f"{document_id}{suffix}"


def existing_website_pipeline_paths(document_id):
    paths = [
        WEBSITE_DOC_ROOT / f"{document_id}.json",
        CLEANED_WEBSITE_DOC_ROOT / f"{document_id}.json",
        CHUNKED_WEBSITE_DOC_ROOT / f"{document_id}_chunks.json",
    ]
    for category in WEBSITE_DERIVED_CATEGORIES:
        paths.extend(
            [
                MERGED_DOC_ROOT / category / f"{document_id}.json",
                CLEANED_DOC_ROOT / category / f"{document_id}.json",
                CHUNKED_DOC_ROOT / category / f"{document_id}_chunks.json",
            ]
        )
    return paths


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_pdf_hashes():
    """Hashes of the original 49 student-processed PDFs.

    Website PDF files that match these exactly are skipped as website documents
    so the expanded corpus adds only new website content instead of duplicating
    law/sub-decree/prakas/guidance PDFs already represented in the base corpus.
    """
    global _SOURCE_PDF_HASHES
    if _SOURCE_PDF_HASHES is None:
        hashes = {}
        for category in ORIGINAL_SOURCE_PDF_CATEGORIES:
            for path in sorted((SOURCE_PDF_ROOT / category).glob("*.pdf")):
                try:
                    hashes[file_sha256(path)] = str(path)
                except OSError:
                    continue
        _SOURCE_PDF_HASHES = hashes
    return _SOURCE_PDF_HASHES


def duplicate_source_pdf_path(path):
    try:
        if not is_pdf_path(path):
            return None
        return source_pdf_hashes().get(file_sha256(path))
    except OSError:
        return None


def safe_filename(url, content_type=None):
    parsed = urlparse(url)
    name = Path(parsed.path).name or stable_id("file", url)
    if "." not in name and content_type:
        if "pdf" in content_type:
            name += ".pdf"
        elif "spreadsheet" in content_type or "excel" in content_type:
            name += ".xlsx"
        elif "word" in content_type:
            name += ".docx"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem or f"{stable_id('file', url)}.bin"


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.text_parts = []
        self.title_parts = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        href = attrs.get("href") or attrs.get("src")
        if href:
            self.links.append(href)
        if tag in {"p", "div", "section", "article", "li", "tr", "br", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)


def extract_next_json(html):
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(html_lib.unescape(match.group(1)))
    except Exception:
        return None


def iter_json_values(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_values(child)


def object_file_url(path):
    if not path or not isinstance(path, str):
        return None
    raw_path = html_lib.unescape(path).strip()
    if raw_path.startswith(("http://", "https://")):
        return normalize_url(raw_path)
    normalized_path = raw_path.lstrip("/")
    if normalized_path.startswith(("upload/", "public-service/")):
        return normalize_url(f"https://object.irc.gov.kh/irc-public/{normalized_path}")
    return None


def public_service_object_url(path):
    """Normalize the public-service page's relative storage links.

    The rendered page sometimes exposes links such as /2026/<id>.pdf, while the
    actual file is stored under object.irc.gov.kh/irc-public/public-service/.
    Next.js data already carries the full public-service/<year>/ path; this
    helper covers both representations.
    """
    exact = object_file_url(path)
    if exact and "/public-service/" in urlparse(exact).path:
        return exact
    if not path or not isinstance(path, str):
        return None
    normalized_path = html_lib.unescape(path).strip().lstrip("/")
    if re.match(
        r"^\d{4}/[^/?#]+\.(?:pdf|docx?|xlsx?|jpg|jpeg|png|webp)$",
        normalized_path,
        flags=re.IGNORECASE,
    ):
        return normalize_url(
            f"https://object.irc.gov.kh/irc-public/public-service/{normalized_path}"
        )
    return None


def management_asset_urls():
    return [normalize_url(urljoin(BASE_URL, path)) for path in MANAGEMENT_ASSET_PATHS]


def format_date(value):
    if not value:
        return None
    return value.replace("T", " ").replace("Z", " UTC")


def article_list_links(url, state):
    parsed = urlparse(url)
    if not parsed.path.endswith("/article/list"):
        return []
    category = (parse_qs(parsed.query).get("category") or [None])[0]
    if not category:
        return []
    lang_prefix = "/en" if parsed.path.startswith("/en") else ""
    links = []
    for key, item in state.items():
        if not isinstance(item, dict):
            continue
        if not key.startswith("articles("):
            continue
        if f'"category":"{category}"' not in key:
            continue
        try:
            total_pages = int(item.get("totalPages") or 0)
        except (TypeError, ValueError):
            total_pages = 0
        for page in range(1, total_pages + 1):
            links.append(
                normalize_url(
                    urljoin(BASE_URL, f"{lang_prefix}/article/list?category={category}&page={page}")
                )
            )
    return links


def storage_file_from_value(value, *, public_service_context=False):
    if isinstance(value, dict):
        return storage_file_from_value(
            value.get("file")
            or value.get("logo")
            or value.get("low")
            or value.get("medium")
            or value.get("placeholder")
            or value.get("micro"),
            public_service_context=public_service_context,
        )
    if not isinstance(value, str):
        return None
    if public_service_context:
        return public_service_object_url(value) or object_file_url(value)
    return object_file_url(value)


def file_size_from_value(value):
    if isinstance(value, dict):
        return value.get("fileSize") or value.get("size")
    return None


def file_name_from_value(value):
    if isinstance(value, dict):
        return value.get("fileName") or Path(str(value.get("file") or "")).name
    if isinstance(value, str):
        return Path(value).name
    return ""


def extract_structured_next_sections(url, state):
    parsed_path = urlparse(url).path.lower()
    public_service_context = "public-service" in parsed_path
    public_service_rows = []
    institution_rows = []
    links = []
    seen_public_service = set()
    seen_institutions = set()

    for item in iter_json_values(state):
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        file_value = item.get("file")
        file_url = storage_file_from_value(
            file_value,
            public_service_context=public_service_context,
        )
        if (
            title
            and file_value is not None
            and (
                public_service_context
                or (file_url and "/public-service/" in urlparse(file_url).path)
            )
        ):
            key = (str(title).strip(), file_url or "")
            if key not in seen_public_service:
                seen_public_service.add(key)
                public_service_rows.append(
                    {
                        "title": clean_text(str(title)),
                        "file_url": file_url,
                        "file_name": file_name_from_value(file_value),
                        "file_size": file_size_from_value(file_value),
                    }
                )
            if file_url:
                links.append(file_url)

        name = item.get("name")
        institution_type = item.get("type")
        logo_url = storage_file_from_value(item.get("logo"))
        if name and institution_type and logo_url:
            clean_name = clean_text(str(name))
            if clean_name.casefold() in {"name", "ឈ្មោះ"}:
                continue
            key = (clean_name, str(institution_type), logo_url)
            if key in seen_institutions:
                continue
            seen_institutions.add(key)
            institution_rows.append(
                {
                    "name": clean_name,
                    "type": str(institution_type),
                    "logo_url": logo_url,
                }
            )
            links.append(logo_url)

    lines = []
    if public_service_rows:
        lines.extend(
            [
                "សេវាសាធារណៈ",
                "Public Service",
                "Official public-service records and downloadable forms from the Insurance Regulator of Cambodia website.",
                "សំណួរពាក់ព័ន្ធ៖ សេវាសាធារណៈ, តារាងសេវាសាធារណៈ, ឯកសារតម្រូវ, កម្រៃសេវា, អាជ្ញាបណ្ណ, ភ្នាក់ងារធានារ៉ាប់រង។",
            ]
        )
        for index, row in enumerate(public_service_rows, 1):
            lines.append(f"{index}. {row['title']}")
            if row.get("file_url"):
                size = f" ({row['file_size']} bytes)" if row.get("file_size") else ""
                label = f" - {row['file_name']}" if row.get("file_name") else ""
                lines.append(f"Download file{label}: {row['file_url']}{size}")

    if institution_rows:
        grouped = {}
        for row in institution_rows:
            grouped.setdefault(row["type"], []).append(row)
        lines.extend(
            [
                "គ្រឹះស្ថានធានារ៉ាប់រង",
                "Insurance Institutions Directory",
                "Official insurance institutions directory scraped from the Insurance Regulator of Cambodia website.",
                "សំណួរពាក់ព័ន្ធ៖ គ្រឹះស្ថានធានារ៉ាប់រង, ក្រុមហ៊ុនធានារ៉ាប់រង, សមាគម, ជើងសាធានារ៉ាប់រង, ក្រុមហ៊ុនធានារ៉ាប់រងទូទៅ, ក្រុមហ៊ុនធានារ៉ាប់រងអាយុជីវិត, ក្រុមហ៊ុនធានារ៉ាប់រងខ្នាតតូច។",
                f"Total listed institutions: {len(institution_rows)}",
            ]
        )
        for institution_type in sorted(grouped):
            label = INSTITUTION_TYPE_LABELS.get(institution_type, institution_type)
            rows = sorted(grouped[institution_type], key=lambda row: row["name"])
            lines.append(f"{label}: {len(rows)}")
            for row in rows:
                lines.append(
                    f"- {row['name']} | Type: {institution_type} | Logo: {row['logo_url']}"
                )

    return {"text": clean_text("\n".join(lines)), "links": sorted(set(links))}


def extract_next_content(url, html):
    data = extract_next_json(html)
    if not data:
        return {"text": "", "links": []}

    state = data.get("props", {}).get("pageProps", {}).get("initialApolloState", {})
    lines = []
    links = article_list_links(url, state)
    structured_content = extract_structured_next_sections(url, state)
    links.extend(structured_content["links"])

    for key, item in state.items():
        if not isinstance(item, dict):
            continue

        typename = item.get("__typename", "")

        if typename == "ArticlePublic":
            title = item.get("title")
            category = item.get("category")
            publish_date = format_date(item.get("publishDate") or item.get("createdAt"))
            article_id = item.get("id")

            if title:
                lines.append(str(title))
            if category:
                lines.append(f"Category: {category}")
            if publish_date:
                lines.append(f"Publish date: {publish_date}")

            description = clean_text(item.get("description") or "")
            if description:
                lines.append(description)

            if article_id:
                lang_prefix = "/en" if urlparse(url).path.startswith("/en") else ""
                links.append(normalize_url(urljoin(BASE_URL, f"{lang_prefix}/article?post={article_id}")))

            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                content_type = content_item.get("type")
                content = content_item.get("content")
                if content_type:
                    lines.append(f"Content type: {content_type}")
                if isinstance(content, dict):
                    for file_key in ("file", "low", "medium", "placeholder", "micro"):
                        file_url = object_file_url(content.get(file_key))
                        if file_url:
                            links.append(file_url)
                            if file_key == "file":
                                size = content.get("fileSize")
                                size_text = f" ({size} bytes)" if size else ""
                                filename = content.get("fileName")
                                label = f" - {filename}" if filename else ""
                                lines.append(f"Download file{label}: {file_url}{size_text}")

        elif typename == "GraphType":
            graph_type = item.get("type")
            graph_data = item.get("data") or {}
            labels = graph_data.get("labels") or []
            datasets = graph_data.get("datasets") or []
            if graph_type == "premium":
                lines.append("Total Gross Premium")
                lines.append("បុព្វលាភធានារ៉ាប់រងដុល")
            elif graph_type == "density":
                lines.append("Insurance Density and Insurance Penetration")
            if labels and datasets:
                for index, year in enumerate(labels):
                    values = []
                    total = 0.0
                    numeric_total = True
                    for dataset in datasets:
                        label = dataset.get("label")
                        series = dataset.get("data") or []
                        if index >= len(series):
                            continue
                        value = series[index]
                        values.append(f"{label}: {value}")
                        try:
                            total += float(value)
                        except (TypeError, ValueError):
                            numeric_total = False
                    if values:
                        suffix = f"; Total: {int(total):,}" if numeric_total and graph_type == "premium" else ""
                        lines.append(f"{year}: " + "; ".join(values) + suffix)

    if structured_content["text"]:
        lines.append(structured_content["text"])

    return {"text": clean_text("\n".join(lines)), "links": sorted(set(links))}


def clean_text(text):
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line.lower() in DROP_TEXT:
            continue
        lines.append(line)
    return "\n".join(lines)


def is_file_url(url):
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in FILE_EXTENSIONS)


def is_image_url(url):
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_pdf_path(path):
    return str(path).lower().endswith(".pdf")


def is_docx_path(path):
    return str(path).lower().endswith(".docx")


def extract_pdf_text(path, timeout=90):
    if not shutil.which("pdftotext"):
        return ""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return clean_text(result.stdout)


def extract_docx_text(path):
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError:
        return ""
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(f"{namespace}p"):
        parts = [
            text_node.text or ""
            for text_node in paragraph.iter(f"{namespace}t")
            if text_node.text
        ]
        if parts:
            paragraphs.append("".join(parts))
    return clean_text("\n".join(paragraphs))


def extract_image_ocr_text(path, timeout=90):
    if not shutil.which("tesseract"):
        return ""
    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "khm+eng", "--psm", "6"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return clean_text(result.stdout)


def should_skip_url(url):
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_robots(session, timeout):
    robots = RobotFileParser()
    robots.set_url(urljoin(BASE_URL, "/robots.txt"))
    try:
        response = session.get(robots.url, timeout=timeout)
        if response.ok:
            robots.parse(response.text.splitlines())
        else:
            robots.parse([])
    except Exception:
        robots.parse([])
    return robots


def fetch_page(session, url, timeout):
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return None
    parser = Extractor()
    parser.feed(response.text)
    next_content = extract_next_content(url, response.text)
    links = []
    public_service_context = "public-service" in urlparse(url).path.lower()
    for href in parser.links + next_content["links"]:
        if public_service_context:
            absolute = (
                public_service_object_url(href)
                or object_file_url(href)
                or normalize_url(urljoin(url, href))
            )
        else:
            absolute = object_file_url(href) or normalize_url(urljoin(url, href))
        if not absolute.startswith(("http://", "https://")):
            continue
        links.append(absolute)
    text_parts = [clean_text("\n".join(parser.text_parts))]
    if next_content["text"]:
        text_parts.append(next_content["text"])
    return {
        "status_code": response.status_code,
        "content_type": content_type,
        "title": clean_text(" ".join(parser.title_parts)),
        "text": clean_text("\n\n".join(part for part in text_parts if part)),
        "links": sorted(set(links)),
    }


def download_file(session, url, max_file_mb, timeout):
    response = session.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    filename = safe_filename(url, content_type)
    output_path = FILE_ROOT / filename
    max_bytes = int(max_file_mb * 1024 * 1024)
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            written += len(chunk)
            if written > max_bytes:
                output_path.unlink(missing_ok=True)
                raise ValueError(f"file exceeds --max-file-mb={max_file_mb}: {url}")
            f.write(chunk)
    record = {
        "url": url,
        "path": str(output_path),
        "content_type": content_type,
        "size_bytes": written,
    }
    if is_pdf_path(output_path):
        record["sha256"] = file_sha256(output_path)
        duplicate_path = source_pdf_hashes().get(record["sha256"])
        if duplicate_path:
            record["duplicate_source_pdf"] = duplicate_path
            record["text_extraction_method"] = "skipped_duplicate_source_pdf"
            return record
        record["extracted_text"] = extract_pdf_text(output_path)
        record["text_extraction_method"] = "pdftotext" if record["extracted_text"] else None
    elif is_docx_path(output_path):
        record["extracted_text"] = extract_docx_text(output_path)
        record["text_extraction_method"] = "docx_xml" if record["extracted_text"] else None
    return record


def write_pipeline_document(page_record):
    url = page_record["url"]
    text = page_record.get("text", "").strip()
    if len(text) < 80:
        return None
    document_id = stable_id("website", url)
    category = infer_website_category(
        source_url=url,
        title=page_record.get("title"),
        text=text,
    )
    output = {
        "document_id": document_id,
        "category": category,
        "total_pages": 1,
        "merged_text": text,
        "source_url": url,
        "scraped_at": page_record["scraped_at"],
    }
    save_json(website_pipeline_path(MERGED_DOC_ROOT, category, document_id), output)
    return document_id


def write_file_pipeline_document(file_record, scraped_at, title=None, parent_url=None):
    if file_record.get("duplicate_source_pdf"):
        return None
    text = (file_record.get("extracted_text") or "").strip()
    if len(text) < 80:
        return None
    url = file_record["url"]
    document_id = stable_id("website_file", url)
    source_title = title or Path(file_record.get("path") or urlparse(url).path).name or url
    category = infer_website_category(
        source_url=url,
        parent_url=parent_url,
        title=source_title,
        text=text,
    )
    output = {
        "document_id": document_id,
        "category": category,
        "total_pages": 1,
        "merged_text": f"{source_title}\nSource file: {url}\n\n{text}",
        "source_url": url,
        "parent_url": parent_url,
        "scraped_at": scraped_at,
    }
    save_json(website_pipeline_path(MERGED_DOC_ROOT, category, document_id), output)
    return document_id


def prune_duplicate_source_pdf_documents(file_records):
    metadata = load_metadata()
    pruned = 0
    changed_metadata = False

    for file_record in file_records:
        path = Path(file_record.get("path") or "")
        duplicate_path = file_record.get("duplicate_source_pdf")
        if not duplicate_path and path.exists():
            duplicate_path = duplicate_source_pdf_path(path)
        if not duplicate_path:
            continue

        url = file_record.get("url")
        if not url:
            continue
        document_id = stable_id("website_file", url)
        removed_any = False
        for candidate in existing_website_pipeline_paths(document_id):
            if candidate.exists():
                candidate.unlink()
                removed_any = True
        if document_id in metadata:
            del metadata[document_id]
            changed_metadata = True
            removed_any = True
        if removed_any:
            pruned += 1

    if changed_metadata:
        save_json(METADATA_PATH, metadata)
    return pruned


def prune_management_image_documents():
    """Remove stale OCR documents created from management JPG/PNG assets.

    The management page links useful PDF biographies, but it also links portrait
    and PDF-page image exports. When image OCR is disabled, old image-derived
    documents must be removed from the pipeline or they can keep surfacing as
    noisy context after an embedding rebuild.
    """
    metadata = load_metadata()
    pruned = 0
    changed_metadata = False

    for url in management_asset_urls():
        if not is_image_url(url):
            continue
        document_id = stable_id("website_file", url)
        removed_any = False
        for candidate in existing_website_pipeline_paths(document_id):
            if candidate.exists():
                candidate.unlink()
                removed_any = True
        if document_id in metadata:
            del metadata[document_id]
            changed_metadata = True
            removed_any = True
        if removed_any:
            pruned += 1

    if changed_metadata:
        save_json(METADATA_PATH, metadata)
    return pruned


def update_document_metadata(page_records):
    metadata = load_metadata()
    changed = 0
    for page in page_records:
        document_id = page.get("document_id")
        if not document_id:
            continue
        title = page.get("title") or page["url"]
        category = infer_website_category(
            source_url=page.get("url"),
            parent_url=page.get("parent_url"),
            title=title,
            text=page.get("text"),
        )
        metadata[document_id] = {
            "document_name": title,
            "document_type": category,
            "release_date": page.get("scraped_at", "")[:10],
            "source_url": page["url"],
            "source_section": category,
        }
        changed += 1
    save_json(METADATA_PATH, metadata)
    return changed


def write_supplemental_documents(scraped_at=None):
    scraped_at = scraped_at or datetime.now(timezone.utc).isoformat()
    metadata = load_metadata()
    management_links = "\n".join(
        f"Official management profile asset: {url}" for url in management_asset_urls()
    )
    publication_catalog = build_publication_catalog_text()
    records = [
        {
            "document_id": "website_homepage_management_structure",
            "category": "home",
            "title": "Management Structure",
            "text": f"{MANAGEMENT_STRUCTURE_TEXT}\n\n{management_links}",
            "source_url": "https://www.irc.gov.kh/en",
        },
        {
            "document_id": "website_homepage_gross_premium_statistics",
            "category": "home",
            "title": "Total Gross Premium",
            "text": GROSS_PREMIUM_STATISTICS_TEXT,
            "source_url": "https://www.irc.gov.kh/en",
        },
    ]
    if publication_catalog:
        records.append(
            {
                "document_id": "website_publication_catalog",
                "category": "publication",
                "title": "បញ្ជីការបោះពុម្ពផ្សាយ និងរបាយការណ៍ប្រចាំឆ្នាំរបស់ ន.ធ.ក.",
                "text": publication_catalog,
                "source_url": "https://www.irc.gov.kh/en/article/list?category=publication&page=1",
            }
        )
    changed = 0
    for record in records:
        category = record["category"]
        output = {
            "document_id": record["document_id"],
            "category": category,
            "total_pages": 1,
            "merged_text": record["text"],
            "source_url": record["source_url"],
            "scraped_at": scraped_at,
        }
        save_json(
            website_pipeline_path(MERGED_DOC_ROOT, category, record["document_id"]),
            output,
        )
        metadata[record["document_id"]] = {
            "document_name": record["title"],
            "document_type": category,
            "release_date": scraped_at[:10],
            "source_url": record["source_url"],
            "source_section": category,
        }
        changed += 1
    save_json(METADATA_PATH, metadata)
    return changed


def build_publication_catalog_text():
    entries = {}

    for path in sorted(MERGED_DOC_ROOT.rglob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if doc.get("document_id") == "website_publication_catalog":
            continue
        text = doc.get("merged_text") or ""
        if "Category: publication" not in text:
            continue
        lines = [line.strip() for line in text.splitlines()]
        for idx, line in enumerate(lines):
            if line != "Category: publication" or idx == 0:
                continue
            title = lines[idx - 1].strip()
            if not title or title.lower() in {"publication", "ការបោះពុម្ពផ្សាយ"}:
                continue
            publish_date = ""
            if idx + 1 < len(lines) and lines[idx + 1].startswith("Publish date:"):
                publish_date = lines[idx + 1].replace("Publish date:", "", 1).strip()
            key = re.sub(r"\s+", " ", title).casefold()
            entries.setdefault(key, {"title": title, "publish_date": publish_date})

    if not entries:
        return ""

    def sort_key(item):
        value = item[1].get("publish_date") or ""
        return value

    sorted_entries = sorted(entries.items(), key=sort_key, reverse=True)
    lines = [
        "បញ្ជីការបោះពុម្ពផ្សាយ និងរបាយការណ៍ប្រចាំឆ្នាំរបស់ ន.ធ.ក.",
        "IRC Publication Catalog and Annual Reports",
        "Official publication category records scraped from the Insurance Regulator of Cambodia website.",
        "សំណួរពាក់ព័ន្ធ៖ ការបោះពុម្ពផ្សាយ, publication, របាយការណ៍ប្រចាំឆ្នាំ, annual report, ផែនការយុទ្ធសាស្ត្រ, សទ្ទានុក្រម។",
        "",
        "របាយការណ៍ប្រចាំឆ្នាំ / Annual reports:",
    ]

    annual = [
        item
        for _key, item in sorted_entries
        if "annual report" in item["title"].casefold()
        or "របាយការណ៍ប្រចាំឆ្នាំ" in item["title"]
    ]
    other = [item for _key, item in sorted_entries if item not in annual]

    for item in annual:
        date = f" - Publish date: {item['publish_date']}" if item.get("publish_date") else ""
        lines.append(f"- {item['title']}{date}")

    lines.extend(["", "ការបោះពុម្ពផ្សាយផ្សេងទៀត / Other publications:"])
    for item in other[:120]:
        date = f" - Publish date: {item['publish_date']}" if item.get("publish_date") else ""
        lines.append(f"- {item['title']}{date}")

    return "\n".join(lines)


def process_file_record(file_record, scraped_at, page_records, title=None, parent_url=None):
    document_id = write_file_pipeline_document(
        file_record,
        scraped_at=scraped_at,
        title=title,
        parent_url=parent_url,
    )
    if not document_id:
        return None
    record = {
        "url": file_record["url"],
        "title": title or Path(file_record.get("path") or file_record["url"]).name,
        "scraped_at": scraped_at,
        "document_id": document_id,
        "parent_url": parent_url,
    }
    page_records.append(record)
    return record


def download_management_assets(session, args, scraped_at, page_records, file_records, errors):
    for url in management_asset_urls():
        if is_image_url(url) and not args.ocr_management_images:
            continue
        try:
            print(f"Downloading management asset: {url}")
            file_record = download_file(session, url, args.max_file_mb, args.timeout)
            if is_image_url(url):
                file_record["extracted_text"] = extract_image_ocr_text(file_record["path"])
                file_record["text_extraction_method"] = (
                    "tesseract khm+eng" if file_record.get("extracted_text") else None
                )
            file_records.append(file_record)
            title = f"Management profile background - {Path(urlparse(url).path).name}"
            process_file_record(
                file_record,
                scraped_at=scraped_at,
                page_records=page_records,
                title=title,
                parent_url="https://www.irc.gov.kh/en",
            )
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})


def finalize_cached_pages(args):
    page_records = []
    for path in sorted(PAGE_ROOT.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            page = json.load(f)
        page["document_id"] = write_pipeline_document(page)
        save_json(path, page)
        page_records.append(page)
    updated = update_document_metadata(page_records)
    supplemental = write_supplemental_documents()
    file_records = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(FILE_ROOT.glob("*"))
        if path.is_file()
    ]
    pruned_duplicates = prune_duplicate_source_pdf_documents(file_records)
    pruned_management_images = (
        prune_management_image_documents() if not args.ocr_management_images else 0
    )
    manifest = {
        "base_url": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_seen": len(page_records),
        "pages_saved": len(page_records),
        "pipeline_documents": updated + supplemental,
        "files_downloaded": len(file_records),
        "duplicate_source_pdfs_pruned": pruned_duplicates,
        "management_image_documents_pruned": pruned_management_images,
        "errors": [],
        "files": file_records,
        "finalized_from_cache": True,
    }
    save_json(SCRAPE_ROOT / "manifest.json", manifest)
    return manifest


def crawl(args):
    session = make_session()
    robots = build_robots(session, args.timeout)
    queue = [normalize_url(seed) for seed in args.seed]
    seen = set()
    attempted_pages = 0
    page_records = []
    file_records = []
    errors = []

    while queue and attempted_pages < args.max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        parsed = urlparse(url)
        if parsed.netloc not in INTERNAL_HOSTS:
            continue
        if should_skip_url(url) or is_file_url(url):
            continue
        if args.respect_robots and not robots.can_fetch(USER_AGENT, url):
            errors.append({"url": url, "error": "blocked by robots.txt"})
            continue

        attempted_pages += 1
        print(f"Fetching page: {url}")
        try:
            page = fetch_page(session, url, args.timeout)
            time.sleep(args.delay)
            if not page:
                continue
            scraped_at = datetime.now(timezone.utc).isoformat()
            record = {"url": url, "scraped_at": scraped_at, **page}
            record["document_id"] = write_pipeline_document(record)
            save_json(PAGE_ROOT / f"{stable_id('page', url)}.json", record)
            page_records.append(record)

            for link in page["links"]:
                link_host = urlparse(link).netloc
                if args.download_files and link_host in FILE_HOSTS and is_file_url(link):
                    if is_image_url(link) and not args.download_images:
                        continue
                    if args.respect_robots and not robots.can_fetch(USER_AGENT, link):
                        errors.append({"url": link, "error": "blocked by robots.txt"})
                        continue
                    if any(item["url"] == link for item in file_records):
                        continue
                    print(f"Downloading file: {link}")
                    try:
                        file_record = download_file(session, link, args.max_file_mb, args.timeout)
                        file_records.append(file_record)
                        file_title = f"{record.get('title') or url} - {Path(urlparse(link).path).name}"
                        process_file_record(
                            file_record,
                            scraped_at=scraped_at,
                            page_records=page_records,
                            title=file_title,
                            parent_url=url,
                        )
                    except Exception as exc:
                        errors.append({"url": link, "error": str(exc)})
                    time.sleep(args.delay)
                elif link_host in INTERNAL_HOSTS and not should_skip_url(link):
                    if is_file_url(link):
                        continue
                    if link not in seen and link not in queue:
                        queue.append(link)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    supplemental = write_supplemental_documents()
    if args.download_files and args.download_management_assets:
        download_management_assets(
            session=session,
            args=args,
            scraped_at=datetime.now(timezone.utc).isoformat(),
            page_records=page_records,
            file_records=file_records,
            errors=errors,
        )

    updated = update_document_metadata(page_records)
    pruned_duplicates = prune_duplicate_source_pdf_documents(file_records)
    pruned_management_images = (
        prune_management_image_documents() if not args.ocr_management_images else 0
    )
    manifest = {
        "base_url": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_seen": len(seen),
        "pages_saved": len(page_records),
        "pipeline_documents": updated + supplemental,
        "files_downloaded": len(file_records),
        "duplicate_source_pdfs_pruned": pruned_duplicates,
        "management_image_documents_pruned": pruned_management_images,
        "errors": errors,
        "files": file_records,
    }
    save_json(SCRAPE_ROOT / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape IRC website pages for the RAG corpus.")
    parser.add_argument("--seed", action="append")
    parser.add_argument("--max-pages", type=int, default=1500)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--download-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--download-management-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ocr-management-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--respect-robots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finalize-cache", action="store_true")
    args = parser.parse_args()
    if not args.seed:
        args.seed = DEFAULT_SEEDS
    return args


if __name__ == "__main__":
    args = parse_args()
    result = finalize_cached_pages(args) if args.finalize_cache else crawl(args)
    print("\n========== SUMMARY ==========")
    print(f"Pages saved        : {result['pages_saved']}")
    print(f"Pipeline documents : {result['pipeline_documents']}")
    print(f"Files downloaded   : {result['files_downloaded']}")
    print(f"Errors             : {len(result['errors'])}")
    print(f"Manifest           : {SCRAPE_ROOT / 'manifest.json'}")
