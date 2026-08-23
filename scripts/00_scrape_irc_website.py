import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

try:
    import requests
except ImportError:
    requests = None


BASE_URL = "https://www.irc.gov.kh/"
USER_AGENT = "Khmer-RAG-Insurance-Chatbot dataset scraper (+research corpus build)"

SCRAPE_ROOT = Path("data/00_irc_website_scrape")
PAGE_ROOT = SCRAPE_ROOT / "pages"
FILE_ROOT = SCRAPE_ROOT / "files"
WEBSITE_DOC_ROOT = Path("data/05_merged_documents/website")
METADATA_PATH = Path("data/document_metadata.json")

LIST_CATEGORIES = [
    "laws",
    "sub-decrees",
    "prakas",
    "guidance",
    "publication",
    "statistics-monthly",
    "news",
    "event",
    "education",
    "announcement",
]
DEFAULT_SEEDS = [
    "https://www.irc.gov.kh/",
    "https://www.irc.gov.kh/en",
]
for category in LIST_CATEGORIES:
    for page in range(1, 26):
        DEFAULT_SEEDS.append(
            f"https://www.irc.gov.kh/article/list?category={category}&page={page}"
        )
        DEFAULT_SEEDS.append(
            f"https://www.irc.gov.kh/en/article/list?category={category}&page={page}"
        )

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
SKIP_EXTENSIONS = {".css", ".js", ".ico", ".svg", ".woff", ".woff2", ".ttf", ".map"}
DROP_TEXT = {
    "image",
    "english",
    "ភាសាខ្មែរ",
    "ទំព័រដើម",
    "home",
    "follow us",
}


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
    links = []
    for href in parser.links:
        absolute = normalize_url(urljoin(url, href))
        if not absolute.startswith(("http://", "https://")):
            continue
        links.append(absolute)
    return {
        "status_code": response.status_code,
        "content_type": content_type,
        "title": clean_text(" ".join(parser.title_parts)),
        "text": clean_text("\n".join(parser.text_parts)),
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
    return {
        "url": url,
        "path": str(output_path),
        "content_type": content_type,
        "size_bytes": written,
    }


def write_pipeline_document(page_record):
    url = page_record["url"]
    text = page_record.get("text", "").strip()
    if len(text) < 80:
        return None
    document_id = stable_id("website", url)
    output = {
        "document_id": document_id,
        "category": "website",
        "total_pages": 1,
        "merged_text": text,
        "source_url": url,
        "scraped_at": page_record["scraped_at"],
    }
    save_json(WEBSITE_DOC_ROOT / f"{document_id}.json", output)
    return document_id


def update_document_metadata(page_records):
    metadata = load_metadata()
    changed = 0
    for page in page_records:
        document_id = page.get("document_id")
        if not document_id:
            continue
        title = page.get("title") or page["url"]
        metadata[document_id] = {
            "document_name": title,
            "document_type": "website",
            "release_date": page.get("scraped_at", "")[:10],
            "source_url": page["url"],
        }
        changed += 1
    save_json(METADATA_PATH, metadata)
    return changed


def finalize_cached_pages():
    page_records = []
    for path in sorted(PAGE_ROOT.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            page = json.load(f)
        page["document_id"] = write_pipeline_document(page)
        save_json(path, page)
        page_records.append(page)
    updated = update_document_metadata(page_records)
    file_records = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in sorted(FILE_ROOT.glob("*"))
        if path.is_file()
    ]
    manifest = {
        "base_url": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_seen": len(page_records),
        "pages_saved": len(page_records),
        "pipeline_documents": updated,
        "files_downloaded": len(file_records),
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
                if link_host in INTERNAL_HOSTS and not should_skip_url(link):
                    if is_file_url(link):
                        continue
                    if link not in seen and link not in queue:
                        queue.append(link)
                elif args.download_files and link_host in FILE_HOSTS and is_file_url(link):
                    if args.respect_robots and not robots.can_fetch(USER_AGENT, link):
                        errors.append({"url": link, "error": "blocked by robots.txt"})
                        continue
                    if any(item["url"] == link for item in file_records):
                        continue
                    print(f"Downloading file: {link}")
                    try:
                        file_records.append(download_file(session, link, args.max_file_mb, args.timeout))
                    except Exception as exc:
                        errors.append({"url": link, "error": str(exc)})
                    time.sleep(args.delay)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    updated = update_document_metadata(page_records)
    manifest = {
        "base_url": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_seen": len(seen),
        "pages_saved": len(page_records),
        "pipeline_documents": updated,
        "files_downloaded": len(file_records),
        "errors": errors,
        "files": file_records,
    }
    save_json(SCRAPE_ROOT / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape IRC website pages for the RAG corpus.")
    parser.add_argument("--seed", action="append", default=DEFAULT_SEEDS)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--download-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--respect-robots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finalize-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = finalize_cached_pages() if args.finalize_cache else crawl(args)
    print("\n========== SUMMARY ==========")
    print(f"Pages saved        : {result['pages_saved']}")
    print(f"Pipeline documents : {result['pipeline_documents']}")
    print(f"Files downloaded   : {result['files_downloaded']}")
    print(f"Errors             : {len(result['errors'])}")
    print(f"Manifest           : {SCRAPE_ROOT / 'manifest.json'}")
