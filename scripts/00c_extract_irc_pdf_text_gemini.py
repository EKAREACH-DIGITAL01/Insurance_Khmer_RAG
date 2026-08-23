import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image


MANIFEST_PATH = Path("data/00_irc_website_scrape/manifest.json")
METADATA_PATH = Path("data/document_metadata.json")
IMAGE_ROOT = Path("data/02_page_images_irc_gemini")
RAW_ROOT = Path("data/03_raw_pages_irc_gemini")
WEBSITE_DOC_ROOT = Path("data/05_merged_documents/website")

DEFAULT_MODEL = "gemini-3.5-flash"
OCR_PROMPT = """Extract all visible text from this PDF page exactly as written.
Preserve Khmer text, English text, numbers, tables, line breaks, headings, and footnotes when possible.
Do not translate, summarize, explain, correct spelling, or add text that is not visible.
If a page has no readable text, return an empty response."""


def stable_id(prefix, value):
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_pdf_record(record):
    content_type = (record.get("content_type") or "").lower()
    path = (record.get("path") or "").lower()
    url = (record.get("url") or "").lower()
    return "pdf" in content_type or path.endswith(".pdf") or url.endswith(".pdf")


def load_pdf_records(only_url_contains=None, limit=None):
    manifest = load_json(MANIFEST_PATH, {})
    records = []
    for record in manifest.get("files", []):
        if not is_pdf_record(record):
            continue
        pdf_path = Path(record.get("path") or "")
        if not pdf_path.exists():
            continue
        url = record.get("url") or str(pdf_path)
        if only_url_contains and only_url_contains not in url:
            continue
        records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def title_from_record(record, metadata):
    url = record.get("url") or ""
    document_id = stable_id("website_file", url)
    existing = metadata.get(document_id, {})
    if existing.get("document_name"):
        return existing["document_name"]
    parsed_name = Path(urlparse(url).path).name
    local_name = Path(record.get("path") or "").name
    return parsed_name or local_name or document_id


def page_image_path(doc_id, page_number):
    return IMAGE_ROOT / doc_id / f"page_{page_number:04d}.png"


def raw_page_path(doc_id, page_number):
    return RAW_ROOT / doc_id / f"page_{page_number:04d}.json"


def get_page_count(pdf_path):
    try:
        return int(pdfinfo_from_path(str(pdf_path)).get("Pages", 0))
    except Exception as exc:
        print(f"Could not read page count for {pdf_path}: {exc}")
        return 0


def convert_missing_pages(pdf_path, doc_id, dpi, max_pages=None):
    total_pages = get_page_count(pdf_path)
    if max_pages:
        total_pages = min(total_pages, max_pages)
    if total_pages < 1:
        return []

    IMAGE_ROOT.joinpath(doc_id).mkdir(parents=True, exist_ok=True)
    page_paths = []
    for page_number in range(1, total_pages + 1):
        output_path = page_image_path(doc_id, page_number)
        page_paths.append(output_path)
        if output_path.exists():
            continue
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            fmt="png",
            single_file=True,
        )
        if not images:
            continue
        images[0].save(output_path, "PNG")
        print(f"Converted page image: {output_path}")
    return page_paths


def build_clients(api_keys):
    clients = []
    for key in api_keys:
        key = key.strip()
        if key:
            clients.append(genai.Client(api_key=key))
    if not clients:
        raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEYS in your environment/.env file.")
    return clients


def generate_ocr_response(client, model, image):
    return client.models.generate_content(
        model=model,
        contents=[OCR_PROMPT, image],
        config={"temperature": 0},
    )


def extract_page_with_timeout(executor, client, model, image_path, timeout):
    with Image.open(image_path) as image:
        future = executor.submit(generate_ocr_response, client, model, image)
        return future.result(timeout=timeout)


def ocr_pages(page_paths, doc_id, clients, model, request_timeout):
    processed = 0
    skipped = 0
    failed = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        for image_path in page_paths:
            page_number = int(image_path.stem.split("_")[1])
            output_path = raw_page_path(doc_id, page_number)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                skipped += 1
                continue

            client = clients[(processed + failed) % len(clients)]
            print(f"Gemini OCR: {doc_id} page {page_number}")
            start = time.time()
            try:
                response = extract_page_with_timeout(
                    executor,
                    client,
                    model,
                    image_path,
                    request_timeout,
                )
            except concurrent.futures.TimeoutError:
                print(f"Timeout after {request_timeout}s: {doc_id} page {page_number}")
                failed += 1
                continue
            except Exception as exc:
                print(f"Gemini error on {doc_id} page {page_number}: {exc}")
                failed += 1
                continue

            save_json(
                output_path,
                {
                    "document_id": doc_id,
                    "page": page_number,
                    "raw_text": response.text or "",
                    "model": model,
                    "usage_metadata": str(getattr(response, "usage_metadata", "")),
                },
            )
            processed += 1
            print(f"Saved OCR page in {time.time() - start:.2f}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return processed, skipped, failed


def merge_ocr_document(record, model, scraped_at):
    url = record.get("url") or ""
    doc_id = stable_id("website_file", url)
    raw_pages = sorted((RAW_ROOT / doc_id).glob("page_*.json"))
    page_texts = []
    for page_path in raw_pages:
        page_data = load_json(page_path, {})
        raw_text = (page_data.get("raw_text") or "").strip()
        if raw_text:
            page_texts.append(f"[Page {page_data.get('page')}]\n{raw_text}")
    merged_body = "\n\n".join(page_texts).strip()
    if len(merged_body) < 80:
        return None

    metadata = load_json(METADATA_PATH, {})
    title = title_from_record(record, metadata)
    output = {
        "document_id": doc_id,
        "category": "website",
        "total_pages": len(raw_pages),
        "merged_text": (
            f"{title}\n"
            f"Source file: {url}\n"
            f"OCR method: Google Gemini API ({model})\n\n"
            f"{merged_body}"
        ),
        "source_url": url,
        "scraped_at": scraped_at,
    }
    save_json(WEBSITE_DOC_ROOT / f"{doc_id}.json", output)

    metadata[doc_id] = {
        "document_name": title,
        "document_type": "website",
        "release_date": scraped_at[:10],
        "source_url": url,
        "ocr_method": f"Google Gemini API ({model})",
    }
    save_json(METADATA_PATH, metadata)
    return doc_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR IRC scraped PDFs with Gemini and merge them into the RAG website corpus."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--limit-pdfs", type=int)
    parser.add_argument("--max-pages-per-pdf", type=int)
    parser.add_argument("--only-url-contains")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    key_text = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    api_keys = [key.strip() for key in key_text.split(",") if key.strip()]
    clients = [] if args.convert_only or args.merge_only else build_clients(api_keys)

    records = load_pdf_records(
        only_url_contains=args.only_url_contains,
        limit=args.limit_pdfs,
    )
    print(f"PDF records selected: {len(records)}")

    scraped_at = datetime.now(timezone.utc).isoformat()
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    merged_docs = 0

    for index, record in enumerate(records, start=1):
        url = record.get("url") or ""
        pdf_path = Path(record.get("path") or "")
        doc_id = stable_id("website_file", url)
        print(f"\n[{index}/{len(records)}] {pdf_path.name}")
        print(url)

        if args.merge_only:
            if merge_ocr_document(record, model=args.model, scraped_at=scraped_at):
                merged_docs += 1
            continue

        page_paths = convert_missing_pages(
            pdf_path=pdf_path,
            doc_id=doc_id,
            dpi=args.dpi,
            max_pages=args.max_pages_per_pdf,
        )

        if args.convert_only:
            continue

        processed, skipped, failed = ocr_pages(
            page_paths=page_paths,
            doc_id=doc_id,
            clients=clients,
            model=args.model,
            request_timeout=args.request_timeout,
        )
        total_processed += processed
        total_skipped += skipped
        total_failed += failed

        if merge_ocr_document(record, model=args.model, scraped_at=scraped_at):
            merged_docs += 1

    print("\n========== GEMINI IRC PDF OCR SUMMARY ==========")
    print(f"PDFs selected    : {len(records)}")
    print(f"Pages processed  : {total_processed}")
    print(f"Pages skipped    : {total_skipped}")
    print(f"Pages failed     : {total_failed}")
    print(f"Merged documents : {merged_docs}")
    print("Next: rerun scripts/05_clean_documents.py through scripts/08_generate_embeddings.py.")


if __name__ == "__main__":
    main()
