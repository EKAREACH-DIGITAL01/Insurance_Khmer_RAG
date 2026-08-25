import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES, infer_website_category


MANIFEST_PATH = Path("data/00_irc_website_scrape/manifest.json")
METADATA_PATH = Path("data/document_metadata.json")
SOURCE_PDF_ROOT = Path("data/01_source_documents")
IMAGE_ROOT = Path("data/02_page_images_irc_gemini")
RAW_ROOT = Path("data/03_raw_pages_irc_gemini")
MERGED_DOC_ROOT = Path("data/05_merged_documents")

DEFAULT_MODEL = "gemini-3.5-flash"
OCR_PROMPT = """Extract all visible text from this PDF page exactly as written.
Preserve Khmer text, English text, numbers, tables, line breaks, headings, and footnotes when possible.
Do not translate, summarize, explain, correct spelling, or add text that is not visible.
If a page has no readable text, return an empty response."""


def stable_id(prefix, value):
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_pdf_hashes():
    hashes = {}
    for path in sorted(SOURCE_PDF_ROOT.rglob("*.pdf")):
        try:
            hashes[file_sha256(path)] = str(path)
        except OSError:
            continue
    return hashes


def duplicate_source_pdf_path(pdf_path, source_hashes):
    try:
        return source_hashes.get(file_sha256(pdf_path))
    except OSError:
        return None


def load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def website_pipeline_path(category, document_id):
    category = category if category in WEBSITE_DERIVED_CATEGORIES else "home"
    return MERGED_DOC_ROOT / category / f"{document_id}.json"


def is_pdf_record(record):
    content_type = (record.get("content_type") or "").lower()
    path = (record.get("path") or "").lower()
    url = (record.get("url") or "").lower()
    return "pdf" in content_type or path.endswith(".pdf") or url.endswith(".pdf")


def load_pdf_records(
    only_url_contains=None,
    limit=None,
    shard_count=1,
    shard_index=0,
    skip_first_pdfs=0,
):
    manifest = load_json(MANIFEST_PATH, {})
    original_pdf_hashes = source_pdf_hashes()
    all_records = []
    duplicate_count = 0
    for record in manifest.get("files", []):
        if not is_pdf_record(record):
            continue
        pdf_path = Path(record.get("path") or "")
        if not pdf_path.exists():
            continue
        duplicate_path = record.get("duplicate_source_pdf") or duplicate_source_pdf_path(
            pdf_path,
            original_pdf_hashes,
        )
        if duplicate_path:
            duplicate_count += 1
            continue
        url = record.get("url") or str(pdf_path)
        if only_url_contains and only_url_contains not in url:
            continue
        all_records.append(record)
    if duplicate_count:
        print(f"Skipped exact duplicate source PDFs: {duplicate_count}")
    if skip_first_pdfs:
        all_records = all_records[skip_first_pdfs:]
    records = [
        record
        for index, record in enumerate(all_records)
        if index % shard_count == shard_index
    ]
    if limit:
        records = records[:limit]
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


def batched_ranges(page_numbers, batch_size):
    batch = []
    for page_number in page_numbers:
        if not batch:
            batch = [page_number, page_number]
            continue
        if page_number == batch[1] + 1 and (batch[1] - batch[0] + 1) < batch_size:
            batch[1] = page_number
            continue
        yield tuple(batch)
        batch = [page_number, page_number]
    if batch:
        yield tuple(batch)


def convert_missing_pages(
    pdf_path,
    doc_id,
    dpi,
    max_pages=None,
    batch_size=20,
    thread_count=4,
):
    total_pages = get_page_count(pdf_path)
    if max_pages:
        total_pages = min(total_pages, max_pages)
    if total_pages < 1:
        return []

    IMAGE_ROOT.joinpath(doc_id).mkdir(parents=True, exist_ok=True)
    page_paths = [page_image_path(doc_id, page_number) for page_number in range(1, total_pages + 1)]
    missing_pages = [
        page_number
        for page_number, output_path in enumerate(page_paths, start=1)
        if not output_path.exists()
    ]
    for first_page, last_page in batched_ranges(missing_pages, batch_size):
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt="png",
            thread_count=thread_count,
        )
        for offset, image in enumerate(images):
            page_number = first_page + offset
            output_path = page_image_path(doc_id, page_number)
            image.save(output_path, "PNG")
        print(f"Converted page images: {doc_id} pages {first_page}-{last_page}")
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


class GeminiClientPool:
    def __init__(self, clients):
        self.clients = list(clients)
        self.cursor = 0
        self.quota_disabled = 0

    def __len__(self):
        return len(self.clients)

    def next_client(self):
        if not self.clients:
            raise RuntimeError("No Gemini clients remain available.")
        client = self.clients[self.cursor % len(self.clients)]
        self.cursor = (self.cursor + 1) % len(self.clients)
        return client

    def disable_for_quota(self, client):
        for index, existing in enumerate(self.clients):
            if existing is client:
                del self.clients[index]
                self.quota_disabled += 1
                if self.clients:
                    self.cursor %= len(self.clients)
                else:
                    self.cursor = 0
                return


def batch_ocr_prompt(page_numbers):
    page_list = ", ".join(str(page_number) for page_number in page_numbers)
    return f"""Extract all visible text from these PDF page images exactly as written.
Preserve Khmer text, English text, numbers, tables, line breaks, headings, and footnotes when possible.
Do not translate, summarize, explain, correct spelling, or add text that is not visible.
Return one section for each page image using this exact marker before the text:
<<<PAGE number>>>
The page numbers are: {page_list}.
If a page has no readable text, include its marker and leave the section empty."""


def generate_ocr_response(client, model, images, page_numbers):
    return client.models.generate_content(
        model=model,
        contents=[batch_ocr_prompt(page_numbers), *images],
        config={"temperature": 0},
    )


def is_quota_error(exc):
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def retry_delay_seconds(exc):
    message = str(exc)
    patterns = [
        r"retryDelay['\"]?:\s*['\"]?([0-9.]+)s",
        r"Please retry in\s+([0-9.]+)s",
        r"retry in\s+([0-9.]+)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def is_daily_quota_error(exc):
    return "GenerateRequestsPerDay" in str(exc)


def is_retryable_rate_limit(exc):
    message = str(exc)
    return (
        "GenerateRequestsPerMinute" in message
        or ("GenerateRequestsPerDay" not in message and "retryDelay" in message)
        or ("GenerateRequestsPerDay" not in message and "Please retry in" in message)
    )


def is_transient_service_error(exc):
    message = str(exc)
    return "503" in message or "UNAVAILABLE" in message or "high demand" in message


def is_model_unavailable_error(exc):
    message = str(exc)
    return (
        "404" in message
        and ("NOT_FOUND" in message or "no longer available" in message)
        and "model" in message.lower()
    )


def parse_batched_ocr_text(text, page_numbers):
    page_numbers = list(page_numbers)
    parsed = {page_number: "" for page_number in page_numbers}
    pattern = re.compile(r"<<<\s*PAGE\s+(\d+)\s*>>>", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text or ""))
    if not matches:
        if page_numbers:
            parsed[page_numbers[0]] = (text or "").strip()
        return parsed

    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        if page_number not in parsed:
            continue
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed[page_number] = text[match.end() : next_start].strip()
    return parsed


def extract_batch_with_timeout(executor, client, model, image_paths, page_numbers, timeout):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.copy())
    future = executor.submit(generate_ocr_response, client, model, images, page_numbers)
    try:
        return future.result(timeout=timeout)
    finally:
        for image in images:
            image.close()
        executor.shutdown(wait=False, cancel_futures=True)


def missing_page_batches(page_paths, batch_size):
    batch = []
    for image_path in page_paths:
        page_number = int(image_path.stem.split("_")[1])
        output_path = raw_page_path(image_path.parent.name, page_number)
        if output_path.exists():
            yield [image_path], True
            continue
        batch.append(image_path)
        if len(batch) >= batch_size:
            yield batch, False
            batch = []
    if batch:
        yield batch, False


def ocr_pages(
    page_paths,
    doc_id,
    client_pool,
    model,
    request_timeout,
    max_consecutive_quota_errors,
    ocr_batch_size,
):
    processed = 0
    skipped = 0
    failed = 0
    consecutive_quota_errors = 0
    stopped_for_quota = False
    for image_batch, already_done in missing_page_batches(page_paths, ocr_batch_size):
        if already_done:
            skipped += 1
            continue

        page_numbers = [int(image_path.stem.split("_")[1]) for image_path in image_batch]
        for page_number in page_numbers:
            raw_page_path(doc_id, page_number).parent.mkdir(parents=True, exist_ok=True)

        response = None
        while response is None:
            if len(client_pool) < 1:
                print("Stopping OCR because all configured Gemini keys hit quota.")
                stopped_for_quota = True
                failed += len(image_batch)
                break

            client = client_pool.next_client()
            page_label = (
                f"page {page_numbers[0]}"
                if len(page_numbers) == 1
                else f"pages {page_numbers[0]}-{page_numbers[-1]}"
            )
            print(f"Gemini OCR: {doc_id} {page_label}")
            start = time.time()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                response = extract_batch_with_timeout(
                    executor,
                    client,
                    model,
                    image_batch,
                    page_numbers,
                    request_timeout,
                )
            except concurrent.futures.TimeoutError:
                print(f"Timeout after {request_timeout}s: {doc_id} {page_label}")
                consecutive_quota_errors = 0
                failed += len(image_batch)
                break
            except Exception as exc:
                print(f"Gemini error on {doc_id} {page_label}: {exc}")
                if is_quota_error(exc):
                    if is_daily_quota_error(exc):
                        client_pool.disable_for_quota(client)
                        consecutive_quota_errors += 1
                        print(
                            "Gemini key hit daily quota; "
                            f"{len(client_pool)} key(s) remain in this worker."
                        )
                        if consecutive_quota_errors >= max_consecutive_quota_errors:
                            print(
                                "Stopping OCR after "
                                f"{consecutive_quota_errors} consecutive quota errors."
                            )
                            stopped_for_quota = True
                            failed += len(image_batch)
                            break
                        continue
                    if is_retryable_rate_limit(exc):
                        delay = retry_delay_seconds(exc)
                        sleep_for = max(5.0, min((delay or 10.0) + 1.0, 60.0))
                        print(
                            "Gemini rate limit is temporary; "
                            f"sleeping {sleep_for:.1f}s before retry."
                        )
                        time.sleep(sleep_for)
                        continue
                    client_pool.disable_for_quota(client)
                    consecutive_quota_errors += 1
                    if consecutive_quota_errors >= max_consecutive_quota_errors:
                        print(
                            "Stopping OCR after "
                            f"{consecutive_quota_errors} consecutive quota errors."
                        )
                        stopped_for_quota = True
                        failed += len(image_batch)
                        break
                    continue

                if is_transient_service_error(exc):
                    consecutive_quota_errors += 1
                    if consecutive_quota_errors >= max_consecutive_quota_errors:
                        print(
                            "Stopping OCR after "
                            f"{consecutive_quota_errors} consecutive transient errors."
                        )
                        failed += len(image_batch)
                        break
                    sleep_for = 10.0
                    print(
                        "Gemini service is temporarily unavailable; "
                        f"sleeping {sleep_for:.1f}s before retry."
                    )
                    time.sleep(sleep_for)
                    continue

                if is_model_unavailable_error(exc):
                    client_pool.disable_for_quota(client)
                    consecutive_quota_errors += 1
                    print(
                        "Gemini key cannot use this model; "
                        f"{len(client_pool)} key(s) remain in this worker."
                    )
                    if consecutive_quota_errors >= max_consecutive_quota_errors:
                        print(
                            "Stopping OCR after "
                            f"{consecutive_quota_errors} consecutive model access errors."
                        )
                        failed += len(image_batch)
                        break
                    continue

                consecutive_quota_errors = 0
                failed += len(image_batch)
                break

        if response is None:
            if stopped_for_quota:
                break
            continue

        page_texts = parse_batched_ocr_text(response.text or "", page_numbers)
        for page_number in page_numbers:
            save_json(
                raw_page_path(doc_id, page_number),
                {
                    "document_id": doc_id,
                    "page": page_number,
                    "raw_text": page_texts.get(page_number, ""),
                    "model": model,
                    "usage_metadata": str(getattr(response, "usage_metadata", "")),
                },
            )
        processed += len(page_numbers)
        consecutive_quota_errors = 0
        print(f"Saved OCR {page_label} in {time.time() - start:.2f}s")

    return processed, skipped, failed, stopped_for_quota


def merge_ocr_document(record, model, scraped_at):
    url = record.get("url") or ""
    doc_id = stable_id("website_file", url)
    raw_pages = sorted((RAW_ROOT / doc_id).glob("page_*.json"))
    page_texts = []
    page_models = set()
    for page_path in raw_pages:
        page_data = load_json(page_path, {})
        page_model = (page_data.get("model") or "").strip()
        if page_model:
            page_models.add(page_model)
        raw_text = (page_data.get("raw_text") or "").strip()
        if raw_text:
            page_texts.append(f"[Page {page_data.get('page')}]\n{raw_text}")
    merged_body = "\n\n".join(page_texts).strip()
    if len(merged_body) < 80:
        return None

    metadata = load_json(METADATA_PATH, {})
    title = title_from_record(record, metadata)
    existing_metadata = metadata.get(doc_id, {})
    category = existing_metadata.get("source_section") or existing_metadata.get("document_type")
    if category not in WEBSITE_DERIVED_CATEGORIES:
        category = infer_website_category(
            source_url=url,
            parent_url=record.get("parent_url"),
            title=title,
            text=merged_body,
        )
    model_summary = ", ".join(sorted(page_models)) if page_models else model
    output = {
        "document_id": doc_id,
        "category": category,
        "total_pages": len(raw_pages),
        "merged_text": (
            f"{title}\n"
            f"Source file: {url}\n"
            f"OCR method: Google Gemini API ({model_summary})\n\n"
            f"{merged_body}"
        ),
        "source_url": url,
        "parent_url": record.get("parent_url"),
        "scraped_at": scraped_at,
    }
    save_json(website_pipeline_path(category, doc_id), output)

    metadata[doc_id] = {
        "document_name": title,
        "document_type": category,
        "release_date": scraped_at[:10],
        "source_url": url,
        "source_section": category,
        "ocr_method": f"Google Gemini API ({model_summary})",
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
    parser.add_argument("--max-consecutive-quota-errors", type=int, default=5)
    parser.add_argument("--ocr-batch-size", type=int, default=3)
    parser.add_argument("--conversion-batch-size", type=int, default=20)
    parser.add_argument("--conversion-thread-count", type=int, default=4)
    parser.add_argument("--limit-pdfs", type=int)
    parser.add_argument("--skip-first-pdfs", type=int, default=0)
    parser.add_argument("--max-pages-per-pdf", type=int)
    parser.add_argument("--only-url-contains")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--skip-first-keys",
        type=int,
        default=0,
        help="Ignore this many leading Gemini keys from GEMINI_API_KEYS.",
    )
    parser.add_argument("--key-shard-count", type=int, default=1)
    parser.add_argument("--key-shard-index", type=int, default=0)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be between 0 and shard-count - 1")
    if args.key_shard_count < 1:
        raise ValueError("--key-shard-count must be at least 1")
    if args.key_shard_index < 0 or args.key_shard_index >= args.key_shard_count:
        raise ValueError("--key-shard-index must be between 0 and key-shard-count - 1")
    if args.ocr_batch_size < 1:
        raise ValueError("--ocr-batch-size must be at least 1")

    key_text = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    api_keys = [key.strip() for key in key_text.split(",") if key.strip()]
    if args.skip_first_keys:
        api_keys = api_keys[args.skip_first_keys :]
        print(f"Skipping first {args.skip_first_keys} configured Gemini key(s).")
    if args.key_shard_count > 1:
        api_keys = [
            key
            for index, key in enumerate(api_keys)
            if index % args.key_shard_count == args.key_shard_index
        ]
        print(
            "Using Gemini key shard "
            f"{args.key_shard_index + 1}/{args.key_shard_count} "
            f"({len(api_keys)} key(s))."
        )
    clients = [] if args.convert_only or args.merge_only else build_clients(api_keys)
    client_pool = GeminiClientPool(clients) if clients else None

    records = load_pdf_records(
        only_url_contains=args.only_url_contains,
        limit=args.limit_pdfs,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        skip_first_pdfs=args.skip_first_pdfs,
    )
    print(
        f"PDF records selected: {len(records)} "
        f"(shard {args.shard_index + 1}/{args.shard_count})"
    )

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
            batch_size=args.conversion_batch_size,
            thread_count=args.conversion_thread_count,
        )

        if args.convert_only:
            continue

        processed, skipped, failed, stopped_for_quota = ocr_pages(
            page_paths=page_paths,
            doc_id=doc_id,
            client_pool=client_pool,
            model=args.model,
            request_timeout=args.request_timeout,
            max_consecutive_quota_errors=args.max_consecutive_quota_errors,
            ocr_batch_size=args.ocr_batch_size,
        )
        total_processed += processed
        total_skipped += skipped
        total_failed += failed

        if merge_ocr_document(record, model=args.model, scraped_at=scraped_at):
            merged_docs += 1
        if stopped_for_quota:
            print("Stopping corpus OCR because no usable Gemini quota remains in this run.")
            break

    print("\n========== GEMINI IRC PDF OCR SUMMARY ==========")
    print(f"PDFs selected    : {len(records)}")
    print(f"Pages processed  : {total_processed}")
    print(f"Pages skipped    : {total_skipped}")
    print(f"Pages failed     : {total_failed}")
    if client_pool:
        print(f"Keys quota-stopped: {client_pool.quota_disabled}")
        print(f"Keys still usable : {len(client_pool)}")
    print(f"Merged documents : {merged_docs}")
    print("Next: rerun scripts/05_clean_documents.py through scripts/08_generate_embeddings.py.")


if __name__ == "__main__":
    main()
