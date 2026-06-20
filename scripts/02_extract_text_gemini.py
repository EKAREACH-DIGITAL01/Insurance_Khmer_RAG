import os
import json
import time
import concurrent.futures
from google import genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

input_root = Path("data/02_page_images")
output_root = Path("data/03_raw_pages")

output_root.mkdir(parents=True, exist_ok=True)

# Find all page images recursively
image_files = sorted(input_root.rglob("*.png"))

# Load Environment Variables
load_dotenv()

# GEMINI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Prompt used for Gemini OCR extraction
prompt = "Just read the image and write the text back exactly. Do not change anything."

REQUEST_TIMEOUT = 100

processed_pages = 0
skipped_pages = 0
failed_pages = 0

start_time = time.time()

# DESIGN NOTE: this pipeline is NOT meant to guarantee every page succeeds
# in a single run. It runs page-by-page against a paid API, potentially
# over thousands of pages, and Gemini failures are often transient -- a
# page that fails today may succeed cleanly on a later run. So each page
# gets exactly ONE attempt here: on any failure (timeout or otherwise) we
# skip it immediately and move on, rather than looping with backoff. No
# output JSON is written for a failed page, so the "skip if already
# processed" check below naturally retries only the actual gaps the next
# time this script runs -- that's the intended retry mechanism, not an
# in-script loop:
#
#   timeout/failure -> skip page -> continue next page -> retry on a
#   future run
#
# rather than:
#
#   timeout/failure -> keep retrying for a long time -> risk blocking
#   the rest of the corpus
#
# A single persistent executor (not a fresh per-call context-managed one)
# is still used so that a timed-out call's background thread is simply
# abandoned rather than blocking script exit while waiting for it to
# finish (which is what `with ThreadPoolExecutor() as executor:` would do
# via its implicit shutdown(wait=True)).
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# Send image to Gemini OCR model
def generate_ocr_response(image):
    return client.models.generate_content(
        model="gemini-3.5-flash", contents=[prompt, image], config={"temperature": 0}
    )


print(f"Total images found: {len(image_files)}")

# Process all page images
for image_path in image_files:
    relative_path = image_path.relative_to(input_root)

    category = relative_path.parts[0]
    document_id = relative_path.parts[1]
    page_name = image_path.stem

    # Extract page number
    page_number = int(page_name.split("_")[1])

    # Output path
    output_folder = output_root / category
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{document_id}_{page_name}.json"

    # Skip pages already successfully processed in a previous run. This is
    # also how failed pages get retried: since a failed page never had its
    # JSON written, it simply isn't skipped here next time.
    if output_path.exists():
        print(f"Skipping already processed: {document_id} - {page_name}")
        skipped_pages += 1
        continue

    # Process each page image
    print(f"\nProcessing: {document_id} - {page_name}")

    page_start_time = time.time()

    with Image.open(image_path) as image:
        try:
            print("Sending Gemini request...")

            future = executor.submit(generate_ocr_response, image)

            try:
                response = future.result(timeout=REQUEST_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(
                    f"Timeout after {REQUEST_TIMEOUT} seconds: "
                    f"{document_id} - {page_name}"
                )
                print("Skipping page (a later run will retry it)...")
                failed_pages += 1
                continue

            # Log Gemini token usage for monitoring and cost tracking
            print("\nUsage Metadata:")
            print(response.usage_metadata)
            print()

            extracted_text = response.text

            # Save JSON immediately so successful pages are never lost or
            # reprocessed, regardless of what happens to later pages.
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "document_id": document_id,
                        "page": page_number,
                        "raw_text": extracted_text,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            processed_pages += 1

            page_elapsed = time.time() - page_start_time
            print(f"Completed: {document_id} - {page_name} ({page_elapsed:.2f}s)")

        except Exception as e:
            print(f"Error processing {document_id} - {page_name}: {e}")
            print("Skipping page (a later run will retry it)...")
            failed_pages += 1
            continue

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Processed Pages : {processed_pages}")
print(f"Skipped Pages   : {skipped_pages}")
print(f"Failed Pages    : {failed_pages}")
print(f"Total Runtime   : {elapsed_time:.2f} seconds")
print("OCR extraction completed successfully.")

# Force process termination if Python does not exit cleanly -- belt and
# braces in case any abandoned/timed-out future thread is still alive.
os._exit(0)
