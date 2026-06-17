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

MAX_RETRIES = 10
REQUEST_TIMEOUT = 100

processed_pages = 0
skipped_pages = 0
failed_pages = 0

start_time = time.time()


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

    # Skip existing files
    if output_path.exists():
        print(f"Skipping already processed: {document_id} - {page_name}")
        skipped_pages += 1
        continue

    # Process each page image
    print(f"\nProcessing: {document_id} - {page_name}")

    page_start_time = time.time()

    with Image.open(image_path) as image:
        retry_count = 0
        success = False

        while retry_count < MAX_RETRIES:
            try:
                print("Sending Gemini request...")

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(generate_ocr_response, image)

                    try:
                        response = future.result(timeout=REQUEST_TIMEOUT)

                    except concurrent.futures.TimeoutError:
                        print(
                            f"Timeout after {REQUEST_TIMEOUT} seconds: {document_id} - {page_name}"
                        )
                        print("Skipping page...")

                        success = False
                        break

                # Log Gemini token usage for monitoring and cost tracking
                print("\nUsage Metadata:")
                print(response.usage_metadata)
                print()

                extracted_text = response.text

                # Save JSON immediately to avoid losing progress
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

                success = True
                break

            # Increase retry delay to reduce repeated API failures
            except Exception as e:
                retry_count += 1
                wait_time = min(15 * retry_count, 120)
                print(f"Error processing {document_id} - {page_name}: {e}")
                print(
                    f"Retry {retry_count}/{MAX_RETRIES} "
                    f"after waiting {wait_time} seconds...\n"
                )
                time.sleep(wait_time)

    # Record all failed pages
    if not success:
        print(f"Failed permanently: {document_id} - {page_name}")
        failed_pages += 1
        continue

    processed_pages += 1

    # Measure processing time for the current page
    page_elapsed = time.time() - page_start_time
    print(f"Completed: {document_id} - {page_name} ({page_elapsed:.2f}s)")

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Processed Pages : {processed_pages}")
print(f"Skipped Pages   : {skipped_pages}")
print(f"Failed Pages    : {failed_pages}")
print(f"Total Runtime   : {elapsed_time:.2f} seconds")
print("OCR extraction completed successfully.")

# Force process termination if Python does not exit cleanly
os._exit(0)
