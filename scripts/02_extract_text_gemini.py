from google import genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
import os
import json
import time

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

# =====================================================
# GEMINI CLIENT
# =====================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================================
# ROOT INPUT FOLDER
# =====================================================

input_root = Path(
    "data/page_images"
)

# =====================================================
# ROOT OUTPUT FOLDER
# =====================================================

output_root = Path(
    "data/extracted_text/raw"
)

output_root.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# OCR PROMPT
# =====================================================

prompt = """
Just read the image and write the text back exactly. Do not change anything.
"""

# =====================================================
# FIND ALL PNG FILES RECURSIVELY
# =====================================================

image_files = sorted(
    input_root.rglob("*.png")
)

# =====================================================
# COUNTERS
# =====================================================

processed_pages = 0
skipped_pages = 0
failed_pages = 0

# =====================================================
# TOTAL RUNTIME
# =====================================================

start_time = time.time()

# =====================================================
# GEMINI OCR FUNCTION
# =====================================================

def generate_ocr_response(image):

    return client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, image],
        config={
            "temperature": 0
        }
    )

# =====================================================
# PROCESS ALL PAGES
# =====================================================

for image_path in image_files:

    # Example:
    # data/page_images/laws/law_01/page_001.png

    relative_path = image_path.relative_to(
        input_root
    )

    # Category:
    # laws
    category = relative_path.parts[0]

    # Folder:
    # law_01
    document_id = relative_path.parts[1]

    # Filename:
    # page_001
    page_name = image_path.stem

    # Extract page number
    page_number = int(
        page_name.split("_")[1]
    )

    # =====================================================
    # OUTPUT PATH
    # =====================================================

    output_folder = (
        output_root / category
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder /
        f"{document_id}_{page_name}.json"
    )

    # =====================================================
    # SKIP EXISTING FILES
    # =====================================================

    if output_path.exists():

        # print(
        #     f"Skipping already processed: "
        #     f"{document_id} - {page_name}"
        # )

        skipped_pages += 1
        continue

    # =====================================================
    # PROCESS PAGE
    # =====================================================

    print(
        f"\nProcessing: "
        f"{document_id} - {page_name}"
    )

    page_start_time = time.time()

    # Open image
    image = Image.open(image_path)

    # Retry settings
    max_retries = 10
    retry_count = 0

    success = False

    # =====================================================
    # RETRY SAME PAGE UNTIL SUCCESS
    # =====================================================

    while retry_count < max_retries:

        try:

            # =====================================================
            # GEMINI OCR
            # =====================================================

            print("Sending Gemini request...")

            executor = concurrent.futures.ThreadPoolExecutor()

            future = executor.submit(
                generate_ocr_response,
                image
            )

            try:

                response = future.result(
                    timeout=100
                )

            except concurrent.futures.TimeoutError:

                print(
                    f"Timeout after 100 seconds: "
                    f"{document_id} - {page_name}"
                )

                print(
                    "Skipping page..."
                )

                future.cancel()
                executor.shutdown(wait=False)

                executor._threads.clear()
                concurrent.futures.thread._threads_queues.clear()

                success = False

                break

            # =====================================================
            # USAGE METADATA
            # =====================================================

            print("\nUsage Metadata:")
            print(response.usage_metadata)
            print()

            extracted_text = response.text

            # =====================================================
            # SAVE JSON IMMEDIATELY
            # =====================================================

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    {
                        "document_id": document_id,
                        "page": page_number,
                        "raw_text": extracted_text
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            success = True
            break

        except Exception as e:

            retry_count += 1

            # =====================================================
            # EXPONENTIAL BACKOFF
            # =====================================================

            wait_time = min(
                15 * retry_count,
                120
            )

            print(
                f"Error processing "
                f"{document_id} - "
                f"{page_name}: {e}"
            )

            print(
                f"Retry "
                f"{retry_count}/{max_retries} "
                f"after waiting "
                f"{wait_time} seconds...\n"
            )

            time.sleep(wait_time)

    # =====================================================
    # PAGE FAILED
    # =====================================================

    if not success:

        print(
            f"Failed permanently: "
            f"{document_id} - "
            f"{page_name}"
        )

        failed_pages += 1
        continue

    processed_pages += 1

    # =====================================================
    # PAGE RUNTIME
    # =====================================================

    page_elapsed = (
        time.time() - page_start_time
    )

    print(
        f"Completed: "
        f"{document_id} - "
        f"{page_name} "
        f"({page_elapsed:.2f}s)"
    )

    # =====================================================
    # OPTIONAL RATE LIMIT SAFETY
    # =====================================================

    # WARNING:
    # Uncomment this only if requests become
    # too fast and start hitting RPM limits.
    #
    # Current OCR processing is already slow
    # enough naturally (~30–60s per page),
    # so extra sleeping is usually unnecessary.
    #
    # time.sleep(5)

# =====================================================
# TOTAL RUNTIME
# =====================================================

elapsed_time = (
    time.time() - start_time
)

# =====================================================
# SUMMARY
# =====================================================

print("\n========== SUMMARY ==========")

print(
    f"Processed Pages : "
    f"{processed_pages}"
)

print(
    f"Skipped Pages   : "
    f"{skipped_pages}"
)

print(
    f"Failed Pages    : "
    f"{failed_pages}"
)

print(
    f"Total Runtime   : "
    f"{elapsed_time:.2f} seconds"
)

print(
    "OCR extraction completed successfully."
)

os._exit(0)