import time
from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path

input_root = Path("data/01_source_documents")
output_root = Path("data/02_page_images")

output_root.mkdir(parents=True, exist_ok=True)

# Find all PDFs recursively
pdf_files = sorted(input_root.rglob("*.pdf"))

processed_pdfs = 0
skipped_pdfs = 0
total_pages = 0

start_time = time.time()

# Process all PDFs
for pdf_path in pdf_files:
    category = pdf_path.parent.name
    pdf_name = pdf_path.stem

    # Create output folder for converted page images
    output_dir = output_root / category / pdf_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read PDF metadata to determine total page count -- done BEFORE the
    # skip check now, since we need page_count to know whether a previous
    # run actually finished this PDF or was interrupted partway through.
    pdf_info = pdfinfo_from_path(pdf_path)
    page_count = pdf_info["Pages"]

    # Skip only if already FULLY converted. Previously this skipped on
    # ANY existing PNGs, regardless of count -- meaning an interrupted run
    # (Ctrl+C, crash, power loss) that only got through e.g. 3 of 10 pages
    # would cause that PDF to be silently skipped forever on every future
    # run, leaving it permanently incomplete with no warning.
    existing_images = list(output_dir.glob("*.png"))

    if len(existing_images) >= page_count:
        print(f"Skipping already processed: {pdf_name}")
        skipped_pdfs += 1
        continue

    print(f"\nProcessing: {pdf_path}")
    print(f"Total Pages: {page_count}")

    if existing_images:
        print(
            f"Found {len(existing_images)}/{page_count} existing pages -- "
            f"resuming from a previous interrupted run."
        )

    # Convert page by page
    for i in range(page_count):
        page_path = output_dir / f"page_{i + 1:03}.png"

        # Skip individual pages already converted in a previous run.
        if page_path.exists():
            continue

        print(f"Converting page {i + 1}/{page_count}")

        try:
            # Convert one page at a time to keep memory usage predictable
            pages = convert_from_path(
                pdf_path, dpi=400, first_page=i + 1, last_page=i + 1
            )
            page = pages[0]

            # Save converted page as PNG
            page.save(page_path, "PNG")

            total_pages += 1

        except Exception as e:
            print(f"Failed page {i + 1}: {e}")
            continue

    processed_pdfs += 1
    print(f"Completed: {pdf_name} ({page_count} pages)")

elapsed_time = time.time() - start_time

print("\n========== SUMMARY ==========")
print(f"Processed PDFs : {processed_pdfs}")
print(f"Skipped PDFs   : {skipped_pdfs}")
print(f"Total PNG pages: {total_pages}")
print(f"Total Runtime  : {elapsed_time:.2f} seconds")
print("All PDFs converted successfully.")
