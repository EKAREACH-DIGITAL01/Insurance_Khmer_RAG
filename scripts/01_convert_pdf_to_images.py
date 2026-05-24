from pdf2image import (
    convert_from_path,
    pdfinfo_from_path
)
from pathlib import Path

# =====================================================
# ROOT SOURCE FOLDER
# =====================================================

source_root = Path(
    "data/source_documents"
)

# =====================================================
# ROOT OUTPUT FOLDER
# =====================================================

output_root = Path(
    "data/page_images"
)

# =====================================================
# FIND ALL PDFs RECURSIVELY
# =====================================================

pdf_files = sorted(
    source_root.rglob("*.pdf")
)

# =====================================================
# COUNTERS
# =====================================================

processed_pdfs = 0
skipped_pdfs = 0
total_pages = 0

# =====================================================
# PROCESS ALL PDFs
# =====================================================

for pdf_path in pdf_files:

    # Example:
    # laws/law_01.pdf

    category = pdf_path.parent.name

    pdf_name = pdf_path.stem

    # =====================================================
    # OUTPUT FOLDER
    # =====================================================

    output_dir = (
        output_root /
        category /
        pdf_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # SKIP IF ALREADY CONVERTED
    # =====================================================

    existing_images = list(
        output_dir.glob("*.png")
    )

    if existing_images:

        print(
            f"Skipping already processed: "
            f"{pdf_name}"
        )

        skipped_pdfs += 1
        continue

    print(
        f"\nProcessing: "
        f"{pdf_path}"
    )

    # =====================================================
    # GET TOTAL PAGE COUNT
    # =====================================================

    pdf_info = pdfinfo_from_path(
        pdf_path
    )

    page_count = pdf_info["Pages"]

    print(
        f"Total Pages: "
        f"{page_count}"
    )

    # =====================================================
    # CONVERT PAGE BY PAGE
    # =====================================================

    for i in range(page_count):

        print(
            f"Converting page "
            f"{i + 1}/{page_count}"
        )

        try:

            # Convert ONE page only
            pages = convert_from_path(
                pdf_path,
                dpi=400,
                first_page=i + 1,
                last_page=i + 1
            )

            page = pages[0]

            # =====================================================
            # SAVE PAGE
            # =====================================================

            page_path = (
                output_dir /
                f"page_{i+1:03}.png"
            )

            page.save(
                page_path,
                "PNG"
            )

            total_pages += 1

        except Exception as e:

            print(
                f"Failed page "
                f"{i + 1}: {e}"
            )

            continue

    processed_pdfs += 1

    print(
        f"Completed: "
        f"{pdf_name} "
        f"({page_count} pages)"
    )

# =====================================================
# SUMMARY
# =====================================================

print("\n========== SUMMARY ==========")

print(
    f"Processed PDFs : "
    f"{processed_pdfs}"
)

print(
    f"Skipped PDFs   : "
    f"{skipped_pdfs}"
)

print(
    f"Total PNG pages: "
    f"{total_pages}"
)

print(
    "All PDFs converted successfully."
)