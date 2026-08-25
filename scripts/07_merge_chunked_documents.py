import json
from pathlib import Path

from irc_website_categories import WEBSITE_DERIVED_CATEGORIES

input_root = Path("data/07_chunked_documents")
output_root = Path("data/08_merged_chunked_documents")

output_root.mkdir(parents=True, exist_ok=True)

preferred_categories = ["guidance", "laws", "prakas", "sub_decrees"] + WEBSITE_DERIVED_CATEGORIES + ["website"]
available_categories = {path.name for path in input_root.iterdir() if path.is_dir()}
categories = [category for category in preferred_categories if category in available_categories]
categories.extend(sorted(available_categories - set(categories)))

total_chunks = 0

for category in categories:
    category_path = input_root / category

    if not category_path.exists():
        print(f"Skipping missing category: {category}")
        continue

    # Merge every per-document chunk file in the category, in sorted order.
    merged_chunks = []
    json_files = sorted(category_path.glob("*.json"))
    output_path = output_root / f"{category}_chunks.json"

    if not json_files:
        if output_path.exists():
            output_path.unlink()
        print(f"Skipping empty category: {category}")
        continue

    for json_file in json_files:
        print(f"Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            merged_chunks.extend(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

    total_chunks += len(merged_chunks)
    print(f"\nSaved: {output_path.name}")
    print(f"Total chunks: {len(merged_chunks)}\n")

print(f"Chunk merging completed successfully. {total_chunks} chunks total.")
