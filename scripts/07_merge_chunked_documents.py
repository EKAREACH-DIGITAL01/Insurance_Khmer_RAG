import json
from pathlib import Path

input_root = Path("data/07_chunked_documents")
output_root = Path("data/08_merged_chunked_documents")

output_root.mkdir(parents=True, exist_ok=True)

categories = ["guidance", "laws", "prakas", "sub_decrees", "website"]

total_chunks = 0

for category in categories:
    category_path = input_root / category

    if not category_path.exists():
        print(f"Skipping missing category: {category}")
        continue

    # Merge every per-document chunk file in the category, in sorted order.
    merged_chunks = []
    json_files = sorted(category_path.glob("*.json"))

    for json_file in json_files:
        print(f"Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            merged_chunks.extend(json.load(f))

    output_path = output_root / f"{category}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

    total_chunks += len(merged_chunks)
    print(f"\nSaved: {output_path.name}")
    print(f"Total chunks: {len(merged_chunks)}\n")

print(f"Chunk merging completed successfully. {total_chunks} chunks total.")
