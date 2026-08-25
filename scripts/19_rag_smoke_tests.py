#!/usr/bin/env python3
"""Fast local smoke tests for the Khmer Insurance RAG corpus and retriever.

This is intentionally smaller than the full evaluation suite. It does not
start llama-server and does not call Gemini. It validates the committed BGE-M3
vector DB, the website-expanded corpus, and the exact sources needed for common
demo questions.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

# Must be set before importing faiss/torch through sentence_transformers.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "models" / "retriever" / ".cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ACTIVE_MODEL_DIR = "BAAI-BGE-M3"
INDEX_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "khmer_insurance.index"
METADATA_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "khmer_insurance_metadata.json"
MODEL_INFO_PATH = PROJECT_ROOT / "vector_db" / ACTIVE_MODEL_DIR / "embedding_model_info.json"

TOP_K = 15
BM25_TOP_K = TOP_K
RRF_K = 60
NGRAM_SIZE = 3

MIN_TOTAL_CHUNKS = 2900
MIN_WEBSITE_CHUNKS = 1500
MIN_DOCUMENTS = 780

KHMER_SCRIPT_PATTERN = re.compile(r"[\u1780-\u17FF]")
DIGIT_RUN_PATTERN = re.compile(r"[\u17E0-\u17E90-9]+")
ARTICLE_REF_PATTERN = re.compile(r"(មាត្រា|ប្រការ)\s*([\u17E0-\u17E90-9]+)")
ARTICLE_REF_BOOST = 20
ARABIC_TO_KHMER = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")

POLICY_COUNT_ROW_PATTERN = re.compile(
    r"\|\s*(?:[\u17E1-\u17E3]\.\s*)?([^|]*ធានារ៉ាប់រង[^|]+?)\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*"
    r"\|\s*([0-9,]+)(?:\.00)?\s*\|"
)
POLICY_COUNT_TOTAL_PATTERN = re.compile(
    r"\|\s*សរុប\s*\|\s*([0-9,]+)(?:\.00)?\s*\|\s*([0-9,]+)(?:\.00)?\s*\|"
)

SMOKE_QUERIES = [
    {
        "label": "management structure",
        "question": "តើថ្នាក់ដឹកនាំន.ធ.ក.មានរណាខ្លះ?",
        "expected_doc": "website_homepage_management_structure",
        "must_be_in": ("dense", "fused"),
    },
    {
        "label": "2025 gross premium",
        "question": "តើមានចំនួនបុព្វលាភប៉ុន្មានសម្រាប់ឆ្នាំ២០២៥?",
        "expected_doc": "website_homepage_gross_premium_statistics",
        "must_be_in": ("dense", "bm25", "fused"),
    },
    {
        "label": "publication catalog",
        "question": "តើការបោះពុម្ពផ្សាយរបស់ន.ធ.ក.មានអ្វីខ្លះ?",
        "expected_doc": "website_publication_catalog",
        "must_be_in": ("dense", "bm25", "fused"),
    },
    {
        "label": "annual reports",
        "question": "តើរបាយការណ៍ប្រចាំឆ្នាំរបស់ន.ធ.ក.មានឆ្នាំណាខ្លះ?",
        "expected_doc": "website_publication_catalog",
        "must_be_in": ("dense", "bm25", "fused"),
    },
    {
        "label": "Bou Chanphirou profile",
        "question": "សូមប្រាប់ប្រវត្តិរបស់ឯកឧត្តម ប៊ូ ច័ន្ទភីរូ",
        "expected_doc": "website_file_ceab89ff2b0c",
        "must_be_in": ("dense", "fused"),
    },
    {
        "label": "policy count 2021 vs 2020",
        "question": (
            "យោងតាមផែនការអភិវឌ្ឍន៍ស្ថាប័នរបស់និយ័តករធានារ៉ាប់រងកម្ពុជា "
            "២០២១-២០២៥ តើចំនួនបណ្ណសន្យាធានារ៉ាប់រងមានការប្រែប្រួលយ៉ាងដូចម្តេច?"
        ),
        "expected_doc": "website_file_386481a18166",
        "must_be_in": ("dense", "fused"),
    },
]

EXPECTED_POLICY_ROWS = {
    "ធានារ៉ាប់រងទូទៅ": ("256,460", "208,450"),
    "ធានារ៉ាប់រងអាយុជីវិត": ("485,249", "398,688"),
    "ធានារ៉ាប់រងខ្នាតតូច": ("301,856", "345,621"),
    "សរុប": ("1,043,565", "952,759"),
}


def to_khmer_digits(value: str) -> str:
    return value.translate(ARABIC_TO_KHMER)


def khmer_aware_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for keyword, ref_num in ARTICLE_REF_PATTERN.findall(text):
        tokens.extend([keyword + to_khmer_digits(ref_num)] * ARTICLE_REF_BOOST)

    for word in text.split():
        if KHMER_SCRIPT_PATTERN.search(word):
            tokens.extend(DIGIT_RUN_PATTERN.findall(word))
            chars = list(word)
            if len(chars) < NGRAM_SIZE:
                tokens.append(word)
            else:
                tokens.extend(
                    "".join(chars[i : i + NGRAM_SIZE])
                    for i in range(len(chars) - NGRAM_SIZE + 1)
                )
        else:
            tokens.append(word.lower())
    return tokens


def reciprocal_rank_fusion(result_lists, k: int = RRF_K):
    fused_scores = {}
    chunk_lookup = {}
    for result_list in result_lists:
        for rank, (_score, chunk) in enumerate(result_list, start=1):
            chunk_id = chunk["chunk_id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(fused_scores[chunk_id], chunk_lookup[chunk_id]) for chunk_id in ranked_ids]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS {message}")
    else:
        print(f"FAIL {message}")
        failures.append(message)


def warn(condition: bool, message: str) -> None:
    if condition:
        print(f"WARN {message}")


def doc_id(chunk) -> str:
    return chunk.get("metadata", {}).get("document_id", "")


def first_doc_rank(results, expected_doc: str) -> int | None:
    for rank, (_score, chunk) in enumerate(results, start=1):
        if doc_id(chunk) == expected_doc:
            return rank
    return None


def search_dense(question: str, model, model_info, index, chunks):
    query = question
    if model_info.get("query_prefix"):
        query = model_info["query_prefix"] + query
    embedding = model.encode([query], normalize_embeddings=True)
    embedding = np.asarray(embedding, dtype="float32")
    scores, indices = index.search(embedding, k=TOP_K)
    return [
        (float(score), chunks[idx])
        for score, idx in zip(scores[0], indices[0])
        if 0 <= idx < len(chunks)
    ]


def search_bm25(question: str, bm25_index, chunks):
    scores = bm25_index.get_scores(khmer_aware_tokenize(question))
    top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
    return [(float(scores[idx]), chunks[idx]) for idx in top_indices]


def extract_policy_rows(chunks):
    text = "\n".join(
        chunk.get("text") or ""
        for chunk in chunks
        if doc_id(chunk) == "website_file_386481a18166"
    )
    section_start = text.find("ចំនួនបណ្ណសន្យារ៉ាប់រង")
    if section_start < 0:
        return {}
    section_end = text.find("ទ្រព្យសកម្មសរុប", section_start)
    section = text[section_start : section_end if section_end >= 0 else len(text)]

    rows = {}
    for label, current, previous in POLICY_COUNT_ROW_PATTERN.findall(section):
        label = re.sub(r"\s+", " ", label).strip()
        label = re.sub(r"^[\u17E1-\u17E3]\.\s*", "", label).strip()
        rows[label] = (current, previous)

    total_match = POLICY_COUNT_TOTAL_PATTERN.search(section)
    if total_match:
        rows["សរុប"] = (total_match.group(1), total_match.group(2))
    return rows


def image_asset_reference_count(chunks) -> int:
    image_ext = re.compile(r"https?://\S+\.(?:jpg|jpeg|png|webp)(?:\s|$)", re.IGNORECASE)
    return sum(1 for chunk in chunks if image_ext.search(chunk.get("text") or ""))


def main() -> int:
    failures: list[str] = []

    check(
        INDEX_PATH.exists(),
        f"FAISS index exists at {INDEX_PATH.relative_to(PROJECT_ROOT)}",
        failures,
    )
    check(
        METADATA_PATH.exists(),
        f"metadata exists at {METADATA_PATH.relative_to(PROJECT_ROOT)}",
        failures,
    )
    check(
        MODEL_INFO_PATH.exists(),
        f"model info exists at {MODEL_INFO_PATH.relative_to(PROJECT_ROOT)}",
        failures,
    )
    if failures:
        return 1

    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(INDEX_PATH))
    chunks = load_json(METADATA_PATH)
    model_info = load_json(MODEL_INFO_PATH)

    type_counts = Counter(chunk.get("metadata", {}).get("document_type") for chunk in chunks)
    document_count = len({doc_id(chunk) for chunk in chunks})

    print("\nCorpus")
    check(
        len(chunks) >= MIN_TOTAL_CHUNKS,
        f"chunk count is {len(chunks)} >= {MIN_TOTAL_CHUNKS}",
        failures,
    )
    check(
        index.ntotal == len(chunks),
        f"FAISS vectors ({index.ntotal}) match metadata chunks ({len(chunks)})",
        failures,
    )
    check(
        document_count >= MIN_DOCUMENTS,
        f"document count is {document_count} >= {MIN_DOCUMENTS}",
        failures,
    )
    check(
        type_counts.get("website", 0) >= MIN_WEBSITE_CHUNKS,
        f"website chunks are {type_counts.get('website', 0)} >= {MIN_WEBSITE_CHUNKS}",
        failures,
    )
    check(type_counts.get("law", 0) > 0, "law chunks are present", failures)
    check(type_counts.get("prakas", 0) > 0, "prakas chunks are present", failures)
    check(type_counts.get("sub_decree", 0) > 0, "sub-decree chunks are present", failures)
    check(type_counts.get("guidance", 0) > 0, "guidance chunks are present", failures)
    check(
        not any(
            "ធានរបង" in (chunk.get("text") or "")
            or "ធានរបង" in (chunk.get("embedding_text") or "")
            for chunk in chunks
        ),
        "known OCR/generation typo 'ធានរបង' is absent from corpus text",
        failures,
    )
    warn(
        image_asset_reference_count(chunks) > 0,
        "some chunks mention image asset URLs; this should stay informational, not answer text",
    )

    print("\nLoading BGE-M3 and BM25")
    try:
        model = SentenceTransformer(
            model_info["hf_id"],
            trust_remote_code=bool(model_info.get("trust_remote_code")),
            local_files_only=True,
        )
    except Exception as exc:
        print(f"FAIL BGE-M3 could not load locally: {exc}")
        return 1

    bm25_index = BM25Okapi(
        [khmer_aware_tokenize(chunk.get("embedding_text") or chunk["text"]) for chunk in chunks]
    )
    print("PASS retrievers loaded")

    print("\nDemo retrieval")
    for item in SMOKE_QUERIES:
        dense = search_dense(item["question"], model, model_info, index, chunks)
        bm25 = search_bm25(item["question"], bm25_index, chunks)
        fused = reciprocal_rank_fusion([dense, bm25])
        rankings = {"dense": dense, "bm25": bm25, "fused": fused}

        ranks = {
            name: first_doc_rank(results, item["expected_doc"])
            for name, results in rankings.items()
        }
        rank_text = ", ".join(
            f"{name}={rank if rank is not None else 'miss'}"
            for name, rank in ranks.items()
        )
        for retrieval_name in item["must_be_in"]:
            check(
                ranks[retrieval_name] is not None,
                f"{item['label']} source {item['expected_doc']} appears in {retrieval_name} top-{TOP_K} ({rank_text})",
                failures,
            )

    print("\nPolicy-count source values")
    policy_rows = extract_policy_rows(chunks)
    for label, expected in EXPECTED_POLICY_ROWS.items():
        check(
            policy_rows.get(label) == expected,
            f"{label}: {policy_rows.get(label)} matches expected {expected}",
            failures,
        )

    if failures:
        print(f"\nFAILED {len(failures)} smoke check(s).")
        return 1

    print("\nAll RAG smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
