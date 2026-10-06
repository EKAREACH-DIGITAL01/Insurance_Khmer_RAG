"""Compatibility imports. New code should use khmer_rag.categories."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khmer_rag.categories import (
    ARTICLE_CATEGORY_ALIASES,
    PATH_CATEGORY_ALIASES,
    TITLE_CATEGORY_HINTS,
    WEBSITE_CATEGORY_LABEL_KH,
    WEBSITE_DERIVED_CATEGORIES,
    category_from_text,
    category_from_url,
    first_content_lines,
    infer_website_category,
    is_home_url,
    normalize_article_category,
    title_hint_category,
)

__all__ = [
    "WEBSITE_DERIVED_CATEGORIES",
    "WEBSITE_CATEGORY_LABEL_KH",
    "ARTICLE_CATEGORY_ALIASES",
    "PATH_CATEGORY_ALIASES",
    "TITLE_CATEGORY_HINTS",
    "normalize_article_category",
    "category_from_url",
    "category_from_text",
    "title_hint_category",
    "first_content_lines",
    "is_home_url",
    "infer_website_category",
]
