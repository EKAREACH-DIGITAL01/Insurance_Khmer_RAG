import unittest

from khmer_rag.answers import (
    POLICY_COUNT_SOURCE_DOC_ID,
    augment_curated_context,
    extract_gross_premium_answer,
    extract_policy_count_answer,
    has_curated_source_candidate,
)
from tests.test_retrieval import chunk

POLICY_TABLE = """ចំនួនបណ្ណសន្យារ៉ាប់រង
| ប្រភេទ | 2021 | 2020 |
| ធានារ៉ាប់រងទូទៅ | 256,460 | 208,450 |
| ធានារ៉ាប់រងអាយុជីវិត | 485,249 | 398,688 |
| ធានារ៉ាប់រងខ្នាតតូច | 301,856 | 345,621 |
| សរុប | 1,043,565 | 952,759 |
"""


class CuratedAnswerTests(unittest.TestCase):
    def test_fallback_search_uses_only_the_supplied_corpus(self):
        question = "ចំនួនបណ្ណសន្យារ៉ាប់រងមានការប្រែប្រួលដូចម្តេច?"
        corpus = [chunk(POLICY_COUNT_SOURCE_DOC_ID, text=POLICY_TABLE)]
        self.assertFalse(has_curated_source_candidate(question, [], []))
        self.assertTrue(has_curated_source_candidate(question, [], corpus))
        selected = augment_curated_context(question, [], [], corpus)
        self.assertEqual(selected[0][1], corpus[0])

    def test_policy_count_preserves_labels_values_and_decrease(self):
        question = "ចំនួនបណ្ណសន្យារ៉ាប់រងមានការប្រែប្រួលដូចម្តេច?"
        result = [(1, chunk(POLICY_COUNT_SOURCE_DOC_ID, text=POLICY_TABLE))]
        answer = extract_policy_count_answer(question, result)
        self.assertIsNotNone(answer)
        self.assertIn("ធានារ៉ាប់រងខ្នាតតូច", answer)
        self.assertIn("345,621", answer)
        self.assertIn("301,856", answer)
        self.assertIn("ថយចុះ", answer)
        self.assertNotIn("ធានរបង", answer)

    def test_premium_lookup_requires_the_requested_year(self):
        source = chunk(
            "website_homepage_gross_premium_statistics",
            text="2025 Total Gross Premium 367,048,905 USD",
        )
        self.assertIn("367,048,905", extract_gross_premium_answer("បុព្វលាភឆ្នាំ២០២៥", [(1, source)]))
        self.assertIsNone(extract_gross_premium_answer("បុព្វលាភឆ្នាំ២០២៤", [(1, source)]))
