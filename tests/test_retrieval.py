import unittest
from unittest.mock import Mock

import numpy as np

from khmer_rag.language import khmer_aware_tokenize
from khmer_rag.provisions import resolve_provision_lookup
from khmer_rag.retrieval.context import select_context_chunks, select_used_sources
from khmer_rag.retrieval.search import reciprocal_rank_fusion, search_faiss


def chunk(document, number=1, text="អត្ថបទ។", **metadata):
    return {
        "chunk_id": f"{document}_chunk_{number}",
        "text": text,
        "metadata": {"document_id": document, "document_name": document, **metadata},
    }


class RetrievalTests(unittest.TestCase):
    def test_article_references_accept_either_digit_script(self):
        for query in ("មាត្រា 10", "មាត្រា១០"):
            self.assertIn("មាត្រា១០", khmer_aware_tokenize(query))
        self.assertNotIn("ប្រការ១០", khmer_aware_tokenize("មាត្រា 10"))

    def test_fusion_rewards_agreement_and_keeps_raw_results(self):
        a, b = chunk("a"), chunk("b")
        dense = [(0.9, a), (0.8, b)]
        fused = reciprocal_rank_fusion([dense, [(20, b)]])
        self.assertEqual(fused[0][1], b)
        self.assertEqual(len(fused), 2)
        self.assertEqual(dense[0][0], 0.9)

    def test_faiss_padding_does_not_return_last_document(self):
        index = Mock()
        index.search.return_value = (np.array([[0.8, -1, -1]]), np.array([[0, -1, -1]]))
        records = [chunk("law")]
        result, _ = search_faiss(np.zeros((1, 3)), index, records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], records[0])

    def test_context_caps_diverse_documents(self):
        records = [(1, chunk(doc, number)) for number in range(4) for doc in ("a", "b", "c", "d")]
        selected = select_context_chunks(records)
        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(
                sum(c["metadata"]["document_id"] == doc for _, c in selected) <= 2
                for doc in ("a", "b", "c", "d")
            )
        )

    def test_dominant_document_keeps_adjacent_provisions(self):
        records = [(1, chunk("law", i)) for i in range(6)]
        selected = select_context_chunks(records)
        self.assertEqual(len(selected), 4)

    def test_exact_provision_does_not_match_number_prefix(self):
        law = "ច្បាប់ធានារ៉ាប់រង"
        candidates = [
            (1, chunk(law, 100, "មាត្រា ១០០ - មិនមែនចម្លើយ។")),
            (0.8, chunk(law, 10, "មាត្រា ១០ - ចម្លើយពេញលេញ។")),
        ]
        self.assertEqual(
            resolve_provision_lookup(f"{law} មាត្រា 10", candidates), ["មាត្រា ១០ - ចម្លើយពេញលេញ។"]
        )

    def test_named_law_does_not_borrow_another_laws_provision(self):
        candidates = [
            (1, chunk("ច្បាប់ក", text="មាត្រា ១ - អត្ថបទ។")),
            (0.9, chunk("ច្បាប់ខ", text="មាត្រា ១០ - អត្ថបទផ្សេង។")),
        ]
        answer = resolve_provision_lookup("ច្បាប់ក មាត្រា 10", candidates)
        self.assertIn("រកមិនឃើញ", answer[0])

    def test_sources_prefer_documents_over_photos(self):
        photo = chunk("photo", source_url="https://irc.gov.kh/photo.png")
        pdf = chunk("law", source_url="https://irc.gov.kh/law.pdf")
        self.assertEqual(select_used_sources([(1, photo), (0.8, pdf)]), [(0.8, pdf)])
