import queue
import unittest
from threading import Event
from unittest.mock import Mock, patch

from khmer_rag.language import GREETING_RESPONSE_TEXT
from khmer_rag.pipeline import run_pipeline_in_thread
from tests.test_retrieval import chunk


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, question, records=()):
        output = queue.Queue()
        run_pipeline_in_thread(
            question, "test", Mock(), Mock(), {}, Mock(), Mock(), records, output, Event()
        )
        events = []
        while not output.empty():
            events.append(output.get_nowait())
        return events

    @patch("khmer_rag.pipeline.encode_query")
    def test_greeting_skips_retrieval_and_returns_no_sources(self, encode):
        events = self.run_pipeline("សួស្ដី")
        encode.assert_not_called()
        kind, payload = events[-1]
        self.assertEqual(kind, "canned_response")
        self.assertEqual(payload["answer"], GREETING_RESPONSE_TEXT)
        self.assertEqual(payload["results"], [])

    @patch("khmer_rag.pipeline.encode_query")
    def test_english_is_rejected_before_retrieval(self, encode):
        events = self.run_pipeline("What is insurance?")
        encode.assert_not_called()
        self.assertEqual(events[-1][1]["reason"], "not_khmer")

    @patch("khmer_rag.pipeline.encode_query", return_value=([], 0.01))
    @patch("khmer_rag.pipeline.search_faiss", return_value=([], 0.01))
    @patch("khmer_rag.pipeline.stream_answer")
    def test_empty_retrieval_declines_without_generation(self, stream, search, encode):
        events = self.run_pipeline("តើធានារ៉ាប់រងជាអ្វី?")
        stream.assert_not_called()
        self.assertEqual(events[-1][1]["reason"], "low_score")

    @patch("khmer_rag.pipeline.count_tokens", return_value=None)
    @patch("khmer_rag.pipeline.choose_max_tokens_live", return_value=512)
    @patch("khmer_rag.pipeline.encode_query", return_value=([], 0.01))
    def test_generated_answer_has_context_and_diagnostics(self, encode, tokens, count):
        records = [chunk("law", text="ធានារ៉ាប់រងជាកិច្ចសន្យា។")]
        ranked = [(0.9, records[0])]
        with (
            patch("khmer_rag.pipeline.search_faiss", return_value=(ranked, 0.01)),
            patch("khmer_rag.pipeline.search_bm25", return_value=(ranked, 0.01)),
            patch("khmer_rag.pipeline.stream_answer", return_value=iter(["ចម្លើយ", "។"])),
        ):
            events = self.run_pipeline("តើធានារ៉ាប់រងជាអ្វី?", records)
        self.assertEqual([value for kind, value in events if kind == "token"], ["ចម្លើយ", "។"])
        kind, result = events[-1]
        self.assertEqual(kind, "done")
        self.assertEqual(result["context_results"][0][1], records[0])
        self.assertEqual(result["best_score"], 0.9)
        self.assertIn("faiss", result["score_lookup"][records[0]["chunk_id"]])

    @patch("khmer_rag.pipeline.encode_query", side_effect=RuntimeError("index mismatch"))
    def test_failure_emits_error_and_logs_traceback(self, encode):
        with self.assertLogs("khmer_rag.pipeline", level="ERROR") as captured:
            events = self.run_pipeline("តើធានារ៉ាប់រងជាអ្វី?")
        self.assertEqual(events[-1], ("error", "index mismatch"))
        self.assertIn("Traceback", captured.output[0])
