import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from khmer_rag.generation.stream import stream_answer
from khmer_rag.language import GREETING_RESPONSE_TEXT, is_khmer_greeting
from khmer_rag.text import normalize_answer_text


def delta(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class Stream:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class GenerationTests(unittest.TestCase):
    def test_khmer_greetings_but_not_greeting_with_question(self):
        for text in ("សួស្ដី", "ជម្រាបសួរ", "សួស្តីបាទ!", "សូមជម្រាបសួរ"):
            self.assertTrue(is_khmer_greeting(text), text)
        self.assertFalse(is_khmer_greeting("សួស្តី តើធានារ៉ាប់រងជាអ្វី?"))
        self.assertEqual(
            GREETING_RESPONSE_TEXT,
            "ជម្រាបសួរ! ខ្ញុំជាជំនួយការធានារ៉ាប់រងកម្ពុជា។ តើលោក/លោកស្រី មានចម្ងល់អ្វីខ្លះដែរទាក់ទងនឹងការធានារ៉ាប់រង?",
        )

    def test_generated_source_footer_is_removed(self):
        answer = "ចម្លើយត្រឹមត្រូវ។\nSources:\nhttps://example.com\nចម្លើយបន្ថែម។"
        self.assertEqual(normalize_answer_text(answer), "ចម្លើយត្រឹមត្រូវ។")

    def test_gemini_request_has_no_frequency_penalty(self):
        client = Mock()
        response = Stream([delta("ចម្លើយ។")])
        client.chat.completions.create.return_value = response
        with patch("khmer_rag.generation.stream.USES_LOCAL_LLAMA_SERVER", False):
            self.assertEqual(
                list(stream_answer(client, "test", "prompt", 512, Event())), ["ចម្លើយ។"]
            )
        self.assertNotIn("frequency_penalty", client.chat.completions.create.call_args.kwargs)
        self.assertTrue(response.closed)

    def test_local_generator_keeps_repetition_penalty(self):
        client = Mock()
        client.chat.completions.create.return_value = Stream([])
        with patch("khmer_rag.generation.stream.USES_LOCAL_LLAMA_SERVER", True):
            list(stream_answer(client, "local", "prompt", 512, Event()))
        self.assertIn("frequency_penalty", client.chat.completions.create.call_args.kwargs)

    @patch("khmer_rag.generation.stream.GEN_RETRY_BACKOFF", 0)
    def test_retry_before_first_output_closes_failed_stream(self):
        client = Mock()
        failed = Stream([RuntimeError("temporary")])
        success = Stream([delta("ចម្លើយ។")])
        client.chat.completions.create.side_effect = [failed, success]
        self.assertEqual(list(stream_answer(client, "test", "prompt", 512, Event())), ["ចម្លើយ។"])
        self.assertTrue(failed.closed and success.closed)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_partial_output_is_never_retried(self):
        client = Mock()
        response = Stream([delta("ចម្លើយ"), RuntimeError("interrupted")])
        client.chat.completions.create.return_value = response
        iterator = stream_answer(client, "test", "prompt", 512, Event())
        self.assertEqual(next(iterator), "ចម្លើយ")
        with self.assertRaises(RuntimeError):
            next(iterator)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertTrue(response.closed)

    def test_cancellation_closes_active_stream(self):
        client = Mock()
        response = Stream([delta("a"), delta("b")])
        client.chat.completions.create.return_value = response
        stop = Event()
        iterator = stream_answer(client, "test", "prompt", 512, stop)
        self.assertEqual(next(iterator), "a")
        stop.set()
        self.assertEqual(list(iterator), [])
        self.assertTrue(response.closed)
