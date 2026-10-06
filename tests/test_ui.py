import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from khmer_rag import storage
from khmer_rag.language import GREETING_RESPONSE_TEXT
from khmer_rag.ui import application
from tests.test_retrieval import chunk

ROOT = Path(__file__).resolve().parents[1]


class UITests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        folder = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.object(storage, "DB_PATH", str(Path(folder) / "history.db")))
        self.generator = stack.enter_context(
            patch.object(application, "get_loaded_model", return_value="test-model")
        )
        stack.enter_context(patch.object(application, "OpenAI"))
        stack.enter_context(
            patch.object(application, "load_embedding_model_info", return_value={"label": "test"})
        )
        stack.enter_context(patch.object(application, "load_embedding_model", return_value=None))
        stack.enter_context(patch.object(application, "load_index", return_value=None))
        stack.enter_context(patch.object(application, "load_metadata", return_value=[]))
        stack.enter_context(patch.object(application, "load_bm25_index", return_value=None))
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10)

    def test_greeting_persists_without_sources_or_action_row(self):
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.app.session_state["pending_question"] = "សួស្ដី"
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        messages = self.app.session_state["messages"]
        self.assertEqual(messages[-1]["answer"], GREETING_RESPONSE_TEXT)
        self.assertTrue(messages[-1]["no_sources"])
        html = "\n".join(element.value for element in self.app.markdown)
        self.assertNotIn('<div class="kc-actions">', html)
        self.assertNotIn('<div class="kc-sources-title">', html)
        self.assertEqual(storage.load_message(messages[-1]["id"])["answer"], GREETING_RESPONSE_TEXT)

    def test_diagnostics_can_open_without_generator(self):
        self.app.query_params["view"] = "diagnostics"
        self.app.query_params["msg"] = "missing"
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.generator.assert_not_called()
        self.assertTrue(any("No diagnostics" in element.value for element in self.app.markdown))

    def test_normal_answer_has_four_actions_and_escaped_copy_payload(self):
        self.app.run()
        message = {
            "id": "normal-message",
            "question": "សំណួរ",
            "answer": 'ចម្លើយ "quoted" <button>។',
            "best_score": 0.9,
            "used_sources": [(1, chunk("law", source_url="https://irc.gov.kh/law.pdf"))],
            "all_results": [],
        }
        self.app.session_state["messages"] = [message]
        with patch("khmer_rag.ui.sources.publish_source_pdfs", return_value={}):
            self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        action_row = next(
            element.value
            for element in self.app.markdown
            if '<div class="kc-actions">' in element.value
        )
        self.assertEqual(action_row.count("data-tip="), 4)
        self.assertIn("data-copy-b64=", action_row)
        self.assertNotIn("quoted", action_row)
        self.assertTrue(
            any("https://irc.gov.kh/law.pdf" in element.value for element in self.app.markdown)
        )
