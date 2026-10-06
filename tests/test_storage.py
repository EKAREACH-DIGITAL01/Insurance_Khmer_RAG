import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from khmer_rag import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        override = patch.object(storage, "DB_PATH", str(Path(temporary.name) / "history.db"))
        override.start()
        self.addCleanup(override.stop)
        storage.init_db()

    def test_round_trip_feedback_and_cascade_delete(self):
        conversation = storage.create_conversation("សំណួរ")
        message = {
            "id": "test-message",
            "question": "សួស្តី",
            "answer": "ជម្រាបសួរ!",
            "best_score": np.float32(0.5),
        }
        storage.append_message(conversation, message)
        self.assertEqual(storage.load_messages(conversation)[0]["best_score"], 0.5)
        self.assertTrue(storage.record_feedback("test-message", "up"))
        self.assertEqual(storage.load_message("test-message")["feedback"], "up")
        self.assertTrue(storage.record_feedback("test-message", "down"))
        self.assertEqual(storage.load_message("test-message")["feedback"], "down")
        storage.delete_conversation(conversation)
        self.assertIsNone(storage.load_message("test-message"))
        with storage.get_db() as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM feedback_events").fetchone()[0], 0
            )

    def test_invalid_feedback_is_ignored(self):
        self.assertFalse(storage.record_feedback("missing", "invalid"))
