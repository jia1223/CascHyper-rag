import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rq6_evidence.adjudication import build_adjudication_queue


class AdjudicationTests(unittest.TestCase):
    def test_writes_only_disputed_items_with_a_decision_draft(self):
        corpus = {
            "sentences": [
                {"sentence_id": "s1", "document_id": "d1", "canonical_topic_id": "d1:t1", "text": "First fact."},
                {"sentence_id": "s2", "document_id": "d1", "canonical_topic_id": "d1:t1", "text": "Second fact."},
                {"sentence_id": "s3", "document_id": "d2", "canonical_topic_id": "d2:t2", "text": "Alternative fact."},
            ]
        }
        split = {"items": [
            {"question_id": "q_same", "stage": 1, "question": "Same question?"},
            {"question_id": "q_disputed", "stage": 2, "question": "Disputed question?"},
        ]}
        common = {"eligible_for_full_chain": True, "gold_hops": [{"hop": 1, "sentence_id": "s1"}], "bridges": [], "gold_topics": ["d1:t1"]}
        annotator_a = {
            "q_same": {"question_id": "q_same", "stage": 1, **common},
            "q_disputed": {"question_id": "q_disputed", "stage": 2, "eligible_for_full_chain": True,
                           "gold_hops": [{"hop": 1, "sentence_id": "s1"}, {"hop": 2, "sentence_id": "s2"}],
                           "bridges": [{"canonical_entity": "shared entity", "from_hop": 1, "to_hop": 2}], "gold_topics": ["d1:t1"]},
        }
        annotator_b = {
            "q_same": {"question_id": "q_same", "stage": 1, **common},
            "q_disputed": {"question_id": "q_disputed", "stage": 2, "eligible_for_full_chain": True,
                           "gold_hops": [{"hop": 1, "sentence_id": "s1"}, {"hop": 2, "sentence_id": "s3"}],
                           "bridges": [{"canonical_entity": "alternate entity", "from_hop": 1, "to_hop": 2}], "gold_topics": ["d2:t2"]},
        }
        with TemporaryDirectory() as directory:
            result = build_adjudication_queue(corpus, split, annotator_a, annotator_b, directory)
            self.assertEqual(result["disputed_question_count"], 1)
            output = Path(directory)
            self.assertFalse((output / "packages" / "q_same.md").exists())
            self.assertTrue((output / "packages" / "q_disputed.md").exists())
            decision = json.loads((output / "decisions" / "q_disputed.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["allowed_decisions"], ["use_A", "use_B", "revised", "ineligible"])
            queue = json.loads((output / "adjudication_queue.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(queue["question_id"], "q_disputed")
            self.assertEqual(queue["differences"], ["gold_hops", "bridges", "gold_topics"])
            markdown = (output / "packages" / "q_disputed.md").read_text(encoding="utf-8")
            self.assertIn("Alternative fact.", markdown)

    def test_rejects_a_nonempty_output_directory(self):
        corpus = {"sentences": []}
        split = {"items": []}
        with TemporaryDirectory() as directory:
            Path(directory, "existing_decision.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                build_adjudication_queue(corpus, split, {}, {}, directory)


if __name__ == "__main__":
    unittest.main()
