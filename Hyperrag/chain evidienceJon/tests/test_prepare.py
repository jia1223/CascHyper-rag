import unittest
from unittest.mock import patch

from rq6_evidence.cli import main
from rq6_evidence.prepare import build_corpus_manifest, build_question_split, split_sentences


class PrepareTests(unittest.TestCase):
    def test_split_sentences_preserves_text_and_offsets(self):
        text = "First fact. Second fact?"
        parts = split_sentences(text)
        self.assertEqual([part[2] for part in parts], ["First fact.", "Second fact?"])
        self.assertEqual(text[parts[1][0]:parts[1][1]], "Second fact?")

    def test_manifest_creates_stable_sentence_ids(self):
        manifest = build_corpus_manifest(["One. Two."])
        self.assertEqual(manifest["documents"][0]["document_id"], "physics_doc_001")
        self.assertEqual(
            [item["sentence_id"] for item in manifest["sentences"]],
            ["physics_doc_001_s_00001", "physics_doc_001_s_00002"],
        )

    def test_manifest_assigns_a_heading_topic_to_following_sentence(self):
        manifest = build_corpus_manifest(["1.1 Heading\nFact after heading."])
        self.assertEqual(manifest["sentences"][-1]["canonical_topic_id"], "physics_doc_001:1.1")

    def test_cli_requires_a_subcommand(self):
        with patch("sys.argv", ["rq6-evidence"]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
