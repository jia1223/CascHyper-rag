import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rq6_evidence.candidates import LexicalSentenceRetriever, build_annotation_packages
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

    def test_candidate_packages_are_separate_for_two_annotators(self):
        corpus = {
            "sentences": [
                {"sentence_id": "s1", "document_id": "d1", "canonical_topic_id": "d1:root", "char_start": 0, "char_end": 17, "text": "Laser gyros use light."},
                {"sentence_id": "s2", "document_id": "d1", "canonical_topic_id": "d1:root", "char_start": 18, "char_end": 40, "text": "Mechanical gyros resist rotation."},
            ]
        }
        split = {"items": [{"question_id": "physics_s2_001", "stage": 2, "source_index": 0, "question": "Why use a laser gyro?"}]}
        with TemporaryDirectory() as directory:
            result = build_annotation_packages(corpus, split, {1: [], 2: ["Laser gyros use optics."], 3: []}, directory, 2)
            self.assertEqual(result["question_count"], 1)
            self.assertTrue((Path(directory) / "annotator_A" / "packages" / "physics_s2_001.md").exists())
            self.assertTrue((Path(directory) / "annotator_B" / "packages" / "physics_s2_001.md").exists())

    def test_lexical_retriever_prioritizes_stage_reference_terms(self):
        retriever = LexicalSentenceRetriever([
            {"sentence_id": "s1", "text": "A laser gyro uses an optical principle."},
            {"sentence_id": "s2", "text": "A flywheel stores kinetic energy."},
        ])
        self.assertEqual(retriever.retrieve("optical laser gyro", 1)[0]["sentence_id"], "s1")


if __name__ == "__main__":
    unittest.main()
