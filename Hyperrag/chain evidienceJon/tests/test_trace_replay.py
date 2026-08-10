import hashlib
import asyncio
import tempfile
import unittest
from pathlib import Path

from rq6_evidence.trace_replay import CanonicalSentenceMapper, ProvenanceError, _casc_trace_for_query, _checkpoint_trace, _load_checkpoint_trace, _selected_v81_topics, _tree_digest


def _manifest(*sentences):
    text = "Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."
    return {
        "documents": [{"document_id": "doc", "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}],
        "sentences": list(sentences),
    }, text


class TraceReplayTests(unittest.TestCase):
    def test_cascaded_queries_share_one_running_event_loop(self):
        class Engine:
            def __init__(self):
                self.loop_ids = []

            async def search(self, _question, **_kwargs):
                self.loop_ids.append(id(asyncio.get_running_loop()))
                print("Selected Topics: [1]")
                return {"top_chunks": [], "hop1": [], "hop2": []}

        async def replay_two_questions(engine):
            first = await _casc_trace_for_query(engine, "q1", "first", None, {}, {1: "topic-1"})
            second = await _casc_trace_for_query(engine, "q2", "second", None, {}, {1: "topic-1"})
            return [first, second]

        engine = Engine()
        traces = asyncio.run(replay_two_questions(engine))
        self.assertEqual([trace["question_id"] for trace in traces], ["q1", "q2"])
        self.assertEqual(len(set(engine.loop_ids)), 1)

    def test_maps_chunk_and_engine_sentence_using_document_spans(self):
        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
            {"document_id": "doc", "sentence_id": "s2", "char_start": 66, "char_end": len("Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."), "text": "Rotation changes the measured phase accumulated by the two light beams."},
        )
        mapper = CanonicalSentenceMapper(manifest, [text])
        self.assertEqual(mapper.chunk_sentence_ids("doc", text), ["s1", "s2"])
        self.assertEqual(mapper.sentence_ids("doc", text, "Rotation changes the measured phase accumulated by the two light beams."), ["s2"])

    def test_rejects_ambiguous_duplicate_retrieval_text(self):
        text = "Repeat source. Repeat source."
        manifest = {
            "documents": [{"document_id": "doc", "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}],
            "sentences": [{"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 14, "text": "Repeat source."}],
        }
        mapper = CanonicalSentenceMapper(manifest, [text])
        with self.assertRaises(ProvenanceError):
            mapper.chunk_sentence_ids("doc", "Repeat source.")

    def test_reads_actual_v81_topic_routing_log(self):
        self.assertEqual(_selected_v81_topics("Selected Topics: [2, 7] (Scores: [0.9, 0.8])"), [2, 7])

    def test_entity_surface_match_respects_word_boundaries(self):
        from rq6_evidence.trace_replay import _entity_surface_matches
        self.assertTrue(_entity_surface_matches("ring", "The ring expands."))
        self.assertFalse(_entity_surface_matches("ring", "The spring expands."))

    def test_hashes_a_string_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.bin"
            path.write_bytes(b"frozen index")
            self.assertEqual(_tree_digest(str(path)), _tree_digest(path))

    def test_checkpoint_reuses_only_matching_validated_trace(self):
        item = {"question_id": "q1", "stage": 2}
        trace = {"question_id": "q1", "method": "Hyper-RAG", "retrieved_chunks": [], "retrieved_evidence_units": [], "retrieved_bridges": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint_trace(root, "Hyper-RAG", item, trace, {"s1"}, "manifest", "split")
            self.assertEqual(_load_checkpoint_trace(root, "Hyper-RAG", item, {"s1"}, "manifest", "split"), trace)
            self.assertIsNone(_load_checkpoint_trace(root, "Hyper-RAG", item, {"s1"}, "different", "split"))


if __name__ == "__main__":
    unittest.main()
