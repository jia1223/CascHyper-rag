import hashlib
import asyncio
import tempfile
import unittest
from pathlib import Path

from rq6_evidence.trace_replay import CanonicalSentenceMapper, ProvenanceError, _casc_trace_for_query, _checkpoint_trace, _load_checkpoint_trace, _selected_topic_candidate_chunk_ids, _selected_v81_topics, _tree_digest, _truncate_casc_chunks, _truncate_hyper_units, _unit_bridges


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
                self.chunks = {}

            async def search(self, _question, **_kwargs):
                self.loop_ids.append(id(asyncio.get_running_loop()))
                print("Selected Topics: [1]")
                return {"top_chunks": [], "hop1": [], "hop2": []}

        async def replay_two_questions(engine):
            first = await _casc_trace_for_query(engine, "q1", "first", None, {})
            second = await _casc_trace_for_query(engine, "q2", "second", None, {})
            return [first, second]

        engine = Engine()
        traces = asyncio.run(replay_two_questions(engine))
        self.assertEqual([trace["question_id"] for trace in traces], ["q1", "q2"])
        self.assertEqual(traces[0]["selected_topic_ids"], ["latent:1"])
        self.assertEqual(len(set(engine.loop_ids)), 1)

    def test_native_evidence_trace_uses_the_rq1_rq2_cascaded_configuration(self):
        class Engine:
            chunks = {}

            async def search(self, _question, **kwargs):
                self.search_kwargs = kwargs
                print("Selected Topics: [1]")
                return {"top_chunks": [], "hop1": [], "hop2": []}

            async def verify_results(self, _question, results):
                self.verified = True
                return results

        engine = Engine()
        trace = asyncio.run(_casc_trace_for_query(engine, "q1", "question", None, {}, native_evidence=True))
        self.assertTrue(engine.verified)
        self.assertEqual(engine.search_kwargs, {"top_k_chunks": 5, "top_k_sents": 10, "enable_multi_hop": True})
        self.assertEqual(trace["trace_diagnostics"]["native_retrieval_config"], {
            "top_k_chunks": 5, "top_k_sents_per_hop": 10,
            "enable_multi_hop": True, "consistency_verification": False,
        })

    def test_selected_latent_topics_define_candidate_chunk_union(self):
        class Chunk:
            def __init__(self, memberships):
                self.topic_memberships = memberships

        class Engine:
            chunks = {1: Chunk([1, 2]), 2: Chunk([2]), 3: Chunk([3])}

        self.assertEqual(_selected_topic_candidate_chunk_ids(Engine(), {2}), {1, 2})

    def test_maps_hyperrag_precombine_sources_to_canonical_spans(self):
        from rq6_evidence.trace_replay import _source_rows_from_hyper_context
        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
            {"document_id": "doc", "sentence_id": "s2", "char_start": 66, "char_end": len("Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."), "text": "Rotation changes the measured phase accumulated by the two light beams."},
        )
        context = f"""-----Sources-----
```csv
id,content
0,"{text}"
```
"""

        self.assertEqual(
            _source_rows_from_hyper_context([context], CanonicalSentenceMapper(manifest, [text])),
            [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
        )

    def test_maps_hyperrag_sources_when_native_csv_header_has_a_tab(self):
        from rq6_evidence.trace_replay import _source_rows_from_hyper_context
        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
            {"document_id": "doc", "sentence_id": "s2", "char_start": 66, "char_end": len("Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."), "text": "Rotation changes the measured phase accumulated by the two light beams."},
        )
        context = f"""-----Sources-----
```csv
id,\tcontent
0,\t"{text}"
```
"""

        self.assertEqual(
            _source_rows_from_hyper_context([context], CanonicalSentenceMapper(manifest, [text])),
            [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
        )

    def test_decodes_a_quoted_multiline_hyperrag_source_row(self):
        from rq6_evidence.trace_replay import _source_texts_from_hyper_context

        context = '''-----Sources-----
```csv
id,content
0,"First line.
Second line."
```
'''

        self.assertEqual(_source_texts_from_hyper_context([context]), ["First line.\nSecond line."])

    def test_captures_hyperrag_final_sources_from_the_complete_combined_context(self):
        from types import SimpleNamespace
        from rq6_evidence.trace_replay import _capture_hyper_final_context_units

        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
            {"document_id": "doc", "sentence_id": "s2", "char_start": 66, "char_end": len("Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."), "text": "Rotation changes the measured phase accumulated by the two light beams."},
        )
        source_context = f'''-----Sources-----
```csv
id,content
0,"{text}"
```
'''
        operate = SimpleNamespace(combine_contexts=lambda relation_context, entity_context: relation_context)

        class Rag:
            async def aquery(self, _question, _param):
                operate.combine_contexts(source_context, "")

        units = asyncio.run(
            _capture_hyper_final_context_units(Rag(), operate, "question", None, CanonicalSentenceMapper(manifest, [text]))
        )
        self.assertEqual(units, [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}])

    def test_accepts_an_empty_hyperrag_final_source_context(self):
        from rq6_evidence.trace_replay import _source_rows_from_hyper_context

        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
        )
        self.assertEqual(_source_rows_from_hyper_context([""], CanonicalSentenceMapper(manifest, [text])), [])

    def test_maps_chunk_and_engine_sentence_using_document_spans(self):
        manifest, text = _manifest(
            {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 65, "text": "Laser gyros use optical interference to measure rotation precisely."},
            {"document_id": "doc", "sentence_id": "s2", "char_start": 66, "char_end": len("Laser gyros use optical interference to measure rotation precisely. Rotation changes the measured phase accumulated by the two light beams."), "text": "Rotation changes the measured phase accumulated by the two light beams."},
        )
        mapper = CanonicalSentenceMapper(manifest, [text])
        self.assertEqual(mapper.chunk_sentence_ids("doc", text), ["s1", "s2"])
        self.assertEqual(mapper.sentence_ids("doc", text, "Rotation changes the measured phase accumulated by the two light beams."), ["s2"])

    def test_maps_engine_sentence_with_formatting_only_differences_by_normalized_span(self):
        text = "A _laser_ gyro measures phase. Rotation changes phase."
        manifest = {
            "documents": [{"document_id": "doc", "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}],
            "sentences": [
                {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 30, "text": "A _laser_ gyro measures phase."},
                {"document_id": "doc", "sentence_id": "s2", "char_start": 31, "char_end": len(text), "text": "Rotation changes phase."},
            ],
        }
        mapper = CanonicalSentenceMapper(manifest, [text])

        self.assertEqual(mapper.engine_sentence_ids(0, text, "A laser gyro measures phase."), ["s1"])

    def test_rejects_ambiguous_duplicate_retrieval_text(self):
        text = "Repeat source. Repeat source."
        manifest = {
            "documents": [{"document_id": "doc", "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}],
            "sentences": [{"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 14, "text": "Repeat source."}],
        }
        mapper = CanonicalSentenceMapper(manifest, [text])
        with self.assertRaises(ProvenanceError):
            mapper.chunk_sentence_ids("doc", "Repeat source.")

    def test_maps_duplicate_baseline_chunk_using_its_stable_occurrence(self):
        text = "Repeat source. Distinct middle. Repeat source."
        manifest = {
            "documents": [{"document_id": "doc", "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}],
            "sentences": [
                {"document_id": "doc", "sentence_id": "s1", "char_start": 0, "char_end": 14, "text": "Repeat source."},
                {"document_id": "doc", "sentence_id": "s2", "char_start": 15, "char_end": 31, "text": "Distinct middle."},
                {"document_id": "doc", "sentence_id": "s3", "char_start": 32, "char_end": len(text), "text": "Repeat source."},
            ],
        }
        mapper = CanonicalSentenceMapper(manifest, [text])
        self.assertEqual(mapper.chunk_sentence_ids_at_occurrence("doc", "Repeat source.", 1, 2), ["s3"])

    def test_reads_actual_v81_topic_routing_log(self):
        self.assertEqual(_selected_v81_topics("Selected Topics: [2, 7] (Scores: [0.9, 0.8])"), [2, 7])

    def test_entity_surface_match_respects_word_boundaries(self):
        from rq6_evidence.trace_replay import _entity_surface_matches
        self.assertTrue(_entity_surface_matches("ring", "The ring expands."))
        self.assertFalse(_entity_surface_matches("ring", "The spring expands."))

    def test_casc_evidence_unit_exports_undirected_entity_connection(self):
        bridges = _unit_bridges(1, ["ring laser gyros"], ["s1", "s2"])
        self.assertEqual(bridges, [{
            "rank": 1,
            "canonical_entity": "ring laser gyros",
            "from_sentence_id": "s1",
            "to_sentence_id": "s2",
        }])

    def test_matched_budget_keeps_native_unit_order_without_overflow(self):
        units = [
            ("r1", {"content": "aaa"}),
            ("r2", {"content": "bb"}),
            ("e1", {"content": "cccc"}),
        ]
        selected, used = _truncate_hyper_units(units, 5, lambda text: len(text))
        self.assertEqual([item[0] for item in selected], ["r1", "r2"])
        self.assertEqual(used, 5)

    def test_matched_budget_caps_casc_chunks_in_native_rank_order(self):
        chunks = [
            {"chunk_id": 1, "score": 0.9},
            {"chunk_id": 2, "score": 0.8},
            {"chunk_id": 3, "score": 0.7},
        ]
        texts = {1: "aaa", 2: "bb", 3: "cccc"}
        selected, used = _truncate_casc_chunks(chunks, 5, lambda item: texts[item["chunk_id"]], lambda text: len(text))
        self.assertEqual([item["chunk_id"] for item in selected], [1, 2])
        self.assertEqual(used, 5)

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

    def test_checkpoint_never_cross_reuses_matched_protocol(self):
        item = {"question_id": "q1", "stage": 2}
        trace = {"question_id": "q1", "method": "Hyper-RAG", "retrieved_chunks": [], "retrieved_evidence_units": [], "retrieved_bridges": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint_trace(root, "Hyper-RAG", item, trace, {"s1"}, "manifest", "split", "native")
            self.assertIsNone(_load_checkpoint_trace(root, "Hyper-RAG", item, {"s1"}, "manifest", "split", "rq6_matched_source_text_budget_v1:6000"))


if __name__ == "__main__":
    unittest.main()
