import unittest

from rq6_evidence.native_evidence import evaluate_native_evidence_question
from rq6_evidence.evaluate import evaluate_native_evidence_pair
from rq6_evidence.validation import validate_native_evidence_traces


GOLD = {
    "question_id": "q1",
    "stage": 2,
    "eligible_for_full_chain": True,
    "gold_hops": [{"hop": 1, "sentence_id": "s1"}, {"hop": 2, "sentence_id": "s2"}],
    "bridges": [{"from_hop": 1, "to_hop": 2, "canonical_entity": "spindle rigidity"}],
}
SENTENCE_TEXTS = {"s1": "Spindle rigidity controls the first mechanism.", "s2": "The second mechanism affects accuracy."}


class NativeEvidenceMetricsTests(unittest.TestCase):
    def test_selected_chunks_and_hop_evidence_together_recover_a_complete_chain(self):
        trace = {
            "question_id": "q1",
            "retrieved_chunks": [{"source_sentence_ids": ["s1"]}],
            "retrieved_evidence_units": [{"source_sentence_ids": ["s2"]}],
            "retrieved_bridges": [{
                "canonical_entity": "spindle rigidity",
                "from_sentence_id": "s1",
                "to_sentence_id": "s2",
            }],
        }
        scores = evaluate_native_evidence_question(GOLD, trace, SENTENCE_TEXTS)
        self.assertEqual(scores.native_evidence_sentence_recall, 1.0)
        self.assertEqual(scores.native_evidence_bridge_recall, 1.0)
        self.assertEqual(scores.native_evidence_full_chain_recall, 1.0)

    def test_missing_one_gold_hop_breaks_selected_evidence_chain(self):
        trace = {
            "question_id": "q1",
            "retrieved_chunks": [{"source_sentence_ids": ["s1"]}],
            "retrieved_evidence_units": [],
            "retrieved_bridges": [],
        }
        scores = evaluate_native_evidence_question(GOLD, trace, SENTENCE_TEXTS)
        self.assertEqual(scores.native_evidence_sentence_recall, 0.5)
        self.assertEqual(scores.native_evidence_bridge_recall, 0.0)
        self.assertEqual(scores.native_evidence_full_chain_recall, 0.0)

    def test_bridge_requires_visible_entity_in_the_selected_native_evidence(self):
        trace = {
            "question_id": "q1",
            "retrieved_chunks": [{"source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [],
            "retrieved_bridges": [],
        }
        scores = evaluate_native_evidence_question(GOLD, trace, {"s1": "first endpoint", "s2": "second endpoint"})
        self.assertEqual(scores.native_evidence_sentence_recall, 1.0)
        self.assertEqual(scores.native_evidence_bridge_recall, 0.0)
        self.assertEqual(scores.native_evidence_full_chain_recall, 0.0)

    def test_pair_evaluation_reports_selected_unit_diagnostics(self):
        casc = [{
            "question_id": "q1",
            "method": "CascHyper-RAG",
            "retrieved_chunks": [{"source_sentence_ids": ["s1"]}],
            "retrieved_evidence_units": [{"source_sentence_ids": ["s2"]}],
            "retrieved_bridges": [{"canonical_entity": "spindle rigidity", "from_sentence_id": "s1", "to_sentence_id": "s2"}],
        }]
        hyper = [{
            "question_id": "q1",
            "method": "Hyper-RAG",
            "retrieved_chunks": [{"source_sentence_ids": ["s1"]}],
            "retrieved_evidence_units": [],
            "retrieved_bridges": [],
        }]
        result = evaluate_native_evidence_pair([GOLD], casc, hyper, SENTENCE_TEXTS, 100, 1)
        self.assertEqual(result["summary"]["overall"]["CascHyper-RAG"]["native_evidence_full_chain_recall"]["mean"], 1.0)
        self.assertEqual(result["selection_diagnostics"]["CascHyper-RAG"]["mean_selected_chunks"], 1.0)

    def test_validation_rejects_a_trace_not_replayed_under_the_frozen_native_configuration(self):
        trace = {
            "question_id": "q1", "method": "CascHyper-RAG", "retrieved_chunks": [],
            "retrieved_evidence_units": [], "retrieved_bridges": [], "trace_diagnostics": {},
        }
        errors = validate_native_evidence_traces([trace], {"s1", "s2"}, {"q1"}, "CascHyper-RAG")
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
