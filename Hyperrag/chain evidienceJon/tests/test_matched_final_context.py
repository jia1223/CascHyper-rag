import unittest
import hashlib

from rq6_evidence.matched_final_protocol import select_complete_source_units
from rq6_evidence.validation import validate_matched_final_context_traces


class MatchedFinalContextProtocolTests(unittest.TestCase):
    def test_keeps_complete_units_in_declared_order_until_the_shared_cap(self):
        candidates = [
            {"source_id": "coarse-1", "text": "aaaa"},
            {"source_id": "coarse-2", "text": "bb"},
            {"source_id": "hop-1", "text": "ccc"},
        ]

        selected, used = select_complete_source_units(candidates, 6, lambda text: len(text))

        self.assertEqual([item["source_id"] for item in selected], ["coarse-1", "coarse-2"])
        self.assertEqual(used, 6)

    def test_stops_instead_of_skipping_an_over_budget_unit(self):
        candidates = [
            {"source_id": "coarse-1", "text": "aaaa"},
            {"source_id": "coarse-2", "text": "bbb"},
            {"source_id": "hop-1", "text": "c"},
        ]

        selected, used = select_complete_source_units(candidates, 6, lambda text: len(text))

        self.assertEqual([item["source_id"] for item in selected], ["coarse-1"])
        self.assertEqual(used, 4)

    def test_rejects_trace_whose_recorded_units_exceed_the_shared_cap(self):
        traces = [{
            "question_id": "q1",
            "method": "Hyper-RAG",
            "retrieved_chunks": [],
            "retrieved_evidence_units": [],
            "retrieved_bridges": [],
            "final_context_units": [{"source_sentence_ids": ["s1"], "source_token_count": 7}],
            "trace_diagnostics": {
                "final_context_protocol": "rq6_matched_final_source_evidence_v1",
                "source_text_budget_tokens": 6,
                "selected_source_tokens": 7,
            },
        }]

        errors = validate_matched_final_context_traces(traces, {"s1"}, {"q1"}, "Hyper-RAG", 6)

        self.assertTrue(any("exceeds" in error for error in errors))

    def test_recomputes_token_count_from_the_auditable_source_text(self):
        text = "A source unit can be independently audited."
        trace = {
            "question_id": "q1",
            "method": "CascHyper-RAG",
            "retrieved_chunks": [],
            "retrieved_evidence_units": [],
            "retrieved_bridges": [],
            "final_context_units": [{
                "source_sentence_ids": ["s1"],
                "source_text": text,
                "source_token_count": 1,
                "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }],
            "trace_diagnostics": {
                "final_context_protocol": "rq6_matched_final_source_evidence_v1",
                "source_text_budget_tokens": 100,
                "selected_source_tokens": 1,
            },
        }

        errors = validate_matched_final_context_traces([trace], {"s1"}, {"q1"}, "CascHyper-RAG", 100)

        self.assertTrue(any("tokenization" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
