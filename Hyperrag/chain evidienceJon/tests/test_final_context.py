import unittest

from rq6_evidence.final_context import evaluate_final_context_question


GOLD = {
    "question_id": "physics_s2_001",
    "stage": 2,
    "eligible_for_full_chain": True,
    "gold_hops": [
        {"hop": 1, "sentence_id": "s1"},
        {"hop": 2, "sentence_id": "s2"},
    ],
    "bridges": [{"canonical_entity": "bridge", "from_hop": 1, "to_hop": 2}],
}

SENTENCE_TEXTS = {"s1": "The bridge begins at the first endpoint.", "s2": "The bridge completes at the second endpoint."}


class FinalContextMetricsTests(unittest.TestCase):
    def test_complete_final_context_recovers_sentence_bridge_and_chain(self):
        trace = {
            "question_id": "physics_s2_001",
            "final_context_units": [
                {"source_sentence_ids": ["s1"]},
                {"source_sentence_ids": ["s2"]},
            ],
            "retrieved_bridges": [
                {"canonical_entity": "bridge", "from_sentence_id": "s1", "to_sentence_id": "s2"},
            ],
        }

        scores = evaluate_final_context_question(GOLD, trace, SENTENCE_TEXTS)

        self.assertEqual(scores.final_context_sentence_recall, 1.0)
        self.assertEqual(scores.final_context_bridge_recall, 1.0)
        self.assertEqual(scores.final_context_full_chain_recall, 1.0)

    def test_final_context_does_not_depend_on_a_second_retrieval_edge(self):
        trace = {
            "question_id": "physics_s2_001",
            "final_context_units": [{"source_sentence_ids": ["s1", "s2"]}],
            "retrieved_bridges": [],
        }

        scores = evaluate_final_context_question(GOLD, trace, SENTENCE_TEXTS)

        self.assertEqual(scores.final_context_sentence_recall, 1.0)
        self.assertEqual(scores.final_context_bridge_recall, 1.0)
        self.assertEqual(scores.final_context_full_chain_recall, 1.0)

    def test_bridge_entity_must_be_visible_in_the_final_context(self):
        trace = {
            "question_id": "physics_s2_001",
            "final_context_units": [{"source_sentence_ids": ["s1", "s2"]}],
            "retrieved_bridges": [{"canonical_entity": "bridge", "from_sentence_id": "s1", "to_sentence_id": "s2"}],
        }

        scores = evaluate_final_context_question(GOLD, trace, {"s1": "first endpoint", "s2": "second endpoint"})

        self.assertEqual(scores.final_context_bridge_recall, 0.0)
        self.assertEqual(scores.final_context_full_chain_recall, 0.0)


if __name__ == "__main__":
    unittest.main()
