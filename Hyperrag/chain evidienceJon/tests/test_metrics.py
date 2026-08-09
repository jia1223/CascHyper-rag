import unittest

from rq6_evidence.evaluate import evaluate_pair
from rq6_evidence.metrics import evaluate_question
from rq6_evidence.validation import validate_gold, validate_question_split, validate_traces


GOLD = {
    "question_id": "physics_s2_001",
    "stage": 2,
    "eligible_for_full_chain": True,
    "gold_hops": [
        {"hop": 1, "sentence_id": "s1"},
        {"hop": 2, "sentence_id": "s2"},
    ],
    "bridges": [{"canonical_entity": "bridge", "from_hop": 1, "to_hop": 2}],
    "gold_topics": ["t1"],
}


class MetricsTests(unittest.TestCase):
    def test_full_chain_requires_all_hops_and_bridge(self):
        trace = {
            "question_id": "physics_s2_001",
            "selected_topic_ids": ["t1"],
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [
                {"rank": 1, "source_sentence_ids": ["s1"]},
                {"rank": 2, "source_sentence_ids": ["s2"]},
            ],
            "retrieved_bridges": [{"canonical_entity": "bridge", "from_sentence_id": "s1", "to_sentence_id": "s2"}],
        }
        scores = evaluate_question(GOLD, trace)
        self.assertEqual(scores.chunk_recall_at_5, 1.0)
        self.assertEqual(scores.sentence_recall_at_10, 1.0)
        self.assertEqual(scores.bridge_recall_at_10, 1.0)
        self.assertEqual(scores.full_chain_at_10, 1.0)

    def test_missing_bridge_breaks_full_chain(self):
        trace = {
            "question_id": "physics_s2_001",
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
        }
        scores = evaluate_question(GOLD, trace)
        self.assertEqual(scores.sentence_recall_at_10, 1.0)
        self.assertEqual(scores.bridge_recall_at_10, 0.0)
        self.assertEqual(scores.full_chain_at_10, 0.0)

    def test_ranked_bridge_after_cutoff_is_not_recalled(self):
        trace = {
            "question_id": "physics_s2_001",
            "selected_topic_ids": [],
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_bridges": [{"canonical_entity": "bridge", "from_sentence_id": "s1", "to_sentence_id": "s2", "rank": 11}],
        }
        scores = evaluate_question(GOLD, trace)
        self.assertEqual(scores.topic_coverage, 0.0)
        self.assertEqual(scores.bridge_recall_at_10, 0.0)

    def test_entity_with_reversed_endpoints_does_not_recover_bridge(self):
        trace = {
            "question_id": "physics_s2_001",
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_bridges": [{"canonical_entity": "bridge", "from_sentence_id": "s2", "to_sentence_id": "s1"}],
        }
        scores = evaluate_question(GOLD, trace)
        self.assertEqual(scores.bridge_recall_at_10, 0.0)

    def test_pair_evaluation_reports_paired_gain(self):
        casc_trace = {
            "question_id": "physics_s2_001",
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1", "s2"]}],
            "retrieved_bridges": [{"canonical_entity": "bridge", "from_sentence_id": "s1", "to_sentence_id": "s2"}],
        }
        hyper_trace = {
            "question_id": "physics_s2_001",
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["s1"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1"]}],
        }
        result = evaluate_pair([GOLD], [casc_trace], [hyper_trace], bootstrap_samples=100, seed=7)
        difference = result["summary"]["overall"]["paired_difference_CascHyper_minus_Hyper"]["full_chain_at_10"]
        self.assertEqual(difference["mean_difference"], 1.0)

    def test_validation_rejects_bad_eligible_stage(self):
        invalid = dict(GOLD)
        invalid["gold_hops"] = [{"hop": 1, "sentence_id": "s1"}]
        errors = validate_gold([invalid], {"s1", "s2"})
        self.assertTrue(any("needs 2 gold hops" in error for error in errors))

    def test_frozen_split_and_canonical_trace_are_enforced(self):
        split = {"items": [{"question_id": "physics_s2_001", "stage": 2}]}
        self.assertEqual(validate_question_split([GOLD], split), [])
        trace = {
            "question_id": "physics_s2_001",
            "retrieved_chunks": [{"rank": 1, "source_sentence_ids": ["not_canonical"]}],
            "retrieved_evidence_units": [{"rank": 1, "source_sentence_ids": ["s1"]}],
        }
        errors = validate_traces([trace], {"s1", "s2"}, {"physics_s2_001"}, "CascHyper-RAG")
        self.assertTrue(any("unknown sentence IDs" in error for error in errors))
        self.assertTrue(any("method must be" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
