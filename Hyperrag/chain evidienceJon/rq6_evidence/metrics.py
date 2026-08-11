"""Metric definitions for method-independent RQ6 retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _as_sentence_ids(unit: dict[str, Any]) -> set[str]:
    for key in ("source_sentence_ids", "sentence_ids", "source_span_ids"):
        value = unit.get(key)
        if isinstance(value, list):
            return {str(item) for item in value}
    return set()


def _ranked_support(units: Iterable[dict[str, Any]], limit: int) -> set[str]:
    ordered = sorted(units, key=lambda item: item.get("rank", 10**9))[:limit]
    supported: set[str] = set()
    for unit in ordered:
        supported.update(_as_sentence_ids(unit))
    return supported


def _retrieved_edges(trace: dict[str, Any], limit: int) -> set[tuple[str, frozenset[str]]]:
    """Return ranked, endpoint-specific *undirected* bridge connections."""
    edges: set[tuple[str, frozenset[str]]] = set()
    for edge in trace.get("retrieved_bridges", []):
        if not isinstance(edge, dict) or int(edge.get("rank", 1)) > limit:
            continue
        entity = edge.get("canonical_entity") or edge.get("entity") or edge.get("entity_id")
        left, right = edge.get("from_sentence_id"), edge.get("to_sentence_id")
        if entity and left and right:
            edges.add((str(entity).casefold(), frozenset((str(left), str(right)))))
    return edges


def _gold_sentence_ids(gold: dict[str, Any]) -> list[str]:
    return [str(hop["sentence_id"]) for hop in gold.get("gold_hops", [])]


def _gold_entity_forms(bridge: dict[str, Any]) -> set[str]:
    """Return the adjudicated canonical bridge name and its approved aliases."""
    forms = {str(bridge.get("canonical_entity") or bridge.get("entity_id") or "").casefold()}
    forms.update(str(alias).casefold() for alias in bridge.get("aliases", []) if alias)
    forms.discard("")
    return forms


def _bridge_recall(gold: dict[str, Any], support: set[str], trace: dict[str, Any], limit: int) -> float:
    bridges = gold.get("bridges", [])
    if not bridges:
        return 1.0
    hop_to_sentence = {hop["hop"]: str(hop["sentence_id"]) for hop in gold.get("gold_hops", [])}
    retrieved_edges = _retrieved_edges(trace, limit)
    recovered = 0
    for bridge in bridges:
        entities = _gold_entity_forms(bridge)
        left = hop_to_sentence.get(bridge.get("from_hop"))
        right = hop_to_sentence.get(bridge.get("to_hop"))
        if any((entity, frozenset((left, right))) in retrieved_edges for entity in entities) and left in support and right in support:
            recovered += 1
    return recovered / len(bridges)


@dataclass(frozen=True)
class QuestionMetrics:
    question_id: str
    stage: int
    eligible_for_full_chain: bool
    topic_coverage: float | None
    chunk_recall_at_5: float
    sentence_recall_at_10: float
    bridge_recall_at_10: float
    full_chain_at_5: float | None
    full_chain_at_10: float | None
    full_chain_at_20: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_question(gold: dict[str, Any], trace: dict[str, Any]) -> QuestionMetrics:
    gold_sentences = set(_gold_sentence_ids(gold))
    chunk_support = _ranked_support(trace.get("retrieved_chunks", []), 5)
    evidence_units = trace.get("retrieved_evidence_units", [])
    support_5 = _ranked_support(evidence_units, 5)
    support_10 = _ranked_support(evidence_units, 10)
    support_20 = _ranked_support(evidence_units, 20)
    chunk_recall = len(gold_sentences & chunk_support) / len(gold_sentences) if gold_sentences else 0.0
    sentence_recall = len(gold_sentences & support_10) / len(gold_sentences) if gold_sentences else 0.0
    bridge_recall = _bridge_recall(gold, support_10, trace, 10)

    gold_topics = set(gold.get("gold_topics", []))
    selected_topics = set(trace.get("selected_topic_ids", []))
    topic_coverage = None
    if gold_topics and "selected_topic_ids" in trace:
        topic_coverage = len(gold_topics & selected_topics) / len(gold_topics)

    eligible = bool(gold.get("eligible_for_full_chain", False))

    def full_chain(support: set[str], limit: int) -> float | None:
        if not eligible:
            return None
        all_hops = gold_sentences.issubset(support)
        all_bridges = _bridge_recall(gold, support, trace, limit) == 1.0
        return float(all_hops and all_bridges)

    return QuestionMetrics(
        question_id=str(gold["question_id"]),
        stage=int(gold["stage"]),
        eligible_for_full_chain=eligible,
        topic_coverage=topic_coverage,
        chunk_recall_at_5=chunk_recall,
        sentence_recall_at_10=sentence_recall,
        bridge_recall_at_10=bridge_recall,
        full_chain_at_5=full_chain(support_5, 5),
        full_chain_at_10=full_chain(support_10, 10),
        full_chain_at_20=full_chain(support_20, 20),
    )
