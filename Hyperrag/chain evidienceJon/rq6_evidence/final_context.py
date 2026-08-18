"""Score gold evidence chains against generator-visible source context."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .metrics import gold_entity_forms


def _final_context_support(trace: dict[str, Any]) -> set[str]:
    support: set[str] = set()
    for unit in trace.get("final_context_units", []):
        support.update(str(item) for item in unit.get("source_sentence_ids", []))
    return support


def _gold_sentences(gold: dict[str, Any]) -> set[str]:
    return {str(item["sentence_id"]) for item in gold.get("gold_hops", [])}


def _entity_is_visible(entity_forms: set[str], support: set[str], sentence_texts: dict[str, str]) -> bool:
    """Require a whole entity surface to occur in the actual final source text."""
    for entity in entity_forms:
        pattern = r"(?<![a-z0-9])" + re.escape(entity) + r"(?![a-z0-9])"
        if any(re.search(pattern, sentence_texts[sentence_id].casefold()) for sentence_id in support):
            return True
    return False


def _bridge_recall(gold: dict[str, Any], support: set[str], sentence_texts: dict[str, str]) -> float:
    bridges = gold.get("bridges", [])
    if not bridges:
        return 1.0
    hop_sentences = {item["hop"]: str(item["sentence_id"]) for item in gold.get("gold_hops", [])}
    recovered = 0
    for bridge in bridges:
        endpoints = frozenset((
            hop_sentences.get(bridge.get("from_hop"), ""),
            hop_sentences.get(bridge.get("to_hop"), ""),
        ))
        if endpoints.issubset(support) and _entity_is_visible(gold_entity_forms(bridge), support, sentence_texts):
            recovered += 1
    return recovered / len(bridges)


@dataclass(frozen=True)
class FinalContextMetrics:
    question_id: str
    stage: int
    eligible_for_full_chain: bool
    final_context_sentence_recall: float
    final_context_bridge_recall: float
    final_context_full_chain_recall: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_final_context_question(gold: dict[str, Any], trace: dict[str, Any], sentence_texts: dict[str, str]) -> FinalContextMetrics:
    """Evaluate only source evidence explicitly included in the generator context.

    No rank cutoff is applied: this protocol answers whether the final context
    available to the generator contains the adjudicated evidence chain.
    """
    gold_sentences = _gold_sentences(gold)
    support = _final_context_support(trace)
    sentence_recall = len(gold_sentences & support) / len(gold_sentences) if gold_sentences else 0.0
    bridge_recall = _bridge_recall(gold, support, sentence_texts)
    eligible = bool(gold.get("eligible_for_full_chain", False))
    full_chain = None
    if eligible:
        full_chain = float(gold_sentences.issubset(support) and bridge_recall == 1.0)
    return FinalContextMetrics(
        question_id=str(gold["question_id"]),
        stage=int(gold["stage"]),
        eligible_for_full_chain=eligible,
        final_context_sentence_recall=sentence_recall,
        final_context_bridge_recall=bridge_recall,
        final_context_full_chain_recall=full_chain,
    )
