"""Score the complete native retrieval evidence selected under the RQ1/RQ2 setup."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .metrics import gold_entity_forms, source_sentence_ids


def _selected_support(trace: dict[str, Any]) -> set[str]:
    """Return every canonical span selected by the native retrieval pipeline."""
    support: set[str] = set()
    for field in ("retrieved_chunks", "retrieved_evidence_units"):
        for unit in trace.get(field, []):
            if isinstance(unit, dict):
                support.update(source_sentence_ids(unit))
    return support


def _gold_sentences(gold: dict[str, Any]) -> set[str]:
    return {str(item["sentence_id"]) for item in gold.get("gold_hops", [])}


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
        if endpoints.issubset(support) and any(
            re.search(r"(?<![a-z0-9])" + re.escape(entity) + r"(?![a-z0-9])", sentence_texts[sentence_id].casefold())
            for entity in gold_entity_forms(bridge)
            for sentence_id in support
        ):
            recovered += 1
    return recovered / len(bridges)


@dataclass(frozen=True)
class NativeEvidenceMetrics:
    question_id: str
    stage: int
    eligible_for_full_chain: bool
    native_evidence_sentence_recall: float
    native_evidence_bridge_recall: float
    native_evidence_full_chain_recall: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_native_evidence_question(gold: dict[str, Any], trace: dict[str, Any], sentence_texts: dict[str, str]) -> NativeEvidenceMetrics:
    """Evaluate evidence selected by a native RQ1/RQ2 retrieval run, without rank cutoffs."""
    gold_sentences = _gold_sentences(gold)
    support = _selected_support(trace)
    sentence_recall = len(gold_sentences & support) / len(gold_sentences) if gold_sentences else 0.0
    bridge_recall = _bridge_recall(gold, support, sentence_texts)
    eligible = bool(gold.get("eligible_for_full_chain", False))
    full_chain = None
    if eligible:
        full_chain = float(gold_sentences.issubset(support) and bridge_recall == 1.0)
    return NativeEvidenceMetrics(
        question_id=str(gold["question_id"]),
        stage=int(gold["stage"]),
        eligible_for_full_chain=eligible,
        native_evidence_sentence_recall=sentence_recall,
        native_evidence_bridge_recall=bridge_recall,
        native_evidence_full_chain_recall=full_chain,
    )
