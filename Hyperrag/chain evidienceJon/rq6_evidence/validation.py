"""Validation of manually adjudicated RQ6 gold evidence chains."""

from __future__ import annotations

from typing import Any


def validate_question_split(gold_items: list[dict[str, Any]], split: dict[str, Any]) -> list[str]:
    """Ensure an analysis uses exactly the preregistered question IDs/stages."""
    expected = {item["question_id"]: item["stage"] for item in split.get("items", [])}
    observed = {item.get("question_id"): item.get("stage") for item in gold_items}
    errors: list[str] = []
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing:
        errors.append(f"gold is missing {len(missing)} frozen split questions, e.g. {missing[:3]}")
    if unexpected:
        errors.append(f"gold contains {len(unexpected)} questions outside the frozen split, e.g. {unexpected[:3]}")
    mismatched = [qid for qid in set(expected) & set(observed) if expected[qid] != observed[qid]]
    if mismatched:
        errors.append(f"gold has stage mismatches for frozen questions, e.g. {mismatched[:3]}")
    return errors


def validate_traces(
    traces: list[dict[str, Any]],
    sentence_ids: set[str],
    expected_question_ids: set[str],
    expected_method: str,
) -> list[str]:
    """Validate that every external trace uses canonical source sentence IDs."""
    errors: list[str] = []
    seen: set[str] = set()
    for index, trace in enumerate(traces):
        prefix = f"trace[{index}]"
        if trace.get("method") != expected_method:
            errors.append(f"{prefix}: method must be {expected_method!r}, found {trace.get('method')!r}")
        question_id = trace.get("question_id")
        if question_id in seen:
            errors.append(f"{prefix}: duplicate question_id {question_id}")
        seen.add(question_id)
        for unit_group in ("retrieved_chunks", "retrieved_evidence_units"):
            for unit_index, unit in enumerate(trace.get(unit_group, [])):
                spans = unit.get("source_sentence_ids") or unit.get("sentence_ids") or unit.get("source_span_ids")
                if not isinstance(spans, list) or not spans:
                    errors.append(f"{prefix}.{unit_group}[{unit_index}]: missing canonical source sentence IDs")
                    continue
                unknown = sorted(set(map(str, spans)) - sentence_ids)
                if unknown:
                    errors.append(f"{prefix}.{unit_group}[{unit_index}]: unknown sentence IDs {unknown[:3]}")
        for edge_index, edge in enumerate(trace.get("retrieved_bridges", [])):
            if not isinstance(edge, dict):
                errors.append(f"{prefix}.retrieved_bridges[{edge_index}]: bridge must be an object")
                continue
            entity = edge.get("canonical_entity") or edge.get("entity") or edge.get("entity_id")
            endpoints = {edge.get("from_sentence_id"), edge.get("to_sentence_id")}
            if not entity:
                errors.append(f"{prefix}.retrieved_bridges[{edge_index}]: missing bridge entity")
            if None in endpoints or not endpoints.issubset(sentence_ids):
                errors.append(f"{prefix}.retrieved_bridges[{edge_index}]: bridge endpoints must be canonical sentence IDs")
    missing = expected_question_ids - seen
    unexpected = seen - expected_question_ids
    if missing:
        errors.append(f"trace is missing {len(missing)} frozen questions, e.g. {sorted(missing)[:3]}")
    if unexpected:
        errors.append(f"trace contains {len(unexpected)} questions outside the frozen split, e.g. {sorted(unexpected)[:3]}")
    return errors


def validate_gold(gold_items: list[dict[str, Any]], sentence_ids: set[str]) -> list[str]:
    """Return all validation errors; an empty list means the file is usable."""
    errors: list[str] = []
    seen_questions: set[str] = set()
    for item_index, item in enumerate(gold_items):
        prefix = f"gold[{item_index}]"
        question_id = item.get("question_id")
        if not question_id:
            errors.append(f"{prefix}: missing question_id")
            continue
        if question_id in seen_questions:
            errors.append(f"{prefix}: duplicate question_id {question_id}")
        seen_questions.add(question_id)
        stage = item.get("stage")
        if stage not in (1, 2, 3):
            errors.append(f"{prefix}: stage must be 1, 2, or 3")
        hops = item.get("gold_hops", [])
        bridges = item.get("bridges", [])
        hop_numbers = [hop.get("hop") for hop in hops]
        if sorted(hop_numbers) != list(range(1, len(hops) + 1)):
            errors.append(f"{prefix}: gold_hops must have consecutive hop numbers beginning at 1")
        for hop_index, hop in enumerate(hops):
            sentence_id = hop.get("sentence_id")
            if sentence_id not in sentence_ids:
                errors.append(f"{prefix}.gold_hops[{hop_index}]: unknown sentence_id {sentence_id}")
        if item.get("eligible_for_full_chain", False):
            expected_hops = stage
            expected_bridges = max(stage - 1, 0)
            if len(hops) != expected_hops:
                errors.append(f"{prefix}: eligible Stage {stage} item needs {expected_hops} gold hops")
            if len(bridges) != expected_bridges:
                errors.append(f"{prefix}: eligible Stage {stage} item needs {expected_bridges} bridges")
        for bridge_index, bridge in enumerate(bridges):
            entity = bridge.get("canonical_entity") or bridge.get("entity_id")
            if not entity:
                errors.append(f"{prefix}.bridges[{bridge_index}]: missing canonical_entity")
            left, right = bridge.get("from_hop"), bridge.get("to_hop")
            if not isinstance(left, int) or not isinstance(right, int) or right != left + 1:
                errors.append(f"{prefix}.bridges[{bridge_index}]: bridge must connect adjacent hops")
    return errors
