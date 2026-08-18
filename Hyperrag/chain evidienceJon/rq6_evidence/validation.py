"""Validation of manually adjudicated RQ6 gold evidence chains."""

from __future__ import annotations

from typing import Any

import hashlib
import tiktoken

from .native_protocol import NATIVE_EVIDENCE_PROTOCOL, native_retrieval_config
from .matched_final_protocol import MATCHED_FINAL_CONTEXT_PROTOCOL


def validate_latent_topic_manifest(
    topic_manifest: dict[str, Any], sentence_ids: set[str]
) -> list[str]:
    """Validate the fixed Casc latent-topic-to-canonical-span lookup."""
    errors: list[str] = []
    if topic_manifest.get("method") != "CascHyper-RAG":
        errors.append("latent topic manifest: method must be 'CascHyper-RAG'")
    if topic_manifest.get("topic_namespace") != "v81_latent_topic":
        errors.append("latent topic manifest: unsupported topic_namespace")
    topics = topic_manifest.get("topics")
    if not isinstance(topics, dict) or not topics:
        return errors + ["latent topic manifest: topics must be a non-empty object"]
    for topic_id, topic in topics.items():
        if not str(topic_id).startswith("latent:"):
            errors.append(f"latent topic manifest: invalid topic ID {topic_id!r}")
        chunk_ids = topic.get("chunk_ids") if isinstance(topic, dict) else None
        if not isinstance(chunk_ids, list) or not chunk_ids:
            errors.append(f"latent topic manifest: {topic_id!r} needs chunk_ids")
        spans = topic.get("source_sentence_ids") if isinstance(topic, dict) else None
        if not isinstance(spans, list) or not spans:
            errors.append(f"latent topic manifest: {topic_id!r} needs source_sentence_ids")
            continue
        unknown = sorted({str(item) for item in spans} - sentence_ids)
        if unknown:
            errors.append(f"latent topic manifest: {topic_id!r} has unknown sentence IDs {unknown[:3]}")
    return errors


def validate_latent_trace_topics(
    traces: list[dict[str, Any]], topic_manifest: dict[str, Any]
) -> list[str]:
    """Ensure every selected trace topic is defined by the fixed candidate manifest."""
    known_topics = {str(item) for item in topic_manifest["topics"]}
    errors: list[str] = []
    for index, trace in enumerate(traces):
        selected_topics = trace.get("selected_topic_ids", [])
        if not isinstance(selected_topics, list):
            errors.append(f"trace[{index}].selected_topic_ids must be a list")
            continue
        unknown = sorted({str(item) for item in selected_topics} - known_topics)
        if unknown:
            errors.append(f"trace[{index}]: selected topics are absent from the latent topic manifest: {unknown[:3]}")
    return errors


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


def validate_final_context_traces(
    traces: list[dict[str, Any]], sentence_ids: set[str], expected_question_ids: set[str], expected_method: str
) -> list[str]:
    """Validate final generator-context spans without accepting inferred fallback fields."""
    errors = validate_traces(traces, sentence_ids, expected_question_ids, expected_method)
    for index, trace in enumerate(traces):
        units = trace.get("final_context_units")
        prefix = f"trace[{index}].final_context_units"
        if not isinstance(units, list):
            errors.append(f"{prefix}: missing explicit generator-context units")
            continue
        for unit_index, unit in enumerate(units):
            spans = unit.get("source_sentence_ids") if isinstance(unit, dict) else None
            if not isinstance(spans, list) or not spans:
                errors.append(f"{prefix}[{unit_index}]: missing canonical source sentence IDs")
                continue
            unknown = sorted(set(map(str, spans)) - sentence_ids)
            if unknown:
                errors.append(f"{prefix}[{unit_index}]: unknown sentence IDs {unknown[:3]}")
    return errors


def validate_matched_final_context_traces(
    traces: list[dict[str, Any]],
    sentence_ids: set[str],
    expected_question_ids: set[str],
    expected_method: str,
    token_budget: int,
) -> list[str]:
    """Validate the auditable shared final source-evidence budget contract."""
    errors = validate_final_context_traces(traces, sentence_ids, expected_question_ids, expected_method)
    for index, trace in enumerate(traces):
        prefix = f"trace[{index}]"
        diagnostics = trace.get("trace_diagnostics")
        if not isinstance(diagnostics, dict):
            errors.append(f"{prefix}.trace_diagnostics: missing matched final-context diagnostics")
            continue
        if diagnostics.get("final_context_protocol") != MATCHED_FINAL_CONTEXT_PROTOCOL:
            errors.append(f"{prefix}.trace_diagnostics: missing matched final-context protocol marker")
        if diagnostics.get("source_text_budget_tokens") != token_budget:
            errors.append(f"{prefix}.trace_diagnostics: source token budget differs from the frozen shared cap")
        units = trace.get("final_context_units", [])
        unmapped_units = trace.get("unmapped_final_context_units", [])
        if not isinstance(unmapped_units, list):
            errors.append(f"{prefix}.unmapped_final_context_units: must be a list when present")
            continue
        audit_units = [*units, *unmapped_units]
        unit_tokens = [unit.get("source_token_count") for unit in audit_units if isinstance(unit, dict)]
        if len(unit_tokens) != len(audit_units) or any(not isinstance(count, int) or count < 0 for count in unit_tokens):
            errors.append(f"{prefix}.final_context units: every mapped or unmapped unit needs a non-negative source_token_count")
            continue
        encoder = tiktoken.encoding_for_model("gpt-4o")
        for unit_index, unit in enumerate(audit_units):
            text = unit.get("source_text") if isinstance(unit, dict) else None
            if not isinstance(text, str):
                errors.append(f"{prefix}.final_context audit unit[{unit_index}]: missing source_text for independent token audit")
                continue
            if len(encoder.encode(text)) != unit["source_token_count"]:
                errors.append(f"{prefix}.final_context audit unit[{unit_index}]: source_token_count does not match GPT-4o tokenization")
            if unit.get("source_text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                errors.append(f"{prefix}.final_context audit unit[{unit_index}]: source_text_sha256 does not match source_text")
        selected_tokens = diagnostics.get("selected_source_tokens")
        if sum(unit_tokens) != selected_tokens:
            errors.append(f"{prefix}.trace_diagnostics: selected_source_tokens does not equal the unit-token sum")
        if not isinstance(selected_tokens, int) or selected_tokens > token_budget:
            errors.append(f"{prefix}.trace_diagnostics: selected source evidence exceeds the frozen shared cap")
    return errors


def validate_native_evidence_traces(
    traces: list[dict[str, Any]], sentence_ids: set[str], expected_question_ids: set[str], expected_method: str
) -> list[str]:
    """Require the frozen RQ1/RQ2 retrieval configuration in each native-evidence trace."""
    errors = validate_traces(traces, sentence_ids, expected_question_ids, expected_method)
    for index, trace in enumerate(traces):
        diagnostics = trace.get("trace_diagnostics")
        prefix = f"trace[{index}].trace_diagnostics"
        if not isinstance(diagnostics, dict) or diagnostics.get("rq6_protocol") != NATIVE_EVIDENCE_PROTOCOL:
            errors.append(f"{prefix}: missing RQ1/RQ2 native-evidence protocol marker")
            continue
        config = diagnostics.get("native_retrieval_config")
        if not isinstance(config, dict):
            errors.append(f"{prefix}: missing native retrieval configuration")
            continue
        expected = native_retrieval_config(expected_method)
        if config != expected:
            errors.append(f"{prefix}: native retrieval configuration does not match the RQ1/RQ2 setup")
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
        eligible = item.get("eligible_for_full_chain")
        if not isinstance(eligible, bool):
            errors.append(f"{prefix}: eligible_for_full_chain must be true or false")
        hop_numbers = [hop.get("hop") for hop in hops]
        if sorted(hop_numbers) != list(range(1, len(hops) + 1)):
            errors.append(f"{prefix}: gold_hops must have consecutive hop numbers beginning at 1")
        for hop_index, hop in enumerate(hops):
            sentence_id = hop.get("sentence_id")
            if sentence_id not in sentence_ids:
                errors.append(f"{prefix}.gold_hops[{hop_index}]: unknown sentence_id {sentence_id}")
        if eligible is True:
            expected_hops = stage
            expected_bridges = max(stage - 1, 0)
            if len(hops) != expected_hops:
                errors.append(f"{prefix}: eligible Stage {stage} item needs {expected_hops} gold hops")
            if len(bridges) != expected_bridges:
                errors.append(f"{prefix}: eligible Stage {stage} item needs {expected_bridges} bridges")
        elif eligible is False and (hops or bridges):
            errors.append(f"{prefix}: ineligible item must use empty gold_hops and bridges")
        for bridge_index, bridge in enumerate(bridges):
            entity = bridge.get("canonical_entity") or bridge.get("entity_id")
            if not entity:
                errors.append(f"{prefix}.bridges[{bridge_index}]: missing canonical_entity")
            left, right = bridge.get("from_hop"), bridge.get("to_hop")
            if not isinstance(left, int) or not isinstance(right, int) or right != left + 1:
                errors.append(f"{prefix}.bridges[{bridge_index}]: bridge must connect adjacent hops")
    return errors
