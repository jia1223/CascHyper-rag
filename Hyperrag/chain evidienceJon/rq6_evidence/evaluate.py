"""End-to-end paired RQ6 evaluation using externally exported traces."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any

from .metrics import evaluate_question
from .statistics import METRICS, paired_bootstrap, summarize
from .final_context import evaluate_final_context_question
from .native_evidence import evaluate_native_evidence_question
from .native_protocol import NATIVE_EVIDENCE_PROTOCOL


FINAL_CONTEXT_METRICS = (
    "final_context_sentence_recall",
    "final_context_bridge_recall",
    "final_context_full_chain_recall",
)

NATIVE_EVIDENCE_METRICS = (
    "native_evidence_sentence_recall",
    "native_evidence_bridge_recall",
    "native_evidence_full_chain_recall",
)


def _trace_index(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for trace in traces:
        question_id = trace.get("question_id")
        if not question_id:
            raise ValueError("Every trace item needs question_id.")
        if question_id in output:
            raise ValueError(f"Duplicate trace question_id: {question_id}")
        output[question_id] = trace
    return output


def score_method(
    gold_items: list[dict[str, Any]], traces: list[dict[str, Any]], topic_manifest: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    indexed = _trace_index(traces)
    missing = [item["question_id"] for item in gold_items if item["question_id"] not in indexed]
    if missing:
        raise ValueError(f"Trace is missing {len(missing)} gold questions, e.g. {missing[:3]}")
    return [
        evaluate_question(item, indexed[item["question_id"]], topic_manifest).as_dict()
        for item in gold_items
    ]


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = rows
    for stage in (1, 2, 3):
        groups[f"stage_{stage}"] = [row for row in rows if row["stage"] == stage]
    groups["eligible_multihop"] = [
        row for row in rows if row["eligible_for_full_chain"] and row["stage"] in (2, 3)
    ]
    return groups


def _provenance_coverage(traces: list[dict[str, Any]]) -> dict[str, int | float]:
    returned = sum(int(item.get("trace_diagnostics", {}).get("returned_evidence_units", len(item.get("retrieved_evidence_units", [])))) for item in traces)
    mapped = sum(int(item.get("trace_diagnostics", {}).get("mapped_evidence_units", len(item.get("retrieved_evidence_units", [])))) for item in traces)
    return {"returned_evidence_units": returned, "mapped_evidence_units": mapped, "coverage": mapped / returned if returned else 1.0}


def _topic_candidate_diagnostics(
    traces: list[dict[str, Any]], topic_manifest: dict[str, Any] | None
) -> dict[str, int | float | None]:
    """Recompute topic-candidate reduction from the immutable topic manifest."""
    if topic_manifest is None:
        return {"n": 0, "mean_candidate_chunk_fraction": None, "mean_candidate_reduction": None}
    topics = topic_manifest["topics"]
    total = int(topic_manifest["total_chunk_count"])
    fractions = []
    for trace in traces:
        candidate_chunk_ids: set[int] = set()
        for topic_id in trace.get("selected_topic_ids", []):
            candidate_chunk_ids.update(int(item) for item in topics[str(topic_id)]["chunk_ids"])
        if total > 0:
            fractions.append(len(candidate_chunk_ids) / total)
    if not fractions:
        return {"n": 0, "mean_candidate_chunk_fraction": None, "mean_candidate_reduction": None}
    mean_fraction = sum(fractions) / len(fractions)
    return {
        "n": len(fractions),
        "mean_candidate_chunk_fraction": mean_fraction,
        "mean_candidate_reduction": 1.0 - mean_fraction,
    }


def evaluate_pair(
    gold_items: list[dict[str, Any]],
    casc_traces: list[dict[str, Any]],
    hyper_traces: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    casc_topic_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    casc_rows = score_method(gold_items, casc_traces, casc_topic_manifest)
    hyper_rows = score_method(gold_items, hyper_traces)
    casc_groups = _group_rows(casc_rows)
    hyper_groups = _group_rows(hyper_rows)
    summary: dict[str, Any] = {}
    for group_name in casc_groups:
        summary[group_name] = {
            "CascHyper-RAG": {metric: summarize(casc_groups[group_name], metric) for metric in METRICS},
            "Hyper-RAG": {metric: summarize(hyper_groups[group_name], metric) for metric in METRICS},
            "paired_difference_CascHyper_minus_Hyper": {
                metric: paired_bootstrap(
                    casc_groups[group_name], hyper_groups[group_name], metric, bootstrap_samples, seed
                )
                for metric in METRICS
            },
        }
    return {
        "schema_version": 1,
        "comparison": "CascHyper-RAG vs Hyper-RAG",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "sentence_provenance_coverage": {"CascHyper-RAG": _provenance_coverage(casc_traces), "Hyper-RAG": _provenance_coverage(hyper_traces)},
        "topic_candidate_diagnostics": _topic_candidate_diagnostics(casc_traces, casc_topic_manifest),
        "per_question": {"CascHyper-RAG": casc_rows, "Hyper-RAG": hyper_rows},
        "summary": summary,
    }


def evaluate_final_context_pair(
    gold_items: list[dict[str, Any]], casc_traces: list[dict[str, Any]], hyper_traces: list[dict[str, Any]],
    sentence_texts: dict[str, str], bootstrap_samples: int, seed: int,
    protocol: str = "rq6_generator_visible_context_v1",
) -> dict[str, Any]:
    """Paired end-to-end evidence sufficiency evaluation for generator contexts."""
    casc_index, hyper_index = _trace_index(casc_traces), _trace_index(hyper_traces)
    casc_rows = [evaluate_final_context_question(item, casc_index[item["question_id"]], sentence_texts).as_dict() for item in gold_items]
    hyper_rows = [evaluate_final_context_question(item, hyper_index[item["question_id"]], sentence_texts).as_dict() for item in gold_items]
    casc_groups, hyper_groups = _group_rows(casc_rows), _group_rows(hyper_rows)
    summary = {
        group_name: {
            "CascHyper-RAG": {metric: summarize(casc_groups[group_name], metric) for metric in FINAL_CONTEXT_METRICS},
            "Hyper-RAG": {metric: summarize(hyper_groups[group_name], metric) for metric in FINAL_CONTEXT_METRICS},
            "paired_difference_CascHyper_minus_Hyper": {
                metric: paired_bootstrap(casc_groups[group_name], hyper_groups[group_name], metric, bootstrap_samples, seed)
                for metric in FINAL_CONTEXT_METRICS
            },
        }
        for group_name in casc_groups
    }
    return {
        "schema_version": 1,
        "protocol": protocol,
        "comparison": "CascHyper-RAG vs Hyper-RAG",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "per_question": {"CascHyper-RAG": casc_rows, "Hyper-RAG": hyper_rows},
        "summary": summary,
    }


def _native_selection_diagnostics(traces: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize selected-unit counts without treating them as ranking cutoffs."""
    return {
        "mean_selected_chunks": fmean(len(trace.get("retrieved_chunks", [])) for trace in traces),
        "mean_selected_evidence_units": fmean(len(trace.get("retrieved_evidence_units", [])) for trace in traces),
        "mean_selected_bridges": fmean(len(trace.get("retrieved_bridges", [])) for trace in traces),
    }


def evaluate_native_evidence_pair(
    gold_items: list[dict[str, Any]], casc_traces: list[dict[str, Any]], hyper_traces: list[dict[str, Any]],
    sentence_texts: dict[str, str], bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    """Paired RQ6 evaluation over all evidence selected by the RQ1/RQ2 configurations."""
    casc_index, hyper_index = _trace_index(casc_traces), _trace_index(hyper_traces)
    casc_rows = [evaluate_native_evidence_question(item, casc_index[item["question_id"]], sentence_texts).as_dict() for item in gold_items]
    hyper_rows = [evaluate_native_evidence_question(item, hyper_index[item["question_id"]], sentence_texts).as_dict() for item in gold_items]
    casc_groups, hyper_groups = _group_rows(casc_rows), _group_rows(hyper_rows)
    summary = {
        group_name: {
            "CascHyper-RAG": {metric: summarize(casc_groups[group_name], metric) for metric in NATIVE_EVIDENCE_METRICS},
            "Hyper-RAG": {metric: summarize(hyper_groups[group_name], metric) for metric in NATIVE_EVIDENCE_METRICS},
            "paired_difference_CascHyper_minus_Hyper": {
                metric: paired_bootstrap(casc_groups[group_name], hyper_groups[group_name], metric, bootstrap_samples, seed)
                for metric in NATIVE_EVIDENCE_METRICS
            },
        }
        for group_name in casc_groups
    }
    return {
        "schema_version": 1,
        "protocol": NATIVE_EVIDENCE_PROTOCOL,
        "comparison": "CascHyper-RAG vs Hyper-RAG",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "selection_diagnostics": {
            "CascHyper-RAG": _native_selection_diagnostics(casc_traces),
            "Hyper-RAG": _native_selection_diagnostics(hyper_traces),
        },
        "per_question": {"CascHyper-RAG": casc_rows, "Hyper-RAG": hyper_rows},
        "summary": summary,
    }
