"""Merge completed RQ6 adjudication decisions into one frozen gold file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .adjudication import _differences
from .io_utils import read_json, write_json
from .validation import validate_gold, validate_question_split


def _require_exact_ids(items: dict[str, dict[str, Any]], expected_ids: set[str], label: str) -> None:
    observed_ids = set(items)
    if observed_ids == expected_ids:
        return
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    raise ValueError(f"{label} IDs must match expected set; missing={missing[:3]}, unexpected={unexpected[:3]}")


def _decision_final_gold(decision: dict[str, Any], question_id: str, stage: int) -> dict[str, Any]:
    final_gold = decision.get("final_gold")
    if not isinstance(final_gold, dict):
        raise ValueError(f"{decision.get('decision')} decision requires final_gold: {question_id}")
    if final_gold.get("question_id") != question_id or final_gold.get("stage") != stage:
        raise ValueError(f"final_gold does not match frozen question/stage: {question_id}")
    return deepcopy(final_gold)


def build_frozen_gold(
    corpus_manifest: dict[str, Any],
    question_split: dict[str, Any],
    annotator_a: dict[str, dict[str, Any]],
    annotator_b: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Resolve A/B records in split order and validate the resulting gold chains."""
    split_items = question_split["items"]
    expected_ids = {item["question_id"] for item in split_items}
    _require_exact_ids(annotator_a, expected_ids, "Annotator A")
    _require_exact_ids(annotator_b, expected_ids, "Annotator B")
    disputed_ids = {
        item["question_id"]
        for item in split_items
        if _differences(annotator_a[item["question_id"]], annotator_b[item["question_id"]])
    }
    _require_exact_ids(decisions, disputed_ids, "Completed adjudication decision")
    gold_items: list[dict[str, Any]] = []
    summary: dict[str, int] = {"consistent_from_A": 0, "adjudicated_use_A": 0, "adjudicated_use_B": 0, "adjudicated_revised": 0, "adjudicated_ineligible": 0}
    for split_item in split_items:
        question_id = split_item["question_id"]
        a_item, b_item = annotator_a[question_id], annotator_b[question_id]
        if question_id not in disputed_ids:
            gold = deepcopy(a_item)
            summary["consistent_from_A"] += 1
        else:
            decision = decisions[question_id]
            if decision.get("decision_status") != "complete":
                raise ValueError(f"Adjudication decision is not complete: {question_id}")
            if decision.get("question_id") != question_id or decision.get("stage") != split_item["stage"]:
                raise ValueError(f"Adjudication decision does not match frozen question/stage: {question_id}")
            choice = decision.get("decision")
            if choice == "use_A":
                if decision.get("final_gold") is not None:
                    raise ValueError(f"use_A decision must not include final_gold: {question_id}")
                gold = deepcopy(a_item)
                summary["adjudicated_use_A"] += 1
            elif choice == "use_B":
                if decision.get("final_gold") is not None:
                    raise ValueError(f"use_B decision must not include final_gold: {question_id}")
                gold = deepcopy(b_item)
                summary["adjudicated_use_B"] += 1
            elif choice == "revised":
                gold = _decision_final_gold(decision, question_id, split_item["stage"])
                summary["adjudicated_revised"] += 1
            elif choice == "ineligible":
                gold = _decision_final_gold(decision, question_id, split_item["stage"])
                summary["adjudicated_ineligible"] += 1
            else:
                raise ValueError(f"Unsupported adjudication decision for {question_id}: {choice!r}")
        gold.pop("annotation_status", None)
        gold_items.append(gold)
    sentence_ids = {sentence["sentence_id"] for sentence in corpus_manifest["sentences"]}
    errors = validate_gold(gold_items, sentence_ids) + validate_question_split(gold_items, question_split)
    if errors:
        raise ValueError("Frozen gold validation failed:\n- " + "\n- ".join(errors))
    summary["total_questions"] = len(gold_items)
    return gold_items, summary


def _load_complete_decisions(directory: str | Path) -> dict[str, dict[str, Any]]:
    decisions = {}
    for path in sorted(Path(directory).glob("*.json")):
        decision = read_json(path)
        question_id = decision.get("question_id")
        if not question_id or question_id in decisions:
            raise ValueError(f"Decision has missing or duplicate question_id: {path}")
        decisions[question_id] = decision
    return decisions


def merge_adjudications(
    manifest_path: str | Path,
    split_path: str | Path,
    annotator_a_directory: str | Path,
    annotator_b_directory: str | Path,
    decisions_directory: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Write a never-overwritten, validated frozen gold JSON list."""
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"Frozen gold output already exists and will not be overwritten: {output}")
    from .adjudication import _load_completed_drafts

    gold_items, summary = build_frozen_gold(
        read_json(manifest_path),
        read_json(split_path),
        _load_completed_drafts(annotator_a_directory),
        _load_completed_drafts(annotator_b_directory),
        _load_complete_decisions(decisions_directory),
    )
    write_json(output, gold_items)
    return summary
