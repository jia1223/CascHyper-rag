"""Build a blinded-preserving adjudication queue from two RQ6 annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


_DECISIONS = ["use_A", "use_B", "revised", "ineligible"]


def _hops(item: dict[str, Any]) -> tuple[tuple[Any, Any], ...]:
    return tuple((hop.get("hop"), hop.get("sentence_id")) for hop in item.get("gold_hops", []))


def _bridges(item: dict[str, Any]) -> tuple[tuple[str, Any, Any], ...]:
    return tuple(
        (
            str(bridge.get("canonical_entity") or bridge.get("entity_id") or "").casefold().strip(),
            bridge.get("from_hop"),
            bridge.get("to_hop"),
        )
        for bridge in item.get("bridges", [])
    )


def _differences(annotator_a: dict[str, Any], annotator_b: dict[str, Any]) -> list[str]:
    differences = []
    if annotator_a.get("eligible_for_full_chain") != annotator_b.get("eligible_for_full_chain"):
        differences.append("eligibility")
    if _hops(annotator_a) != _hops(annotator_b):
        differences.append("gold_hops")
    if _bridges(annotator_a) != _bridges(annotator_b):
        differences.append("bridges")
    if set(annotator_a.get("gold_topics", [])) != set(annotator_b.get("gold_topics", [])):
        differences.append("gold_topics")
    return differences


def _referenced_sentences(
    annotator_a: dict[str, Any], annotator_b: dict[str, Any], by_sentence_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    sentence_ids = []
    for item in (annotator_a, annotator_b):
        sentence_ids.extend(str(hop.get("sentence_id")) for hop in item.get("gold_hops", []))
    unique_ids = list(dict.fromkeys(sentence_ids))
    return [
        by_sentence_id.get(sentence_id, {"sentence_id": sentence_id, "text": "[MISSING FROM CANONICAL MANIFEST]"})
        for sentence_id in unique_ids
    ]


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _markdown_package(entry: dict[str, Any]) -> str:
    source_rows = []
    for sentence in entry["referenced_sentences"]:
        text = str(sentence.get("text", "")).replace("|", "\\|").replace("\n", " ")
        source_rows.append(
            f"| `{sentence['sentence_id']}` | `{sentence.get('document_id', '')}` | "
            f"{sentence.get('canonical_topic_id', '')} | {text} |"
        )
    return f"""# RQ6 Gold-Chain Adjudication: {entry['question_id']}

**Stage:** {entry['stage']}<br>
**Difference fields:** {', '.join(entry['differences'])}

## Question

{entry['question']}

## Canonical source sentences selected by either annotator

| sentence_id | document_id | canonical topic | source sentence |
|---|---|---|---|
{chr(10).join(source_rows)}

## Annotator A (frozen original)

```json
{_json_block(entry['annotator_A'])}
```

## Annotator B (frozen original)

```json
{_json_block(entry['annotator_B'])}
```

## Adjudication instructions

Edit only the paired decision file at `../decisions/{entry['question_id']}.json`.
Choose `use_A` or `use_B` only if that record is fully defensible. Choose
`revised` and fill `final_gold` when neither record is correct as written.
Choose `ineligible` only when no complete ordered chain can be supported; its
final gold must have empty `gold_hops` and `bridges`. Record the reason and
consult the canonical source sentences, not RAG outputs.
"""


def _decision_draft(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_status": "draft",
        "question_id": entry["question_id"],
        "stage": entry["stage"],
        "difference_fields": entry["differences"],
        "allowed_decisions": _DECISIONS,
        "decision": "",
        "final_gold": None,
        "adjudication_rationale": "",
    }


def build_adjudication_queue(
    corpus_manifest: dict[str, Any],
    question_split: dict[str, Any],
    annotator_a: dict[str, dict[str, Any]],
    annotator_b: dict[str, dict[str, Any]],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write packages and decision drafts for exactly the A/B-disputed questions."""
    questions = {item["question_id"]: item for item in question_split["items"]}
    expected_ids = set(questions)
    for label, annotations in (("A", annotator_a), ("B", annotator_b)):
        if set(annotations) != expected_ids:
            missing = sorted(expected_ids - set(annotations))
            unexpected = sorted(set(annotations) - expected_ids)
            raise ValueError(f"Annotator {label} IDs must match frozen split; missing={missing[:3]}, unexpected={unexpected[:3]}")
    by_sentence_id = {sentence["sentence_id"]: sentence for sentence in corpus_manifest["sentences"]}
    entries = []
    for question in question_split["items"]:
        question_id = question["question_id"]
        a_item, b_item = annotator_a[question_id], annotator_b[question_id]
        differences = _differences(a_item, b_item)
        if not differences:
            continue
        entries.append(
            {
                "question_id": question_id,
                "stage": question["stage"],
                "question": question["question"],
                "differences": differences,
                "annotator_A": a_item,
                "annotator_B": b_item,
                "referenced_sentences": _referenced_sentences(a_item, b_item, by_sentence_id),
            }
        )
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Adjudication output directory must be empty to avoid retaining stale decisions: {output}"
        )
    packages = output / "packages"
    decisions = output / "decisions"
    packages.mkdir(parents=True, exist_ok=True)
    decisions.mkdir(parents=True, exist_ok=True)
    with (output / "adjudication_queue.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            (packages / f"{entry['question_id']}.md").write_text(_markdown_package(entry), encoding="utf-8")
            write_json(decisions / f"{entry['question_id']}.json", _decision_draft(entry))
    manifest = {
        "schema_version": 1,
        "disputed_question_count": len(entries),
        "decision_choices": _DECISIONS,
        "items": [
            {
                "question_id": entry["question_id"],
                "stage": entry["stage"],
                "differences": entry["differences"],
                "package": str(Path("packages") / f"{entry['question_id']}.md"),
                "decision_draft": str(Path("decisions") / f"{entry['question_id']}.json"),
            }
            for entry in entries
        ],
    }
    write_json(output / "adjudication_manifest.json", manifest)
    return manifest


def _load_completed_drafts(directory: str | Path) -> dict[str, dict[str, Any]]:
    drafts = {}
    for path in sorted(Path(directory).glob("*.json")):
        item = read_json(path)
        if item.get("annotation_status") != "complete":
            raise ValueError(f"Annotation draft is not complete: {path}")
        question_id = item.get("question_id")
        if not question_id or question_id in drafts:
            raise ValueError(f"Draft has missing or duplicate question_id: {path}")
        drafts[question_id] = item
    return drafts


def generate_adjudication_queue(
    manifest_path: str | Path,
    split_path: str | Path,
    annotator_a_directory: str | Path,
    annotator_b_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Load completed A/B drafts and generate an isolated adjudication queue."""
    return build_adjudication_queue(
        read_json(manifest_path),
        read_json(split_path),
        _load_completed_drafts(annotator_a_directory),
        _load_completed_drafts(annotator_b_directory),
        output_directory,
    )
