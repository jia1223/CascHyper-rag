#!/usr/bin/env python3
"""Prepare single-blind expert-review forms for Physics answer-quality evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any


METHOD_LABELS = {"casc": "CascHyper-RAG", "hyper": "Hyper-RAG"}
FORM_FIELDS = [
    "question_id",
    "expert_id",
    "display_order",
    "question",
    "reference_answer",
    "first_system_label",
    "first_system_answer",
    "second_system_label",
    "second_system_answer",
    "system_a_correctness",
    "system_a_completeness",
    "system_a_clarity",
    "system_b_correctness",
    "system_b_completeness",
    "system_b_clarity",
    "overall_preference",
    "optional_comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casc-results", type=Path, nargs="+", required=True)
    parser.add_argument("--hyper-results", type=Path, nargs="+", required=True)
    parser.add_argument("--references", type=Path, help="JSON reference-answer mapping or list.")
    parser.add_argument("--run-metadata", type=Path, help="Frozen method/configuration metadata JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--calibration-size", type=int, default=5)
    parser.add_argument("--expert-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stage_from_path(path: Path) -> str:
    match = re.search(r"_(\d+)_stage_result\.json$", path.name)
    return f"stage_{match.group(1)}" if match else "unlabeled"


def load_results(paths: list[Path]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON list.")
        for row in rows:
            query = str(row.get("query", "")).strip()
            answer = str(row.get("result", "")).strip()
            if not query or not answer:
                raise ValueError(f"{path} contains an item without query or result.")
            if query in records:
                raise ValueError(f"Duplicate query in result inputs: {query!r}")
            records[query] = {"answer": answer, "stage": stage_from_path(path)}
    return records


def load_references(path: Path) -> dict[str, str]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(query): str(answer).strip() for query, answer in raw.items() if str(answer).strip()}
    if isinstance(raw, list):
        references: dict[str, str] = {}
        for row in raw:
            query = str(row.get("query", "")).strip()
            answer = str(row.get("reference_answer", row.get("answer", ""))).strip()
            if query and answer:
                references[query] = answer
        return references
    raise ValueError("Reference JSON must be a query-to-answer object or a list of objects.")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reference_template(path: Path, rows: list[dict[str, str]]) -> None:
    payload = [
        {
            "question_id": row["question_id"],
            "query": row["question"],
            "reference_answer": "",
        }
        for row in rows
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_metadata_template(path: Path) -> None:
    template = {
        "casc_hyper_rag": {
            "code_version": "",
            "generation_model": "",
            "generation_prompt_version": "",
            "decoding_parameters": "",
            "context_token_budget": "",
            "embedding_model": "",
        },
        "hyper_rag": {
            "code_version": "",
            "generation_model": "",
            "generation_prompt_version": "",
            "decoding_parameters": "",
            "context_token_budget": "",
            "embedding_model": "",
        },
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def load_run_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {"casc_hyper_rag", "hyper_rag"}
    if not isinstance(metadata, dict) or set(metadata) != required_keys:
        raise ValueError("Run metadata must contain casc_hyper_rag and hyper_rag objects.")
    required_fields = {"code_version", "generation_model", "generation_prompt_version", "decoding_parameters", "context_token_budget", "embedding_model"}
    for method, details in metadata.items():
        if not isinstance(details, dict) or set(details) != required_fields:
            raise ValueError(f"Run metadata for {method} must contain all required fields.")
        if any(value in (None, "") for value in details.values()):
            raise ValueError(f"Run metadata for {method} contains blank fields.")
    return metadata


def file_provenance(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]


def prepare_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    casc = load_results(args.casc_results)
    hyper = load_results(args.hyper_results)
    if set(casc) != set(hyper):
        casc_only = sorted(set(casc) - set(hyper))
        hyper_only = sorted(set(hyper) - set(casc))
        raise ValueError(f"Result files must contain identical queries (casc-only={len(casc_only)}, hyper-only={len(hyper_only)}).")

    if args.sample_size <= 0 or args.calibration_size < 0 or args.expert_count <= 0:
        raise ValueError("sample size and expert count must be positive; calibration size cannot be negative.")
    all_queries = sorted(casc)
    required_count = args.sample_size + args.calibration_size
    if required_count > len(all_queries):
        raise ValueError(f"Requested {required_count} questions, but only {len(all_queries)} paired questions are available.")

    chosen = random.Random(args.seed).sample(all_queries, required_count)
    formal, calibration = chosen[: args.sample_size], chosen[args.sample_size :]

    def make_rows(queries: list[str], prefix: str) -> list[dict[str, str]]:
        return [
            {
                "question_id": f"{prefix}{index:03d}",
                "question": query,
                "casc_answer": casc[query]["answer"],
                "hyper_answer": hyper[query]["answer"],
                "stage": casc[query]["stage"],
            }
            for index, query in enumerate(queries, start=1)
        ]

    return make_rows(formal, "P"), make_rows(calibration, "C")


def create_forms(
    output_dir: Path,
    formal_rows: list[dict[str, str]],
    calibration_rows: list[dict[str, str]],
    references: dict[str, str],
    expert_count: int,
    seed: int,
    audit_metadata: dict[str, Any],
) -> None:
    materials_dir = output_dir / "materials"
    private_dir = output_dir / "private"
    materials_dir.mkdir(parents=True)
    private_dir.mkdir()
    mapping_rows: list[dict[str, str]] = []

    for expert_index in range(1, expert_count + 1):
        expert_id = f"expert_{expert_index:02d}"
        form_rows: list[dict[str, str]] = []
        calibration_form_rows: list[dict[str, str]] = []
        for is_calibration, source_rows, target_rows in (
            (False, formal_rows, form_rows),
            (True, calibration_rows, calibration_form_rows),
        ):
            for row in source_rows:
                randomizer = random.Random(f"{seed}:{expert_id}:{row['question_id']}")
                a_is_casc = randomizer.choice((True, False))
                a_method = "casc" if a_is_casc else "hyper"
                b_method = "hyper" if a_is_casc else "casc"
                display_order = randomizer.choice(("A/B", "B/A"))
                answers = {"A": row[f"{a_method}_answer"], "B": row[f"{b_method}_answer"]}
                first_label, second_label = display_order.split("/")
                target_rows.append(
                    {
                        "question_id": row["question_id"],
                        "expert_id": expert_id,
                        "display_order": display_order,
                        "question": row["question"],
                        "reference_answer": references[row["question"]],
                        "first_system_label": f"System {first_label}",
                        "first_system_answer": answers[first_label],
                        "second_system_label": f"System {second_label}",
                        "second_system_answer": answers[second_label],
                        "system_a_correctness": "",
                        "system_a_completeness": "",
                        "system_a_clarity": "",
                        "system_b_correctness": "",
                        "system_b_completeness": "",
                        "system_b_clarity": "",
                        "overall_preference": "",
                        "optional_comment": "",
                    }
                )
                mapping_rows.append(
                    {
                        "question_id": row["question_id"],
                        "expert_id": expert_id,
                        "is_calibration": str(is_calibration).lower(),
                        "system_a_method": a_method,
                        "system_b_method": b_method,
                        "stage": row["stage"],
                    }
                )
        write_csv(materials_dir / f"{expert_id}.csv", FORM_FIELDS, form_rows)
        write_csv(materials_dir / f"calibration_{expert_id}.csv", FORM_FIELDS, calibration_form_rows)

    write_csv(
        private_dir / "unblinding_key.csv",
        ["question_id", "expert_id", "is_calibration", "system_a_method", "system_b_method", "stage"],
        mapping_rows,
    )
    (private_dir / "sample_manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "created_on": date.today().isoformat(),
                "formal_questions": formal_rows,
                "calibration_questions": calibration_rows,
                **audit_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (materials_dir / "README.md").write_text(
        "# 专家评分说明\n\n"
        "请独立评价每份匿名答案。Correctness、Completeness 和 Clarity 均填 1--5 分；"
        "Overall preference 填 `A`、`B` 或 `Tie`。请勿修改题目、参考答案或 System A/B 的答案。"
        "校准题仅用于统一评分标准，不会进入最终统计。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        formal_rows, calibration_rows = prepare_rows(args)
        selected_rows = formal_rows + calibration_rows
        if args.output_dir.exists():
            if not args.overwrite:
                raise ValueError(f"Output directory already exists: {args.output_dir}. Use --overwrite to replace it.")
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True)

        if args.references is None:
            write_reference_template(args.output_dir / "reference_answers_template.json", selected_rows)
            write_run_metadata_template(args.output_dir / "run_metadata_template.json")
            print(
                "Created the fixed sample plus reference-answer and run-metadata templates. Fill both, then rerun with --references and --run-metadata.",
                file=sys.stderr,
            )
            return 2

        if args.run_metadata is None:
            write_run_metadata_template(args.output_dir / "run_metadata_template.json")
            print("Created the run-metadata template. Fill it, then rerun with --run-metadata.", file=sys.stderr)
            return 2

        references = load_references(args.references)
        missing = [row["question_id"] for row in selected_rows if not references.get(row["question"])]
        if missing:
            raise ValueError(f"References are missing for {len(missing)} selected questions: {', '.join(missing[:10])}")
        run_metadata = load_run_metadata(args.run_metadata)
        audit_metadata = {
            "sample_size": args.sample_size,
            "calibration_size": args.calibration_size,
            "expert_count": args.expert_count,
            "casc_result_inputs": file_provenance(args.casc_results),
            "hyper_result_inputs": file_provenance(args.hyper_results),
            "frozen_run_metadata": run_metadata,
        }
        create_forms(args.output_dir, formal_rows, calibration_rows, references, args.expert_count, args.seed, audit_metadata)
        print(f"Created blinded review materials in {args.output_dir / 'materials'}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
