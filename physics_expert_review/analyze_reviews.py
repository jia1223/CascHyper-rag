#!/usr/bin/env python3
"""Unblind completed expert forms and calculate the planned Physics statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("correctness", "completeness", "clarity")
METHOD_DISPLAY = {"casc": "CascHyper-RAG", "hyper": "Hyper-RAG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forms-dir", type=Path, required=True, help="Directory containing completed expert_*.csv forms.")
    parser.add_argument("--unblinding-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("is_calibration", "false").lower() == "true":
            continue
        identifier = (row["question_id"], row["expert_id"])
        if identifier in key:
            raise ValueError(f"Duplicate unblinding key entry: {identifier}")
        if {row["system_a_method"], row["system_b_method"]} != {"casc", "hyper"}:
            raise ValueError(f"Invalid method mapping for {identifier}")
        key[identifier] = row
    if not key:
        raise ValueError("No formal-review entries found in unblinding key.")
    return key


def score(value: str | None, field: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a score from 1 to 5.") from error
    if parsed < 1 or parsed > 5 or not parsed.is_integer():
        raise ValueError(f"{field} must be an integer score from 1 to 5.")
    return parsed


def read_completed_forms(forms_dir: Path, key: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], dict[str, int]]:
    question_scores: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rater_scores: dict[str, Any] = {metric: defaultdict(dict) for metric in METRICS}
    preference_votes = {"casc_win": 0, "tie": 0, "hyper_win": 0}
    form_paths = sorted(forms_dir.glob("expert_*.csv"))
    if not form_paths:
        raise ValueError(f"No completed expert_*.csv forms found in {forms_dir}")

    seen: set[tuple[str, str]] = set()
    for form_path in form_paths:
        with form_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            identifier = (row.get("question_id", ""), row.get("expert_id", ""))
            if identifier not in key:
                raise ValueError(f"Form row is absent from the formal unblinding key: {identifier}")
            if identifier in seen:
                raise ValueError(f"Duplicate completed form row: {identifier}")
            seen.add(identifier)
            mapping = key[identifier]
            for system in ("a", "b"):
                method = mapping[f"system_{system}_method"]
                for metric in METRICS:
                    value = score(row.get(f"system_{system}_{metric}"), f"system_{system}_{metric}")
                    question_scores[identifier[0]][metric][method].append(value)
                    rater_scores[metric][f"{identifier[0]}:{method}"][identifier[1]] = value
            preference = str(row.get("overall_preference", "")).strip().upper()
            if preference not in {"A", "B", "TIE"}:
                raise ValueError(f"overall_preference for {identifier} must be A, B, or Tie.")
            if preference == "TIE":
                preference_votes["tie"] += 1
            else:
                winning_method = mapping[f"system_{preference.lower()}_method"]
                preference_votes[f"{winning_method}_win"] += 1

    missing = sorted(set(key) - seen)
    if missing:
        preview = ", ".join(f"{question}/{expert}" for question, expert in missing[:10])
        raise ValueError(f"Missing completed form rows for {len(missing)} key entries: {preview}")
    return question_scores, preference_votes, rater_scores


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def bootstrap_ci(differences: list[float], samples: int, seed: int) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    rng = random.Random(seed)
    size = len(differences)
    bootstrap_means = [sum(rng.choice(differences) for _ in range(size)) / size for _ in range(samples)]
    return percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)


def icc_a1(subject_scores: dict[str, dict[str, float]]) -> float | None:
    """Two-way random-effects, absolute-agreement, single-rater ICC(A,1)."""
    subjects = sorted(subject_scores)
    if len(subjects) < 2:
        return None
    raters = sorted(next(iter(subject_scores.values())))
    if len(raters) < 2 or any(set(subject_scores[subject]) != set(raters) for subject in subjects):
        return None
    values = [[subject_scores[subject][rater] for rater in raters] for subject in subjects]
    n, k = len(values), len(raters)
    grand_mean = sum(sum(row) for row in values) / (n * k)
    row_means = [sum(row) / k for row in values]
    column_means = [sum(row[column] for row in values) / n for column in range(k)]
    ms_rows = k * sum((mean - grand_mean) ** 2 for mean in row_means) / (n - 1)
    ms_columns = n * sum((mean - grand_mean) ** 2 for mean in column_means) / (k - 1)
    residual = sum(
        (values[row][column] - row_means[row] - column_means[column] + grand_mean) ** 2
        for row in range(n)
        for column in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return None if denominator == 0 else (ms_rows - ms_error) / denominator


def calculate_summary(question_scores: dict[str, Any], preference_votes: dict[str, int], rater_scores: dict[str, Any], samples: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_question: list[dict[str, Any]] = []
    metric_summary: dict[str, Any] = {}
    for question_id in sorted(question_scores):
        row: dict[str, Any] = {"question_id": question_id}
        for metric in METRICS:
            casc_scores = question_scores[question_id][metric]["casc"]
            hyper_scores = question_scores[question_id][metric]["hyper"]
            if not casc_scores or not hyper_scores:
                raise ValueError(f"Incomplete paired scores for {question_id} / {metric}")
            casc_mean = sum(casc_scores) / len(casc_scores)
            hyper_mean = sum(hyper_scores) / len(hyper_scores)
            row[f"casc_{metric}"] = casc_mean
            row[f"hyper_{metric}"] = hyper_mean
            row[f"difference_{metric}"] = casc_mean - hyper_mean
        per_question.append(row)

    for index, metric in enumerate(METRICS):
        casc_means = [row[f"casc_{metric}"] for row in per_question]
        hyper_means = [row[f"hyper_{metric}"] for row in per_question]
        differences = [row[f"difference_{metric}"] for row in per_question]
        ci_low, ci_high = bootstrap_ci(differences, samples, seed + index)
        metric_summary[metric] = {
            "casc_mean": sum(casc_means) / len(casc_means),
            "hyper_mean": sum(hyper_means) / len(hyper_means),
            "difference_casc_minus_hyper": sum(differences) / len(differences),
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "icc_a1": icc_a1(rater_scores[metric]),
        }
    return {"question_count": len(per_question), "metrics": metric_summary, "preference_votes": preference_votes}, per_question


def write_outputs(output_dir: Path, summary: dict[str, Any], per_question: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["question_id"] + [field for metric in METRICS for field in (f"casc_{metric}", f"hyper_{metric}", f"difference_{metric}")]
    with (output_dir / "per_question_scores.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_question)

    lines = [
        "# Physics 专家审核结果", "", f"- 正式问题数：{summary['question_count']}", "- 比较：CascHyper-RAG vs. Hyper-RAG", "",
        "| Metric | Hyper-RAG | CascHyper-RAG | Δ (Casc - Hyper) | 95% paired-bootstrap CI | ICC(A,1) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for metric in METRICS:
        values = summary["metrics"][metric]
        ci_low, ci_high = values["paired_bootstrap_95_ci"]
        icc = "N/A" if values["icc_a1"] is None else f"{values['icc_a1']:.3f}"
        lines.append(
            f"| {metric.title()} | {values['hyper_mean']:.3f} | {values['casc_mean']:.3f} | "
            f"{values['difference_casc_minus_hyper']:+.3f} | [{ci_low:+.3f}, {ci_high:+.3f}] | {icc} |"
        )
    votes = summary["preference_votes"]
    lines.extend(["", "## Overall preference", "", f"CascHyper-RAG win / Tie / Hyper-RAG win = {votes['casc_win']} / {votes['tie']} / {votes['hyper_win']}", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        if args.output_dir.exists():
            if not args.overwrite:
                raise ValueError(f"Output directory already exists: {args.output_dir}. Use --overwrite to replace it.")
            shutil.rmtree(args.output_dir)
        key = load_key(args.unblinding_key)
        question_scores, preference_votes, rater_scores = read_completed_forms(args.forms_dir, key)
        summary, per_question = calculate_summary(question_scores, preference_votes, rater_scores, args.bootstrap_samples, args.seed)
        write_outputs(args.output_dir, summary, per_question)
        print(f"Wrote expert-review analysis to {args.output_dir}")
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
