"""Paired summary statistics with deterministic bootstrap confidence intervals."""

from __future__ import annotations

import random
from statistics import fmean
from typing import Any


METRICS = (
    "topic_coverage",
    "chunk_recall_at_5",
    "sentence_recall_at_10",
    "bridge_recall_at_10",
    "full_chain_at_5",
    "full_chain_at_10",
    "full_chain_at_20",
)


def _values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row.get(metric) is not None]


def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = _values(rows, metric)
    return {"n": len(values), "mean": fmean(values) if values else None}


def paired_bootstrap(
    casc_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    casc_by_id = {row["question_id"]: row for row in casc_rows}
    hyper_by_id = {row["question_id"]: row for row in hyper_rows}
    differences = [
        float(casc_by_id[qid][metric]) - float(hyper_by_id[qid][metric])
        for qid in sorted(casc_by_id.keys() & hyper_by_id.keys())
        if casc_by_id[qid].get(metric) is not None and hyper_by_id[qid].get(metric) is not None
    ]
    if not differences:
        return {"n": 0, "mean_difference": None, "ci_95": None}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(fmean(rng.choice(differences) for _ in differences))
    estimates.sort()
    lower = estimates[max(0, int(0.025 * samples) - 1)]
    upper = estimates[min(samples - 1, int(0.975 * samples) - 1)]
    return {
        "n": len(differences),
        "mean_difference": fmean(differences),
        "ci_95": [lower, upper],
    }
