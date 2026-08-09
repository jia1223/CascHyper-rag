"""Optional publication-oriented figures for RQ6 result JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_plots(results: dict[str, Any], output_directory: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install matplotlib to generate plots.") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    overall = results["summary"]["overall"]
    casc = overall["CascHyper-RAG"]
    hyper = overall["Hyper-RAG"]
    funnel_labels = ["Topic\nrouting", "Chunk\nR@5", "Sentence\nR@10", "Bridge\nR@10", "Full-chain\nR@10"]
    funnel_metrics = ["topic_coverage", "chunk_recall_at_5", "sentence_recall_at_10", "bridge_recall_at_10", "full_chain_at_10"]
    funnel_values = [casc[name]["mean"] or 0.0 for name in funnel_metrics]
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    positions = list(range(len(funnel_labels)))
    axis.plot(positions, funnel_values, marker="o", linewidth=2.5, color="#1f77b4")
    axis.set_xticks(positions, funnel_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Coverage / completion rate")
    axis.set_title("CascHyper-RAG cascaded evidence funnel")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    funnel_path = output / "rq6_caschyper_funnel.png"
    figure.savefig(funnel_path, dpi=300)
    plt.close(figure)

    labels = ["Chunk\nR@5", "Sentence\nR@10", "Bridge\nR@10", "Full-chain\nR@10"]
    metric_names = ["chunk_recall_at_5", "sentence_recall_at_10", "bridge_recall_at_10", "full_chain_at_10"]
    casc_values = [casc[name]["mean"] or 0.0 for name in metric_names]
    hyper_values = [hyper[name]["mean"] or 0.0 for name in metric_names]

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    positions = list(range(len(labels)))
    axis.plot(positions, casc_values, marker="o", linewidth=2.5, label="CascHyper-RAG")
    axis.plot(positions, hyper_values, marker="o", linewidth=2.5, label="Hyper-RAG")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall / completion rate")
    axis.set_title("RQ6 evidence-retrieval funnel")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    comparison_path = output / "rq6_evidence_comparison.png"
    figure.savefig(comparison_path, dpi=300)
    plt.close(figure)

    stages = ["stage_1", "stage_2", "stage_3"]
    stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
    casc_stage = [results["summary"][stage]["CascHyper-RAG"]["full_chain_at_10"]["mean"] or 0.0 for stage in stages]
    hyper_stage = [results["summary"][stage]["Hyper-RAG"]["full_chain_at_10"]["mean"] or 0.0 for stage in stages]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    width = 0.36
    locations = list(range(len(stages)))
    axis.bar([location - width / 2 for location in locations], casc_stage, width, label="CascHyper-RAG")
    axis.bar([location + width / 2 for location in locations], hyper_stage, width, label="Hyper-RAG")
    axis.set_xticks(locations, stage_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Full-chain Recall@10")
    axis.set_title("RQ6 full-chain retrieval by question stage")
    axis.legend(frameon=False)
    figure.tight_layout()
    stage_path = output / "rq6_full_chain_by_stage.png"
    figure.savefig(stage_path, dpi=300)
    plt.close(figure)
    return [funnel_path, comparison_path, stage_path]
