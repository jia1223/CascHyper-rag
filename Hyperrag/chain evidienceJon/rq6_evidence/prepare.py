"""Create frozen canonical Physics manifests without touching RAG caches."""

from __future__ import annotations

import hashlib
import random
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file, write_json


# This deliberately conservative splitter keeps the original character span.
# Annotators can merge adjacent units in gold records when one semantic unit
# consists of multiple grammatical sentences.
_SENTENCE_PATTERN = re.compile(r".+?(?:(?<=[.!?])(?:\s+|$)|$)", re.DOTALL)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return non-empty source sentences as (start, end, text) tuples."""
    pieces: list[tuple[int, int, str]] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        end = start + len(stripped)
        pieces.append((start, end, stripped))
    return pieces


def _heading_positions(text: str) -> tuple[list[int], list[str]]:
    """Return heading positions once, avoiding repeated full-document scans."""
    headings = list(re.finditer(r"(?m)^\s*(\d+(?:\.\d+)*)\s+[^\n]{3,120}", text))
    return [match.start() for match in headings], [match.group(1) for match in headings]


def build_corpus_manifest(contexts: list[str]) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    for document_index, text in enumerate(contexts, start=1):
        document_id = f"physics_doc_{document_index:03d}"
        document_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        documents.append(
            {
                "document_id": document_id,
                "source_sha256": document_hash,
                "char_length": len(text),
            }
        )
        heading_positions, heading_ids = _heading_positions(text)
        for sentence_index, (start, end, sentence_text) in enumerate(split_sentences(text), start=1):
            heading_index = bisect_right(heading_positions, start) - 1
            section = heading_ids[heading_index] if heading_index >= 0 else "document_root"
            sentences.append(
                {
                    "document_id": document_id,
                    "sentence_id": f"{document_id}_s_{sentence_index:05d}",
                    "char_start": start,
                    "char_end": end,
                    "text": sentence_text,
                    "canonical_topic_id": f"{document_id}:{section}",
                }
            )
    return {"schema_version": 1, "documents": documents, "sentences": sentences}


def build_question_split(
    questions_root: str | Path, stage1_count: int, seed: int
) -> dict[str, Any]:
    root = Path(questions_root)
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for stage in (1, 2, 3):
        questions = read_json(root / f"{stage}_stage.json")
        selected_indices = list(range(len(questions)))
        if stage == 1:
            if stage1_count > len(questions):
                raise ValueError(f"Stage 1 has only {len(questions)} questions.")
            selected_indices = sorted(rng.sample(selected_indices, stage1_count))
        for index in selected_indices:
            items.append(
                {
                    "question_id": f"physics_s{stage}_{index + 1:03d}",
                    "stage": stage,
                    "source_index": index,
                    "question": questions[index],
                }
            )
    return {
        "schema_version": 1,
        "seed": seed,
        "stage1_count": stage1_count,
        "items": items,
    }


def prepare_inputs(
    contexts_path: str | Path,
    questions_root: str | Path,
    output_directory: str | Path,
    stage1_count: int,
    seed: int,
) -> dict[str, Any]:
    output = Path(output_directory)
    contexts = read_json(contexts_path)
    if not isinstance(contexts, list) or not all(isinstance(item, str) for item in contexts):
        raise ValueError("The Physics context file must be a JSON list of strings.")
    corpus = build_corpus_manifest(contexts)
    split = build_question_split(questions_root, stage1_count, seed)
    source_files = [Path(contexts_path)] + [Path(questions_root) / f"{stage}_stage.json" for stage in (1, 2, 3)]
    run_manifest = {
        "schema_version": 1,
        "seed": seed,
        "stage1_count": stage1_count,
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "counts": {
            "documents": len(corpus["documents"]),
            "sentences": len(corpus["sentences"]),
            "questions": len(split["items"]),
        },
    }
    write_json(output / "corpus_manifest.json", corpus)
    write_json(output / "question_split.json", split)
    write_json(output / "run_manifest.json", run_manifest)
    return run_manifest
