"""Create independent human-annotation candidate packs for RQ6 gold chains."""

from __future__ import annotations

import json
import math
import re
from heapq import nlargest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


class LexicalSentenceRetriever:
    """A deterministic BM25-style retriever over canonical source sentences."""

    def __init__(self, sentences: list[dict[str, Any]]) -> None:
        self.sentences = sentences
        self.document_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, sentence in enumerate(sentences):
            counts = Counter(tokenize(sentence["text"]))
            self.document_lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
        self.average_length = sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 1.0

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        matched_terms: dict[int, set[str]] = defaultdict(set)
        total = len(self.sentences)
        for token in query_tokens:
            postings = self.postings.get(token, [])
            if not postings:
                continue
            inverse_frequency = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                length_norm = 1.2 * (1 - 0.75 + 0.75 * self.document_lengths[index] / self.average_length)
                scores[index] += inverse_frequency * (frequency * 2.2) / (frequency + length_norm)
                matched_terms[index].add(token)
        ranked = nlargest(
            top_k,
            scores,
            key=lambda index: (scores[index], self.sentences[index]["sentence_id"]),
        )
        return [
            {
                **self.sentences[index],
                "lexical_score": round(scores[index], 8),
                "matched_terms": sorted(matched_terms[index]),
            }
            for index in ranked
        ]


def _draft_template(question: dict[str, Any]) -> str:
    hop_count = int(question["stage"])
    hops = [
        {
            "hop": hop,
            "sentence_id": "",
            "role": "",
            "required": True,
        }
        for hop in range(1, hop_count + 1)
    ]
    bridges = [
        {
            "canonical_entity": "",
            "aliases": [],
            "from_hop": hop,
            "to_hop": hop + 1,
        }
        for hop in range(1, hop_count)
    ]
    return json.dumps(
        {
            "annotation_status": "draft",
            "question_id": question["question_id"],
            "stage": question["stage"],
            "eligible_for_full_chain": None,
            "gold_hops": hops,
            "bridges": bridges,
            "gold_topics": [],
            "chain_scope": "",
            "annotation_rationale": "",
        },
        ensure_ascii=False,
        indent=2,
    )


def _markdown_package(
    annotator: str,
    question: dict[str, Any],
    stage_reference: str,
    candidates: list[dict[str, Any]],
) -> str:
    rows = []
    for rank, sentence in enumerate(candidates, start=1):
        text = sentence["text"].replace("|", "\\|").replace("\n", " ")
        terms = ", ".join(sentence["matched_terms"])
        rows.append(
            f"| {rank} | `{sentence['sentence_id']}` | `{sentence['document_id']}` | "
            f"{sentence['canonical_topic_id']} | {sentence['char_start']}-{sentence['char_end']} | {terms} | {text} |"
        )
    stage_guidance = {
        1: "Select one indispensable evidence sentence.",
        2: "Select two ordered indispensable evidence sentences and one bridge entity connecting them.",
        3: "Select three ordered indispensable evidence sentences and two bridge entities connecting adjacent hops.",
    }[int(question["stage"])]
    return f"""# RQ6 Evidence-Chain Annotation: {question['question_id']}

**Annotator:** {annotator}<br>
**Stage:** {question['stage']}<br>
**Instruction:** {stage_guidance}

## Question

{question['question']}

## Existing stage_ref candidate evidence

> This text is a candidate aid only. Do not treat it as the gold chain without
> checking the source sentences below.

{stage_reference}

## Independently retrievable source-sentence candidates

Candidates are ranked by a deterministic lexical search over the frozen
canonical Physics sentence manifest using the question and stage_ref. Rank is
not a relevance judgement; inspect the source text and record only
indispensable evidence. If the needed evidence is missing, search the canonical
manifest and record the source sentence ID you identify.

| Rank | sentence_id | document_id | canonical topic | char span | matched terms | source sentence |
|---:|---|---|---|---|---|---|
{chr(10).join(rows)}

## Annotation decision

1. Mark `eligible_for_full_chain` `true` only when the required number of
   ordered, indispensable hops can be supported by source sentences.
2. Use canonical sentence IDs, not row ranks or RAG-internal IDs.
3. For every bridge, record the normalized entity and its adjacent hop numbers.
4. Record `intra_chunk`, `cross_chunk`, or `cross_document` for `chain_scope`.
5. Explain why each selected sentence is indispensable in `annotation_rationale`.

Copy the following JSON to your own completed annotation file and replace all
blank values. Do not consult the other annotator's package or any RAG output.

```json
{_draft_template(question)}
```
"""


def build_annotation_packages(
    corpus_manifest: dict[str, Any],
    question_split: dict[str, Any],
    stage_references: dict[int, list[str]],
    output_directory: str | Path,
    top_k: int,
) -> dict[str, Any]:
    """Write independent A/B Markdown packages and a machine-readable candidate index."""
    output = Path(output_directory)
    sentences = corpus_manifest["sentences"]
    retriever = LexicalSentenceRetriever(sentences)
    candidate_records: list[dict[str, Any]] = []
    package_entries: list[dict[str, Any]] = []
    for question in question_split["items"]:
        stage = int(question["stage"])
        source_index = int(question["source_index"])
        references = stage_references.get(stage, [])
        if source_index >= len(references):
            raise ValueError(f"Missing stage_ref for {question['question_id']}")
        stage_reference = references[source_index]
        candidates = retriever.retrieve(f"{question['question']}\n{stage_reference}", top_k)
        candidate_records.append(
            {
                "question_id": question["question_id"],
                "stage": stage,
                "question": question["question"],
                "stage_reference": stage_reference,
                "candidates": candidates,
            }
        )
        for annotator in ("A", "B"):
            path = output / f"annotator_{annotator}" / "packages" / f"{question['question_id']}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_markdown_package(annotator, question, stage_reference, candidates), encoding="utf-8")
        package_entries.append(
            {
                "question_id": question["question_id"],
                "stage": stage,
                "candidate_count": len(candidates),
                "annotator_a": str(Path("annotator_A") / "packages" / f"{question['question_id']}.md"),
                "annotator_b": str(Path("annotator_B") / "packages" / f"{question['question_id']}.md"),
            }
        )
    candidate_path = output / "candidate_index.jsonl"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for record in candidate_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    package_manifest = {
        "schema_version": 1,
        "candidate_retrieval": "deterministic_bm25_style_question_plus_stage_ref",
        "top_k": top_k,
        "question_count": len(package_entries),
        "packages": package_entries,
    }
    write_json(output / "package_manifest.json", package_manifest)
    return package_manifest


def generate_annotation_packages(
    manifest_path: str | Path,
    split_path: str | Path,
    questions_root: str | Path,
    output_directory: str | Path,
    top_k: int,
) -> dict[str, Any]:
    stage_references = {
        stage: read_json(Path(questions_root) / f"{stage}_stage_ref.json") for stage in (1, 2, 3)
    }
    return build_annotation_packages(
        read_json(manifest_path), read_json(split_path), stage_references, output_directory, top_k
    )
